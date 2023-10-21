# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""All functions and modules related to model definition.
"""
from typing import Any

import flax
import functools
import jax.numpy as jnp
import sde_lib
import jax
import numpy as np
from models import wideresnet_noise_conditional
from flax.training import checkpoints
from utils import bmm
from einops import rearrange

# The dataclass that stores all training states
@flax.struct.dataclass
class State:
  step: int
  optimizer: flax.optim.Optimizer
  lr: float
  model_state: Any
  ema_rate: float
  params_ema: Any
  rng: Any


_MODELS = {}


def register_model(cls=None, *, name=None):
  """A decorator for registering model classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _MODELS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _MODELS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def get_model(name):
  return _MODELS[name]


def get_sigmas(config):
  """Get sigmas --- the set of noise levels for SMLD from config files.
  Args:
    config: A ConfigDict object parsed from the config file
  Returns:
    sigmas: a jax numpy arrary of noise levels
  """
  sigmas = jnp.exp(
    jnp.linspace(
      jnp.log(config.model.sigma_max), jnp.log(config.model.sigma_min),
      config.model.num_scales))

  return sigmas



def init_model(rng, config):
  """ Initialize a `flax.linen.Module` model. """
  model_name = config.model.name
  model_def = functools.partial(get_model(model_name), config=config)
  if "ps" in config.data.dataset:
    input_shape = (jax.local_device_count(), config.data.dim)
  else:
    input_shape = (jax.local_device_count(), config.data.image_size, config.data.image_size, config.data.num_channels)
  label_shape = input_shape[:1]
  fake_input = jnp.zeros(input_shape)
  fake_label = jnp.zeros(label_shape, dtype=jnp.int32)
  params_rng, dropout_rng = jax.random.split(rng)
  model = model_def()
  variables = model.init({'params': params_rng, 'dropout': dropout_rng}, fake_input, fake_label)
  # Variables is a `flax.FrozenDict`. It is immutable and respects functional programming
  init_model_state, initial_params = variables.pop('params')
  return model, init_model_state, initial_params


def get_model_fn(model, params, states, train=False):
  """Create a function to give the output of the score-based model.

  Args:
    model: A `flax.linen.Module` object the represent the architecture of score-based model.
    params: A dictionary that contains all trainable parameters.
    states: A dictionary that contains all mutable states.
    train: `True` for training and `False` for evaluation.

  Returns:
    A model function.
  """

  def model_fn(x, labels, rng=None):
    """Compute the output of the score-based model.

    Args:
      x: A mini-batch of input data.
      labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
        for different models.
      rng: If present, it is the random state for dropout

    Returns:
      A tuple of (model output, new mutable states)
    """

    variables = {'params': params, **states}
    if not train:
      output = model.apply(variables, x, labels, train=False, mutable=False)
      return output, states
    else:
      rngs = {'dropout': rng}
      output, new_states = model.apply(variables, x, labels, train=True, mutable=list(states.keys()), rngs=rngs)
      return output, new_states

  return model_fn

def get_eps_fn(sde, model, params, states, train=False, continuous=False, return_state=False):
  model_fn = get_model_fn(model, params, states, train=train)

  def eps_fn(x, t, rng=None):
    labels = sde.encode_t(t)
    model_output, state = model_fn(sde.encode_x(x), labels, rng)
    x_eps = sde.model2eps(x, t, model_output)
    if return_state:
      return x_eps, state
    else:
      return x_eps

  return eps_fn

def get_yeps_fn(sde, model, params, states, train=False, continuous=False):
    xeps_fn = get_eps_fn(sde, model, params, states, train, continuous, return_state=False)
    def eps_fn(y, t, rng=None):
        xeps = xeps_fn(sde.y2x(y), t, rng)
        return sde.x2y(xeps)
    return eps_fn

def create_classifier(prng_key, batch_size, ckpt_path):
  """Create a noise-conditional image classifier.

  Args:
    prng_key: A JAX random state.
    batch_size: The batch size of input data.
    ckpt_path: The path to stored checkpoints for this classifier.

  Returns:
    classifier: A `flax.linen.Module` object that represents the architecture of the classifier.
    classifier_params: A dictionary that contains trainable parameters of the classifier.
  """
  input_shape = (batch_size, 32, 32, 3)
  classifier = wideresnet_noise_conditional.WideResnet(
    blocks_per_group=4,
    channel_multiplier=10,
    num_outputs=10
  )
  initial_variables = classifier.init({'params': prng_key, 'dropout': jax.random.PRNGKey(0)},
                                      jnp.ones(input_shape, dtype=jnp.float32),
                                      jnp.ones((batch_size,), dtype=jnp.float32), train=False)
  model_state, init_params = initial_variables.pop('params')
  classifier_params = checkpoints.restore_checkpoint(ckpt_path, init_params)
  return classifier, classifier_params


def get_logit_fn(classifier, classifier_params):
  """ Create a logit function for the classifier. """

  def preprocess(data):
    image_mean = jnp.asarray([[[0.49139968, 0.48215841, 0.44653091]]])
    image_std = jnp.asarray([[[0.24703223, 0.24348513, 0.26158784]]])
    return (data - image_mean[None, ...]) / image_std[None, ...]

  def logit_fn(data, ve_noise_scale):
    """Give the logits of the classifier.

    Args:
      data: A JAX array of the input.
      ve_noise_scale: time conditioning variables in the form of VE SDEs.

    Returns:
      logits: The logits given by the noise-conditional classifier.
    """
    data = preprocess(data)
    logits = classifier.apply({'params': classifier_params}, data, ve_noise_scale, train=False, mutable=False)
    return logits

  return logit_fn


def get_classifier_grad_fn(logit_fn):
  """Create the gradient function for the classifier in use of class-conditional sampling. """

  def grad_fn(data, ve_noise_scale, labels):
    def prob_fn(data):
      logits = logit_fn(data, ve_noise_scale)
      prob = jax.nn.log_softmax(logits, axis=-1)[jnp.arange(labels.shape[0]), labels].sum()
      return prob

    return jax.grad(prob_fn)(data)

  return grad_fn
