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

"""All functions related to loss computation and optimization.
"""

from functools import partial
import flax
import jax
import jax.numpy as jnp
import jax.random as random
from models import utils as mutils


def get_optimizer(config):
  """Returns a flax optimizer object based on `config`."""
  if config.optim.optimizer == 'Adam':
    optimizer = flax.optim.Adam(beta1=config.optim.beta1, eps=config.optim.eps,
                                weight_decay=config.optim.weight_decay)
  else:
    raise NotImplementedError(
      f'Optimizer {config.optim.optimizer} not supported yet!')

  return optimizer


def optimization_manager(config):
  """Returns an optimize_fn based on `config`."""

  def optimize_fn(state,
                  grad,
                  warmup=config.optim.warmup,
                  grad_clip=config.optim.grad_clip):
    """Optimizes with warmup and gradient clipping (disabled if negative)."""
    lr = state.lr
    if warmup > 0:
      lr = lr * jnp.minimum(state.step / warmup, 1.0)
    if grad_clip >= 0:
      # Compute global gradient norm
      grad_norm = jnp.sqrt(
        sum([jnp.sum(jnp.square(x)) for x in jax.tree_leaves(grad)]))
      # Clip gradient
      clipped_grad = jax.tree_map(
        lambda x: x * grad_clip / jnp.maximum(grad_norm, grad_clip), grad)
    else:  # disabling gradient clipping if grad_clip < 0
      clipped_grad = grad
    return state.optimizer.apply_gradient(clipped_grad, learning_rate=lr)

  return optimize_fn


def get_sde_loss_fn(sde, model, train, reduce_mean=True, continuous=True, likelihood_weighting=True, t_eps=1e-5):
  """Create a loss function for training with arbirary SDEs.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A `flax.linen.Module` object that represents the architecture of the score-based model.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses
      according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
    eps: A `float` number. The smallest time step to sample from.

  Returns:
    A loss function.
  """
  reduce_op = jnp.mean if reduce_mean else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)
  def loss_fn(rng, params, states, batch):
    """Compute the loss function.

    Args:
      rng: A JAX random state.
      params: A dictionary that contains trainable parameters of the score-based model.
      states: A dictionary that contains mutable states of the score-based model.
      batch: A mini-batch of training data.

    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
      new_model_state: A dictionary that contains the mutated states of the score-based model.
    """

    eps_fn = mutils.get_eps_fn(sde, model, params, states, train=train, continuous=continuous, return_state=True)
    batch = batch["image"]
    v = jnp.zeros_like(batch)
    data =  jnp.stack([batch, v], axis=-1)

    rng, step_rng = random.split(rng)
    t = random.uniform(step_rng, (data.shape[0],), minval=t_eps, maxval=sde.T)
    rng, step_rng = random.split(rng)
    perturbed_data, _, z = sde.perturb_data(data, t, step_rng)
    rng, step_rng = random.split(rng)
    eps, new_model_state = eps_fn(perturbed_data, t, rng=step_rng)

    if not likelihood_weighting:
      assert eps.shape == z.shape
      losses = jnp.square(eps - z)
      losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1)
    else:
        raise RuntimeError
    #   g2 = sde.sde(jnp.zeros_like(data), t)[1] ** 2
    #   losses = jnp.square(score + batch_mul(z, 1. / std))
    #   losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1) * g2

    loss = jnp.mean(losses)
    return loss, ({
        "score_loss": loss,
    }, new_model_state)

  return loss_fn

def get_step_fn(sde, model, train, optimize_fn=None, reduce_mean=False, continuous=True, likelihood_weighting=False, is_p=True):
  """Create a one-step training/evaluation function.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A `flax.linen.Module` object that represents the architecture of the score-based model.
    train: `True` for training and `False` for evaluation.
    optimize_fn: An optimization function.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses according to
      https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.

  Returns:
    A one-step function for training or evaluation.
  """
  loss_fn = get_sde_loss_fn(sde, model, train, reduce_mean=reduce_mean,continuous=True, likelihood_weighting=likelihood_weighting)
  mean_op = partial(jax.lax.pmean, axis_name='batch') if is_p else lambda x:x

  def step_fn(carry_state, batch):
    """Running one step of training or evaluation.

    This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
    for faster execution.

    Args:
      carry_state: A tuple (JAX random state, `flax.struct.dataclass` containing the training state).
      batch: A mini-batch of training/evaluation data.

    Returns:
      new_carry_state: The updated tuple of `carry_state`.
      loss: The average loss value of this state.
    """

    (rng, state) = carry_state
    rng, step_rng = jax.random.split(rng)
    grad_fn = jax.value_and_grad(loss_fn, argnums=1, has_aux=True)
    if train:
      params = state.optimizer.target
      states = state.model_state
      (loss, (loss_info, new_model_state)), grad = grad_fn(step_rng, params, states, batch)
      grad = mean_op(grad)
      new_optimizer = optimize_fn(state, grad)
      new_params_ema = jax.tree_multimap(
        lambda p_ema, p: p_ema * state.ema_rate + p * (1. - state.ema_rate),
        state.params_ema, new_optimizer.target
      )
      step = state.step + 1
      new_state = state.replace(
        step=step,
        optimizer=new_optimizer,
        model_state=new_model_state,
        params_ema=new_params_ema
      )
    else:
      loss, (loss_info, _) = loss_fn(step_rng, state.params_ema, state.model_state, batch)
      new_state = state

    loss = mean_op(loss)
    loss_info = mean_op(loss_info)
    new_carry_state = (rng, new_state)
    return new_carry_state, loss_info

  return step_fn