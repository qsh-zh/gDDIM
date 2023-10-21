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

# pylint: skip-file
"""Training and evaluation for score-based generative models. """

import gc
import io
import os
import time
from typing import Any, Dict
import tqdm

import flax
import flax.jax_utils as flax_utils
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_gan as tfgan
import logging
import functools
import wandb
from utils import Wandb
import json

from flax.metrics import tensorboard
from flax.training import checkpoints
# Keep the import below for registering all model definitions
from models import ncsnpp
import losses
import sampling
import utils
from models import utils as mutils
import datasets
# TODO: FIXME
# import likelihood
import sde_lib
from absl import flags

FLAGS = flags.FLAGS

def launch_wandb(config, workdir):
    meta_file = f"{workdir}/meta.json"
    run_id = None
    if os.path.exists(meta_file):
        with open(meta_file, 'r') as json_file:
            data = json.load(json_file)
            run_id = data["id"]
            logging.info(f"resume wandb from {run_id}")
        Wandb.launch(dict(config), id=run_id)
    else:
        Wandb.launch(dict(config))
        if Wandb.run:
            with open(meta_file, 'w') as json_file:
                json.dump(
                    {"id": Wandb.run.id},
                    json_file
                )

def prefix_info(info: Dict, prefix: str):
    return {f"{prefix}/{key}": value for key, value in info.items()}
    
def train(config, workdir):
  """Runs the training pipeline.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """

  # Create directories for experimental logs
  sample_dir = os.path.join(workdir, "samples")
  tf.io.gfile.makedirs(sample_dir)

  rng = jax.random.PRNGKey(config.seed)
  tb_dir = os.path.join(workdir, "tensorboard")
  tf.io.gfile.makedirs(tb_dir)
#   if jax.host_id() == 0:
#     writer = tensorboard.SummaryWriter(tb_dir)

  # Initialize model.
  rng, step_rng = jax.random.split(rng)
  score_model, init_model_state, initial_params = mutils.init_model(step_rng, config)
  optimizer = losses.get_optimizer(config).create(initial_params)
  state = mutils.State(step=0, optimizer=optimizer, lr=config.optim.lr,
                       model_state=init_model_state,
                       ema_rate=config.model.ema_rate,
                       params_ema=initial_params,
                       rng=rng)  # pytype: disable=wrong-keyword-args

  # Create checkpoints directory
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  # Intermediate checkpoints to resume training after pre-emption in cloud environments
  checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta")
  tf.io.gfile.makedirs(checkpoint_dir)
  tf.io.gfile.makedirs(checkpoint_meta_dir)
  # Resume training when intermediate checkpoints are detected
  state = checkpoints.restore_checkpoint(checkpoint_meta_dir, state)
  # `state.step` is JAX integer on the GPU/TPU devices
  initial_step = int(state.step)
  rng = state.rng

  # Build data iterators
  train_ds, eval_ds, _ = datasets.get_dataset(config,
                                              additional_dim=config.training.n_jitted_steps,
                                              uniform_dequantization=config.data.uniform_dequantization)
  train_iter = iter(train_ds)  # pytype: disable=wrong-arg-types
  eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Setup SDEs
  sde = sde_lib.from_config(config)
  if jax.host_id() == 0:
    logging.info("Initialized sde and dataset")

  # Build one-step training and evaluation functions
  optimize_fn = losses.optimization_manager(config)
  continuous = config.training.continuous
  reduce_mean = config.training.reduce_mean
  likelihood_weighting = config.training.likelihood_weighting
  train_step_fn = losses.get_step_fn(sde, score_model, train=True, optimize_fn=optimize_fn,
                                     reduce_mean=reduce_mean, continuous=continuous,
                                     likelihood_weighting=likelihood_weighting)
  # Pmap (and jit-compile) multiple training steps together for faster running
  p_train_step = jax.pmap(functools.partial(jax.lax.scan, train_step_fn), axis_name='batch', donate_argnums=1)
  eval_step_fn = losses.get_step_fn(sde, score_model, train=False, optimize_fn=optimize_fn,
                                    reduce_mean=reduce_mean, continuous=continuous,
                                    likelihood_weighting=likelihood_weighting)
  # Pmap (and jit-compile) multiple evaluation steps together for faster running
  p_eval_step = jax.pmap(functools.partial(jax.lax.scan, eval_step_fn), axis_name='batch', donate_argnums=1)

  # Building sampling functions
  if config.training.snapshot_sampling:
    sampling_shape = None # TODO:
    sampling_fn = sampling.get_sampling_fn(config, sde, score_model, sampling_shape, inverse_scaler)

  # Replicate the training state to run on multiple devices
  pstate = flax_utils.replicate(state)
  num_train_steps = config.training.n_iters

  # In case there are multiple hosts (e.g., TPU pods), only log to host 0
  if jax.host_id() == 0:
    logging.info("Starting training loop at step %d." % (initial_step,))
  rng = jax.random.fold_in(rng, jax.host_id())

  # JIT multiple training steps together for faster training
  n_jitted_steps = config.training.n_jitted_steps
  # Must be divisible by the number of steps jitted together
  assert config.training.log_freq % n_jitted_steps == 0 and \
         config.training.snapshot_freq_for_preemption % n_jitted_steps == 0 and \
         config.training.eval_freq % n_jitted_steps == 0 and \
         config.training.snapshot_freq % n_jitted_steps == 0, "Missing logs or checkpoints!"

  start_time = time.time()
  for step in range(initial_step, num_train_steps + 1, config.training.n_jitted_steps):
    # Convert data to JAX arrays and normalize them. Use ._numpy() to avoid copy.
    batch = jax.tree_map(lambda x: scaler(x._numpy()), next(train_iter))  # pylint: disable=protected-access
    rng, *next_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
    next_rng = jnp.asarray(next_rng)
    # Execute one training step
    (_, pstate), ptrain_loss_info = p_train_step((next_rng, pstate), batch)
    train_loss_info = jax.tree_map(
        jnp.mean, flax_utils.unreplicate(ptrain_loss_info)
    )
    # Log to console, file and tensorboard on host 0
    if jax.host_id() == 0 and step % config.training.log_freq == 0:
      log_str = "\t".join([f"{item:>9}{value:>9.4f}" for item, value in train_loss_info.items()])
      logging.info(f"{'Train step':>10}: {step:>9}" + '\t' + log_str)
    #   writer.scalar("training_loss", loss, step)
      train_log_info = prefix_info(jax.tree_util.tree_map(lambda item:item.item(), train_loss_info), "train")
      Wandb.log({
              **train_log_info,
              "step": step
          }
      )

    # Save a temporary checkpoint to resume training after pre-emption periodically
    if step != 0 and step % config.training.snapshot_freq_for_preemption == 0 and jax.host_id() == 0:
      saved_state = flax_utils.unreplicate(pstate)
      saved_state = saved_state.replace(rng=rng)
      checkpoints.save_checkpoint(checkpoint_meta_dir, saved_state,
                                  step=step // config.training.snapshot_freq_for_preemption,
                                  keep=1)

    # Report the loss on an evaluation dataset periodically
    if step % config.training.eval_freq == 0:
      eval_batch = jax.tree_map(lambda x: scaler(x._numpy()), next(eval_iter))  # pylint: disable=protected-access
      rng, *next_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
      next_rng = jnp.asarray(next_rng)
      (_, _), peval_loss_info = p_eval_step((next_rng, pstate), eval_batch)
      eval_loss_info = jax.tree_map(
        jnp.mean, flax_utils.unreplicate(peval_loss_info)
      )
      # Log to console, file and tensorboard on host 0
      if jax.host_id() == 0 and step % config.training.log_freq == 0:
        avg_time = (time.time() - start_time) / config.training.eval_freq
        start_time = time.time()
        log_str = "\t".join([f"{item:>9}{value:>9.4f}" for item, value in eval_loss_info.items()])
        logging.info(f"{'Eval step':>10}: {step:>9}" + '\t' + log_str + '\t' + f"avg t: {avg_time:>9.4f}")
        eval_log_info = prefix_info(jax.tree_util.tree_map(lambda item:item.item(), eval_loss_info), "eval")
        Wandb.log({
              **eval_log_info,
              "step": step,
              "avg_t": avg_time,
          }
        )

    # Save a checkpoint periodically and generate samples if needed
    if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
      # Save the checkpoint.
      if jax.host_id() == 0:
        saved_state = flax_utils.unreplicate(pstate)
        saved_state = saved_state.replace(rng=rng)
        checkpoints.save_checkpoint(checkpoint_dir, saved_state,
                                    step=step // config.training.snapshot_freq,
                                    keep=np.inf)

    # Generate and save samples
    if step % config.training.snapshot_freq_for_sampling == 0 or step == num_train_steps:
      if config.training.snapshot_sampling:
        rng, *sample_rng = jax.random.split(rng, jax.local_device_count() + 1)
        sample_rng = jnp.asarray(sample_rng)
        sample, n = sampling_fn(sample_rng, pstate, config.training.snapshot_sampling_batch // jax.local_device_count())
        this_sample_dir = os.path.join(
          sample_dir, "iter_{}_host_{}".format(step, jax.host_id()))
        tf.io.gfile.makedirs(this_sample_dir)
        sample_log_info = {"step": step}

        if "ps" in config.data.dataset:
          ps_sample = sample.reshape(-1, 2)
          with tf.io.gfile.GFile(
              os.path.join(this_sample_dir, "sample.np"), "wb") as fout:
            np.save(fout, ps_sample)
          with tf.io.gfile.GFile(
              os.path.join(this_sample_dir, "sample.png"), "wb") as fout:
            utils.save_ps_img(ps_sample, fout)
            sample_log_info["sample"] = wandb.Image(os.path.join(this_sample_dir, "sample.png"), caption=f"{step:09d}")
        else:
          image_grid = sample.reshape((-1, *sample.shape[2:]))
          nrow = int(np.sqrt(image_grid.shape[0]))
          sample = np.clip(sample * 255, 0, 255).astype(np.uint8)
          with tf.io.gfile.GFile(
              os.path.join(this_sample_dir, "sample.np"), "wb") as fout:
            np.save(fout, sample)
  
          with tf.io.gfile.GFile(
            os.path.join(this_sample_dir, "sample.png"), "wb") as fout:
            utils.save_image(image_grid, fout, nrow=nrow, padding=2)
            sample_log_info["img"] = wandb.Image(os.path.join(this_sample_dir, "sample.png"), caption=f"{step:09d}")
        Wandb.log(sample_log_info)


def sampling_from_fn(config, state, sampling_fn, result_folder):
  if jax.host_id() == 0:
    logging.critical(f"sample data for {result_folder}")
  tf.io.gfile.makedirs(result_folder)

  rng = jax.random.PRNGKey(config.seed + 1)

  # Create data normalizer and its inverse
  num_sampling_rounds = config.eval.num_samples // config.eval.batch_size + 1
  pstate = flax_utils.replicate(state)

  state = jax.device_put(state)
  for r in tqdm.tqdm(range(0, num_sampling_rounds)):
    rng, *sample_rng = jax.random.split(rng, jax.local_device_count() + 1)
    sample_rng = jnp.asarray(sample_rng)
    samples_org_x, samples_v, nfe_cnt = sampling_fn(sample_rng, pstate, config.eval.batch_size // jax.local_device_count())
    samples_x = np.clip(samples_org_x * 255., 0, 255).astype(np.uint8)
    samples_x = samples_x.reshape(
      (-1, config.data.image_size, config.data.image_size, config.data.num_channels))
        # Write samples to disk or Google Cloud Storage
    with tf.io.gfile.GFile(
        os.path.join(result_folder, f"samples_{r}.npz"), "wb") as fout:
      io_buffer = io.BytesIO()
      np.savez_compressed(io_buffer, samples=samples_x, nfe_cnt=nfe_cnt, samples_v=samples_v, samples_x=samples_org_x)
      fout.write(io_buffer.getvalue())
    gc.collect()


def sample_data(config,
                ckpt_file,
                result_folder):
  if jax.host_id() == 0:
    logging.critical(f"sample data for {result_folder}")
  tf.io.gfile.makedirs(result_folder)

  rng = jax.random.PRNGKey(config.seed + 1)

  # Create data normalizer and its inverse
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Initialize model
  rng, model_rng = jax.random.split(rng)
  score_model, init_model_state, initial_params = mutils.init_model(model_rng, config)
  optimizer = losses.get_optimizer(config).create(initial_params)
  state = mutils.State(step=0, optimizer=optimizer, lr=config.optim.lr,
                       model_state=init_model_state,
                       ema_rate=config.model.ema_rate,
                       params_ema=initial_params,
                       rng=rng)  # pytype: disable=wrong-keyword-args

  # Setup SDEs
  sde = sde_lib.from_config(config)
  if jax.host_id() == 0:
    logging.info("Initialized sde and state")

  sampling_fn = sampling.get_sampling_fn(config, sde, score_model, None, inverse_scaler)

  num_sampling_rounds = config.eval.num_samples // config.eval.batch_size + 1

  if os.path.exists(ckpt_file):
    state = checkpoints.restore_checkpoint(ckpt_file, state)
  else:
    raise RuntimeError(f"{ckpt_file} not exist")

  pstate = flax_utils.replicate(state)

  state = jax.device_put(state)
  for r in tqdm.tqdm(range(0, num_sampling_rounds)):
    rng, *sample_rng = jax.random.split(rng, jax.local_device_count() + 1)
    sample_rng = jnp.asarray(sample_rng)
    samples_org_x, nfe_cnt = sampling_fn(sample_rng, pstate, config.eval.batch_size // jax.local_device_count())
    samples_x = np.clip(samples_org_x * 255., 0, 255).astype(np.uint8)
    samples_x = samples_x.reshape(
      (-1, config.data.image_size, config.data.image_size, config.data.num_channels))
        # Write samples to disk or Google Cloud Storage
    with tf.io.gfile.GFile(
        os.path.join(result_folder, f"samples_{r}.npz"), "wb") as fout:
      io_buffer = io.BytesIO()
      np.savez_compressed(io_buffer, samples=samples_x, nfe_cnt=nfe_cnt)
      fout.write(io_buffer.getvalue())
    gc.collect()
  
#   check_fid(config, result_folder)

def check_fid(config, result_folder):
  if jax.host_id() == 0:
    logging.critical(f"fid for {result_folder}")
  import evaluation
  all_logits = []
  all_pools = []
  all_nfe_cnt = []

  inceptionv3 = config.data.image_size >= 256
  inception_model = evaluation.get_inception_model(inceptionv3=inceptionv3)

  num_sampling_rounds = config.eval.num_samples // config.eval.batch_size + 1
  for r in tqdm.tqdm(range(0, num_sampling_rounds)):
    with tf.io.gfile.GFile(os.path.join(result_folder, f"samples_{r}.npz"), "rb") as fin:
      info = np.load(fin)
      samples = info["samples"]
      cur_nfe = info["nfe_cnt"]

      latents = evaluation.run_inception_distributed(samples, inception_model,
                                                  inceptionv3=inceptionv3)

      # Save latent represents of the Inception network to disk or Google Cloud Storage
      with tf.io.gfile.GFile(os.path.join(result_folder, f"statistics_{r}.npz"), "wb") as fout:
          io_buffer = io.BytesIO()
          np.savez_compressed(
          io_buffer, pool_3=latents["pool_3"], logits=latents["logits"], nfe_cnt=cur_nfe)
          fout.write(io_buffer.getvalue())
      
      if not inceptionv3:
        all_logits.append(latents["logits"])
      all_pools.append(latents["pool_3"])
      all_nfe_cnt.append(cur_nfe)

  if not inceptionv3:
    all_logits = np.concatenate(
      all_logits, axis=0)[:config.eval.num_samples]
  all_pools = np.concatenate(all_pools, axis=0)[:config.eval.num_samples]

  # Load pre-computed dataset statistics.
  data_stats = evaluation.load_dataset_stats(config)
  data_pools = data_stats["pool_3"]

  # Compute FID/KID/IS on all samples together.
  if not inceptionv3:
    inception_score = tfgan.eval.classifier_score_from_logits(all_logits)
  else:
    inception_score = -1

  fid = tfgan.eval.frechet_classifier_distance_from_activations(data_pools, all_pools)

  avg_nfe_cnt = np.mean(all_nfe_cnt)
  logging.critical("%.6e inception_score: %.6e, FID: %.6e" % (avg_nfe_cnt,inception_score, fid))

  with tf.io.gfile.GFile(os.path.join(result_folder, f"report.npz"),"wb") as f:
    io_buffer = io.BytesIO()
    np.savez_compressed(io_buffer, IS=inception_score, fid=fid, nfe=avg_nfe_cnt)
    f.write(io_buffer.getvalue())