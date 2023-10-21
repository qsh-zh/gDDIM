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

"""Training and evaluation"""

import run_lib
from absl import app
from absl import flags
from utils import Wandb
from ml_collections.config_flags import config_flags
import tensorflow as tf
import logging
import os

os.environ['JAM_PDB']='ipdb'

from jammy.utils.debug import decorate_exception_hook

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_bool("wandb", False, "launch wandb or not")
flags.DEFINE_enum("mode", None, ["train", "eval", "sampling", "fid", "check"], "Running mode: train or eval")
flags.DEFINE_string("eval_folder", "eval",
                    "The folder name for storing evaluation results")

flags.DEFINE_string("ckpt", None, "sampling mode ckpt")
flags.DEFINE_string("result_folder", None, "sampling result folder")
flags.mark_flags_as_required(["config", "mode"])

def resolve_result_folder(config, ckpt_path):
  sampler_name = config.sampling.method
  if sampler_name.lower() == 'deis':
    info_path=f"deis_order{config.sampling.deis_order}_nfe{config.sampling.nfe}_ts{config.sampling.ts_order}{'denoising' if config.sampling.noise_removal else ''}"
    return os.path.join(f"{ckpt_path}_eval", info_path)
  elif sampler_name.lower() == 'mldeis':
    info_path=f"mldeis_order{config.sampling.deis_order}_nfe{config.sampling.nfe}_ts{config.sampling.ts_order}{'denoising' if config.sampling.noise_removal else ''}"
    return os.path.join(f"{ckpt_path}_eval", info_path)
  elif sampler_name.lower() == 'hybdeis':
    info_path=f"hybdeis_order{config.sampling.deis_order}_nfe{config.sampling.nfe}_nfer{config.sampling.noise_nfe_ratio}_imgtr{config.sampling.img_t_ratio}_ts{config.sampling.ts_order}{'denoising' if config.sampling.noise_removal else ''}"
    return os.path.join(f"{ckpt_path}_eval", info_path)
  else:
    raise RuntimeError(f"{sampler_name} not supported")

@decorate_exception_hook
def main(argv):
  os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

  if FLAGS.mode == "train":
    tf.config.experimental.set_visible_devices([], "GPU")
    Wandb.name = f"{FLAGS.workdir[5:]}-{FLAGS.mode}"
    Wandb.ready_launch = FLAGS.wandb

    tf.config.experimental.set_visible_devices([], "GPU")
    # Create the working directory
    tf.io.gfile.makedirs(FLAGS.workdir)
    run_lib.launch_wandb(FLAGS.config,FLAGS.workdir)
    # Set logger so that it outputs to both console and file
    # Make logging work for both disk and Google Cloud Storage
    gfile_stream = tf.io.gfile.GFile(os.path.join(FLAGS.workdir, 'stdout.txt'), 'w')
    handler = logging.StreamHandler(gfile_stream)
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel('INFO')
    # Run the training pipeline
    run_lib.train(FLAGS.config, FLAGS.workdir)
  elif FLAGS.mode == "eval":
    # Run the evaluation pipeline
    run_lib.evaluate(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder)
  elif FLAGS.mode == "sampling":
    result_folder = FLAGS.result_folder if FLAGS.result_folder else resolve_result_folder(FLAGS.config, FLAGS.ckpt)
    run_lib.sample_data(FLAGS.config, FLAGS.ckpt, result_folder)
  elif FLAGS.mode == "fid":
    result_folder = FLAGS.result_folder if FLAGS.result_folder else resolve_result_folder(FLAGS.config, FLAGS.ckpt)
    run_lib.check_fid(FLAGS.config, result_folder)
  elif FLAGS.mode == "check":
    result_folder = FLAGS.result_folder if FLAGS.result_folder else resolve_result_folder(FLAGS.config, FLAGS.ckpt)
    run_lib.sample_data(FLAGS.config, FLAGS.ckpt, result_folder)
    run_lib.check_fid(FLAGS.config, result_folder)
  else:
    raise ValueError(f"Mode {FLAGS.mode} not recognized.")


if __name__ == "__main__":
  app.run(main)
