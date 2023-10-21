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

# Lint as: python3
"""Training NCSNv3 on CIFAR-10 with continuous sigmas."""

from configs.default_cifar10_config import get_default_configs


def get_config():
  config = get_default_configs()
  # training
  training = config.training
  training.continuous = True
  training.reduce_mean = True
  training.log_freq = 100
  training.eval_freq = 500
  training.n_jitted_steps = 100
  training.snapshot_freq_for_sampling = 1000
  config.training.batch_size = 32
  config.eval.batch_size = 32

  training.snapshot_freq = 10000
  training.snapshot_freq_for_preemption = 5000

  # data
  data = config.data
  data.centered = True
  data.num_channels = 3
  data.is_partial = True
  data.random_flip = False

  # model
  model = config.model
  model.name = 'ncsnpp'
  model.scale_by_sigma = False
  model.ema_rate = 0.5
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 64
  model.ch_mult = (1, 2, 2, 2)
  model.num_res_blocks = 4
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.conditional = True
  model.fir = False
  model.fir_kernel = [1, 3, 3, 1]
  model.skip_rescale = True
  model.resblock_type = 'biggan'
  model.progressive = 'none'
  model.progressive_input = 'none'
  model.progressive_combine = 'sum'
  model.attention_type = 'ddpm'
  model.init_scale = 0.
  model.embedding_type = 'positional'
  model.fourier_scale = 16
  model.conv_size = 3

  return config
