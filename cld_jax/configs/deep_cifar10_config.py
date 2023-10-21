from configs.default_cifar10_config import get_default_configs


def get_config():
  config = get_default_configs()
  # training
  training = config.training
  training.n_iters = 800001
  training.continuous = True
  training.reduce_mean = True
  training.log_freq = 100
  training.eval_freq = 2000
  training.snapshot_freq_for_sampling = 10000

  # data
  data = config.data
  data.centered = True

  # model
  model = config.model
  model.name = 'ncsnpp'
  model.scale_by_sigma = False
  model.ema_rate = 0.9999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 2, 2, 2)
  model.num_res_blocks = 8
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.conditional = True
  model.fir = True
  model.fir_kernel = [1, 3, 3, 1]
  model.skip_rescale = True
  model.resblock_type = 'biggan'
  model.progressive = 'none'
  model.progressive_input = 'residual'
  model.progressive_combine = 'sum'
  model.attention_type = 'ddpm'
  model.init_scale = 0.
  model.embedding_type = 'fourier'
  model.fourier_scale = 16
  model.conv_size = 3

  return config
