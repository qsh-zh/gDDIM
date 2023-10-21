import ml_collections


def get_default_configs():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  config.training.batch_size = 128
  training.n_iters = 1300001
  training.snapshot_freq = 50000
  training.log_freq = 50
  training.eval_freq = 100
  ## store additional checkpoints for preemption in cloud computing environments
  training.snapshot_freq_for_preemption = 10000
  ## produce samples at each snapshot.
  training.snapshot_sampling = True
  training.likelihood_weighting = False
  training.continuous = True
  training.n_jitted_steps = 5
  training.reduce_mean = True

  ## new
  training.snapshot_freq_for_preemption = 50000
  training.snapshot_sampling_batch=100
  training.continuous = True
  training.reduce_mean = True
  training.ema_update_freq = 1e9

  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.n_steps_each = 1
  sampling.noise_removal = True
  sampling.probability_flow = False
  sampling.snr = 0.16

  # new
  sampling.method = 'deis'
  sampling.nfe = 20
  sampling.is_em = False
  sampling.deis_order = 1
  sampling.ts_order = 2
  sampling.noise_nfe_ratio = 0.3
  sampling.img_t_ratio = 0.3
  sampling.atol = 1e-5
  sampling.rtol = 1e-5
  sampling.ode_method = 'RK45'
  sampling.lambda_coef = 1.0
  sampling.sdeis_use_order0 = True

  # evaluation
  config.eval = evaluate = ml_collections.ConfigDict()
  evaluate.begin_ckpt = 9
  evaluate.end_ckpt = 26
  evaluate.batch_size = 1024
  evaluate.enable_sampling = False
  evaluate.num_samples = 50000
  evaluate.enable_loss = True
  evaluate.enable_bpd = False
  evaluate.bpd_dataset = 'test'

  # data
  config.data = data = ml_collections.ConfigDict()
  data.dataset = 'CIFAR10'
  data.image_size = 32
  data.random_flip = True
  data.centered = False
  data.uniform_dequantization = False
  data.num_channels = 3

  # model
  config.model = model = ml_collections.ConfigDict()
  model.sigma_min = 0.01
  model.sigma_max = 50
  model.num_scales = 1000
  model.beta_min = 0.1
  model.beta_max = 20.
  model.dropout = 0.1
  model.embedding_type = 'fourier'

  # new
  model.m_inv = 4.0
  model.beta_0 = 4.0
  model.beta_1 = 0.0
  model.vv_gamma = 0.04
  model.mixed_score = False
  model.is_R_rk = False
  model.R_dt = 1e-5
  model.used_cache = True
  model.x64 = False

  # optimization
  config.optim = optim = ml_collections.ConfigDict()
  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 2e-4
  optim.beta1 = 0.9
  optim.eps = 1e-8
  optim.warmup = 5000
  optim.grad_clip = 1.

  config.seed = 42

  return config