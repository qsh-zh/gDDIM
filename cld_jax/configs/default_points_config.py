import ml_collections


def get_default_configs():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  config.training.batch_size = 1024
  training.n_iters = 200001
  training.snapshot_freq = 10000
  training.log_freq = 500
  training.eval_freq = 2000
  ## store additional checkpoints for preemption in cloud computing environments
  training.snapshot_freq_for_preemption = 20000
  ## produce samples at each snapshot.
  training.snapshot_sampling = True
  training.likelihood_weighting = False
  training.continuous = True
  training.n_jitted_steps = 50

  ## new
  training.snapshot_freq_for_sampling = 3000
  training.snapshot_sampling_batch=1000
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
  sampling.method = 'order0'
  sampling.nfe = 50
  sampling.is_em = False
  sampling.deis_order = 1
  sampling.ts_order = 2
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
  data.dataset = 'ps_olympic'
  data.centered = False
  data.uniform_dequantization = False
  data.dim = 2

  # model
  config.model = model = ml_collections.ConfigDict()
  model.name = 'ps_fmlp'
  model.nf = 256
  model.ema_rate = 0.99

  # new
  model.m_inv = 4.0
  model.beta_0 = 4.0
  model.beta_1 = 0.0
  model.vv_gamma = 0.04
  model.mixed_score = False
  model.is_R_rk = False
  model.R_dt = 1e-5
  model.used_cache = True

  # optimization
  config.optim = optim = ml_collections.ConfigDict()
  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 2e-3
  optim.beta1 = 0.9
  optim.eps = 1e-8
  optim.warmup = 5000
  optim.grad_clip = 1.

  config.seed = 42

  return config


def get_config():
    return get_default_configs()
