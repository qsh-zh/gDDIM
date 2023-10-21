from configs.deep_cifar10_config import get_config as old_get_config

def get_config():
    config = old_get_config()
    config.model.mixed_score = True
    config.model.is_R_rk = True
    config.model.R_dt = 1e-6
    return config