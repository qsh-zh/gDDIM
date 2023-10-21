import jax
import matplotlib.pyplot as plt
from models import utils as mutils
import utils
import losses as losses_lib
import datasets
import numpy as np
from models import ncsnpp
import importlib

def load_ckpt(ckpt_filename, ftype="ddpmpp"):
    cfgs = {
        "ddpmpp": "configs.ddpmpp_cifar10_config",
        "deep": "configs.deep_cifar10_config",
        "ndeep": "configs.ndeep_cifar10_config",
        "celeba": "configs.ddpmpp_celeba_config",
    }
    configs = importlib.import_module(cfgs[ftype])
    config = configs.get_config()
    random_seed = 0
    rng = jax.random.PRNGKey(random_seed)
    rng, run_rng = jax.random.split(rng)
    rng, model_rng = jax.random.split(rng)
    score_model, init_model_state, initial_params = mutils.init_model(run_rng, config)
    optimizer = losses_lib.get_optimizer(config).create(initial_params)

    state = mutils.State(step=0, optimizer=optimizer, lr=config.optim.lr,
                        model_state=init_model_state,
                        ema_rate=config.model.ema_rate,
                        params_ema=initial_params,
                        rng=rng)  # pytype: disable=wrong-keyword-args
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)
    state = utils.load_training_state(ckpt_filename, state)
    return config, score_model, state, scaler, inverse_scaler


def show_samples(config, x, re_shift=True, fpng=None):
    def image_grid(x):
        size = config.data.image_size
        channels = config.data.num_channels
        img = x.reshape(-1, size, size, channels)
        w = int(np.sqrt(img.shape[0]))
        img = img.reshape((w, w, size, size, channels)).transpose((0, 2, 1, 3, 4)).reshape((w * size, w * size, channels))
        return img
    if re_shift:
        x = (x + 1.0) / 2
    img = image_grid(x)
    plt.figure(figsize=(8,8))
    plt.axis('off')
    plt.imshow(img)
    plt.show()
    if fpng:
        plt.savefig(fpng)
    plt.close("all")
