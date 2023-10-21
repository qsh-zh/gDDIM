import datasets
from configs import default_points_config as configs
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
import os
from models import utils as mutils
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

def test_dataset():
    config = configs.get_config()
    train_ds, _, _ = datasets.get_dataset(config, uniform_dequantization=False, evaluation=False)
    train_iter = iter(train_ds)
    train_batch = jnp.array((next(train_iter)["image"])).reshape(-1,2)
    plt.scatter(train_batch[:,0],train_batch[:,1],s=3)
    plt.savefig("logs/ps.png")

def test_model():
    config = configs.get_config()

    random_seed = 0 #@param {"type": "integer"}
    rng = jax.random.PRNGKey(random_seed)
    rng, run_rng = jax.random.split(rng)
    score_model, init_model_state, initial_params = mutils.init_model(run_rng, config)

    if "ps" in config.data.dataset:
        input_shape = (jax.local_device_count(), config.data.dim * 2)
    else:
        input_shape = (jax.local_device_count(), config.data.image_size, config.data.image_size, config.data.num_channels * 2)

    label_shape = input_shape[:1]
    fake_input = jnp.zeros(input_shape)
    fake_label = jnp.zeros(label_shape, dtype=jnp.int32)
    variables = {'params': initial_params, **init_model_state}
    score_model.apply(variables, fake_input, fake_label, train=False, mutable=False)