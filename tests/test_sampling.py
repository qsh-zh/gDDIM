from jax import random
from tests.img_helper import load_ckpt, show_samples
from jammy.utils.git import git_rootdir
import sampling
import utils
from sde_lib import from_config

def test_sample_one_batch():
    fckpt = git_rootdir("logs/ddpmpp/checkpoints/checkpoint_4")
    init_rng = random.PRNGKey(0)
    config, score_model, state, scaler, inverse_scaler = load_ckpt(fckpt)
    sde = from_config(config)

    data_shape = utils.get_data_shape(config)

    deis_fn = sampling.get_deis_sampler(sde, score_model, data_shape, 20, inverse_scaler, deis_order=1, is_p=False)
    samples, _, _ = deis_fn(init_rng, state, 64)
    show_samples(config, samples, False, git_rootdir("data/deis_d1_n20_t2.png"))

    mldeis_fn = sampling.get_mldeis_sampler(sde, score_model, data_shape, 20, inverse_scaler, deis_order=1, is_p=False)
    samples, _, _ = mldeis_fn(init_rng, state, 64)
    show_samples(config, samples, False, git_rootdir("data/mldeis_d1_n20_t2.png"))


