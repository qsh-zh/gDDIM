from functools import partial
import jax
import jax.numpy as jnp
from deis import multistep_ab_step
import flax
from jax import random, jit
from models import utils as mutils
import utils
from deis import get_ab_eps_coef, runge_kutta

def get_sampling_fn(config, sde, model, shape, inverse_scaler, is_p=True):
  """Create a sampling function.

  Args:
    config: A `ml_collections.ConfigDict` object that contains all configuration information.
    sde: A `sde_lib.SDE` object that represents the forward SDE.
    model: A `flax.linen.Module` object that represents the architecture of a time-dependent score-based model.
    shape: A sequence of integers representing the expected shape of a single sample.
    inverse_scaler: The inverse data normalizer function.
    eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.

  Returns:
    A function that takes random states and a replicated training state and outputs samples with the
      trailing dimensions matching `shape`.
  """
  del shape
  sampler_name = config.sampling.method
  data_shape = utils.get_data_shape(config)
  # Probability flow ODE sampling with black-box ODE solvers
  if sampler_name.lower() == 'order0':
    sampling_fn = get_order0_sampler(sde=sde,
                                  model=model,
                                  data_shape=data_shape,
                                  ts_order=config.sampling.ts_order,
                                  nfe=config.sampling.nfe,
                                  inverse_scaler=inverse_scaler,
                                  is_p=is_p)
  else:
      raise RuntimeError
  return sampling_fn

def get_rev_ts(sde, ts_order, num_step):
    rev_ts = jnp.power(
        jnp.linspace(
            jnp.power(sde.sampling_T, 1.0 / ts_order),
            jnp.power(sde.sampling_eps, 1.0 / ts_order),
            num_step + 1
        ),
        ts_order
    )
    return rev_ts

def get_order0_sampler(sde, model, data_shape, ts_order, nfe, inverse_scaler, is_p=False):
    rev_ts = get_rev_ts(sde, ts_order=ts_order, num_step  = nfe)
    def sampler(rng, state, batch_size, u=None):
        rng, step_rng = random.split(rng)
        if u is None:
            u = sde.prior_sampling(step_rng, (batch_size,) + data_shape)
        y_eps_fn = mutils.get_yeps_fn(sde, model, state.params_ema, state.model_state, train=False, continuous=True)
        _ones = jnp.ones(batch_size)

        def body_fn(i, val):
            cur_t,next_t = rev_ts[i], rev_ts[i+1]
            y_cur = val
            y_eps = y_eps_fn(y_cur, cur_t * _ones)

            y_mean_coef, y_std_coef = sde.y_mean_coef(cur_t * _ones), \
                                        sde.y_std_coef(cur_t * _ones) #(B, H, W, 1), (B, 1)
            y_0 = 1.0 / y_mean_coef * (y_cur - utils.batch_mul(y_std_coef, y_eps))

            y_next = sde.y_mean_coef(next_t * _ones) * y_0 + utils.batch_mul(
                sde.y_std_coef(next_t * _ones),
                y_eps
            )
            return y_next

        y0 = jax.lax.fori_loop(0, nfe, body_fn, u)
        x0 = sde.y2x(y0)
        x = inverse_scaler(x0)
        return x, nfe

    def psampler(prng, pstate, batch_size, u=None):
        rng = flax.jax_utils.unreplicate(prng)
        if u is None:
            # If not represent, sample the latent code from the prior distibution of the SDE.
            u = sde.prior_sampling(rng, (jax.local_device_count(),batch_size, ) + data_shape)
        xs, nfes = jax.pmap(sampler, static_broadcasted_argnums=2)(prng, pstate, batch_size, u)
        return xs, nfes[0]

    return psampler if is_p else sampler