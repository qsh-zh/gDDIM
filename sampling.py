from functools import partial
import jax
import jax.numpy as jnp
from deis import multistep_ab_step
import flax
from jax import random, jit
from models import utils as mutils
from sde_lib import LambdaSDE, get_interp_fn, CLD
import utils
from deis import get_ab_eps_coef, runge_kutta

# def get_afterward_denoising_step(sde, model, t, denoising_eps, state):
#     cur_F, cur_G  = sde.s_F(t), sde.s_G(t)
#     eps_fn = mutils.get_eps_fn(sde, model, state.params_ema, state.model_state, train=False, continuous=True)
#     def step_fn(u):
#         _ones = jnp.ones(u.shape[0])
#         dt = denoising_eps * (-1)
#         cur_eps = eps_fn(u, _ones * t)
#         cur_score = sde.eps2score(cur_eps, _ones*t)
#         return u + utils.sbmm(cur_F, u) * dt - utils.sbmm(cur_G@cur_G, cur_score) * dt
#     return step_fn

def get_cld_denoising_step(sde, t, denoising_eps):
    cur_F = sde.s_F(t) # (2,2)
    def step_fn(u):
        return u + utils.sbmm(cur_F, u) * (-denoising_eps)
    return step_fn


def get_denoising_step(sde, model, t, denoising_eps):
    cur_F, cur_G  = sde.s_F(t), sde.s_G(t)
    def step_fn(state, u):
        eps_fn = mutils.get_eps_fn(sde, model, state.params_ema, state.model_state, train=False, continuous=True)
        _ones = jnp.ones(u.shape[0])
        dt = denoising_eps * (-1)
        cur_eps = eps_fn(u, _ones * t)
        cur_score = sde.eps2score(cur_eps, _ones*t)
        return u + utils.sbmm(cur_F, u) * dt - utils.sbmm(cur_G@cur_G, cur_score) * dt
    return step_fn

def get_sampling_fn(config, sde, model, shape, inverse_scaler):
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
                                  nfe=config.sampling.nfe,
                                  inverse_scaler=inverse_scaler,
                                  is_em=config.sampling.is_em,
                                  denoising=config.sampling.noise_removal,
                                  is_p=True)
  elif sampler_name.lower() == 'deis':
    sampling_fn = get_deis_sampler(sde=sde,
                                  model=model,
                                  data_shape=data_shape,
                                  nfe=config.sampling.nfe,
                                  inverse_scaler=inverse_scaler,
                                  deis_order=config.sampling.deis_order,
                                  ts_order=config.sampling.ts_order,
                                  denoising=config.sampling.noise_removal,
                                  is_p=True)
  elif sampler_name.lower() == 'sdeis':
    sampling_fn = get_sdeis_sampler(sde=sde,
                                  model=model,
                                  data_shape=data_shape,
                                  nfe=config.sampling.nfe,
                                  inverse_scaler=inverse_scaler,
                                  deis_order=config.sampling.deis_order,
                                  lambda_coef=config.sampling.lambda_coef,
                                  use_order0=config.sampling.sdeis_use_order0,
                                  ts_order=config.sampling.ts_order,
                                  denoising=config.sampling.noise_removal,
                                  is_p=True)
  elif sampler_name.lower() == 'ldeis':
    sampling_fn = get_L_deis_sampler(sde=sde,
                                  model=model,
                                  data_shape=data_shape,
                                  nfe=config.sampling.nfe,
                                  inverse_scaler=inverse_scaler,
                                  deis_order=config.sampling.deis_order,
                                  ts_order=config.sampling.ts_order,
                                  denoising=config.sampling.noise_removal,
                                  is_p=True)
  elif sampler_name.lower() == 'hybdeis':
    sampling_fn = get_hyd_deis_sampler(sde=sde,
                                  model=model,
                                  data_shape=data_shape,
                                  nfe=config.sampling.nfe,
                                  inverse_scaler=inverse_scaler,
                                  deis_order=config.sampling.deis_order,
                                  noise_nfe_ratio=config.sampling.noise_nfe_ratio,
                                  img_t_ratio=config.sampling.img_t_ratio,
                                  ts_order=config.sampling.ts_order,
                                  denoising=config.sampling.noise_removal,
                                  is_p=True)
  elif sampler_name.lower() == 'mldeis':
    sampling_fn = get_mldeis_sampler(sde=sde,
                                  model=model,
                                  data_shape=data_shape,
                                  nfe=config.sampling.nfe,
                                  inverse_scaler=inverse_scaler,
                                  deis_order=config.sampling.deis_order,
                                  ts_order=config.sampling.ts_order,
                                  denoising=config.sampling.noise_removal,
                                  is_p=True)
  elif sampler_name.lower() == 'ode':
    sampling_fn = get_ode_sampler(sde=sde,
                                  model=model,
                                  data_shape=data_shape,
                                  inverse_scaler=inverse_scaler,
                                  denoising=config.sampling.noise_removal,
                                  atol=config.sampling.atol,
                                  rtol=config.sampling.rtol,
                                  method=config.sampling.ode_method,
                                  is_p=True)
  elif sampler_name.lower() == "sscs":
    sampling_fn = get_sscs_sampler(sde=sde,
                                  model=model,
                                  data_shape=data_shape,
                                  nfe=config.sampling.nfe,
                                  inverse_scaler=inverse_scaler,
                                  ts_order=config.sampling.ts_order,
                                  denoising=config.sampling.noise_removal,
                                  is_p=True)
  elif sampler_name.lower() == "em":
    sampling_fn = get_em_sampler(sde=sde,
                                  model=model,
                                  data_shape=data_shape,
                                  nfe=config.sampling.nfe,
                                  inverse_scaler=inverse_scaler,
                                  lambda_coef = config.sampling.lambda_coef,
                                  ts_order=config.sampling.ts_order,
                                  denoising=config.sampling.noise_removal,
                                  is_p=True)
  else:
      raise RuntimeError
  return sampling_fn

def get_order0_sampler(sde, model, data_shape, nfe, inverse_scaler, is_em=False, denoising=False, is_p=False):
    num_step = nfe - 1 if denoising else nfe
    if denoising:
        step_fn = get_denoising_step(sde, model, sde.sampling_eps, sde.sampling_eps)
    else:
        step_fn = lambda _, u:u
    ts_order = 2
    rev_ts = jnp.power(
        jnp.linspace(
            jnp.power(sde.T, 1.0 / ts_order),
            jnp.power(sde.sampling_eps, 1.0 / ts_order),
            num_step + 1
        ),
        ts_order
    )
    if is_em:
        mean_matrix, eps_matrix = sde.prepare_naive_coef(rev_ts)
    else:
        mean_matrix, eps_matrix = sde.prepare_order0_coef(rev_ts)

    def sampler(rng, state, batch_size, u=None):
        rng, step_rng = random.split(rng)
        if u is None:
            u = sde.prior_sampling(step_rng, (batch_size,) + data_shape)
        eps_fn = mutils.get_eps_fn(sde, model, state.params_ema, state.model_state, train=False, continuous=True)
        _ones = jnp.ones(batch_size)

        def body_fn(i, val):
            cur_t = rev_ts[i]
            u_linear = utils.sbmm(mean_matrix[i], val)
            u_score = utils.sbmm(eps_matrix[i], eps_fn(val, _ones * cur_t))
            return u_linear + u_score
        u = jax.lax.fori_loop(0, num_step, body_fn, u)
        u = step_fn(state, u)
        x, v = u[...,0], u[...,1]
        x = inverse_scaler(x)
        return x, v, nfe

    def psampler(prng, pstate, batch_size, u=None):
        rng = flax.jax_utils.unreplicate(prng)
        if u is None:
            # If not represent, sample the latent code from the prior distibution of the SDE.
            u = sde.prior_sampling(rng, (jax.local_device_count(),batch_size, ) + data_shape)
        xs, vs, nfes = jax.pmap(sampler, static_broadcasted_argnums=2)(prng, pstate, batch_size, u)
        return xs, vs, nfes[0]

    return psampler if is_p else sampler

def _impl_deis_sampler(sde, model, data_shape, nfe, inverse_scaler,deis_order, rev_ts, denoising=False, is_p=False):
    num_step = nfe - 1 if denoising else nfe
    if denoising:
        step_fn = get_denoising_step(sde, model, sde.sampling_eps, sde.sampling_eps)
    else:
        step_fn = lambda _, u:u

    deis_coef = sde.get_deis_coef(deis_order, rev_ts)
    def sampler(rng, state, batch_size, u=None):
        rng, step_rng = random.split(rng)
        if u is None:
            u = sde.prior_sampling(step_rng, (batch_size,) + data_shape)
        eps_fn = mutils.get_eps_fn(sde, model, state.params_ema, state.model_state, train=False, continuous=True)
        _ones = jnp.ones(batch_size)

        def body_fn(i, val):
            cur_u, eps_pred = val
            cur_eps_pred = eps_fn(cur_u, _ones * rev_ts[i])
            return multistep_ab_step(
                cur_u, deis_coef[i], cur_eps_pred, eps_pred
            )
        eps_pred = jnp.stack([u,] * (deis_order+1))
        u, _ = jax.lax.fori_loop(0, num_step, body_fn, (u, eps_pred))
        u = step_fn(state, u)
        x, v = u[...,0], u[...,1]
        x = inverse_scaler(x)
        return x, v, nfe

    def psampler(prng, pstate, batch_size, u=None):
        rng = flax.jax_utils.unreplicate(prng)
        if u is None:
            u = sde.prior_sampling(rng, (jax.local_device_count(),batch_size, ) + data_shape)
        xs, vs, nfes = jax.pmap(sampler, static_broadcasted_argnums=2)(prng, pstate, batch_size, u)
        return xs, vs, nfes[0]

    return psampler if is_p else sampler

def get_rev_ts(sde, ts_order, num_step):
    return jnp.power(
        jnp.linspace(
            jnp.power(sde.T, 1.0 / ts_order),
            jnp.power(sde.sampling_eps, 1.0 / ts_order),
            num_step + 1
        ),
        ts_order
    )

def get_deis_sampler(sde, model, data_shape, nfe, inverse_scaler,deis_order, ts_order=2, denoising=False, is_p=False):
    rev_ts = get_rev_ts(sde, ts_order, nfe - 1 if denoising else nfe)
    return _impl_deis_sampler(sde, model, data_shape, nfe, inverse_scaler, deis_order, rev_ts, denoising, is_p)

def get_hyd_deis_sampler(
        sde, model, data_shape, nfe, inverse_scaler, 
        deis_order, noise_nfe_ratio=0.3, img_t_ratio=0.3, ts_order=2.0, denoising=False, is_p=False
    ):
    num_step = nfe - 1 if denoising else nfe

    mid_t = sde.T * img_t_ratio
    noise_nfe  =  int(num_step * noise_nfe_ratio)
    img_nfe = num_step - noise_nfe 
    noise_ts = jnp.linspace(sde.T, mid_t, noise_nfe, endpoint=False)
    img_ts = get_rev_ts(sde, ts_order, img_nfe)

    rev_ts = jnp.concatenate([noise_ts, img_ts])
    assert rev_ts.shape[0] == num_step + 1
    return _impl_deis_sampler(sde, model, data_shape, nfe, inverse_scaler, deis_order, rev_ts, denoising, is_p)


def get_ml_psi2_fn(sde):
    N = 100000
    fn = lambda psi2, cur_t: sde.s_inv_psi1(cur_t) @ sde.s_F2(cur_t) @ sde.s_psi1(cur_t) @ psi2

    def body_fn(carry, dt):
        prev_psi2, cur_t = carry
        new_psi2 = runge_kutta(prev_psi2, cur_t, dt, fn)
        new_t = cur_t + dt
        return (new_psi2, new_t), (prev_psi2, cur_t)

    init_carry = jnp.eye(2), 0.0
    _, (psi2s, psi2_ts) = jax.lax.scan(body_fn, init_carry, jnp.ones(N+1)* 1 / N, None)
    return get_interp_fn(psi2_ts, psi2s)
    
class MLCLD:
    def __init__(self, sde):
        assert isinstance(sde, CLD)
        assert sde.beta_1 == 0
        self.s_psi2_fn = get_ml_psi2_fn(sde)

        self.sampling_eps = sde.sampling_eps
        self.T = sde.T
        self.mixed_score = sde.mixed_score
        self.sde = sde
        self.s_G = sde.s_G
        # self.v_invR = sde.v_invR
        self.vv_psi = jit(jax.vmap(self.s_psi, in_axes=(0,0)))
        self.vs_psi = jit(jax.vmap(self.s_psi, in_axes=(0,None)))
        
        self.v_eps_integrand = jit(jax.vmap(self.s_eps_integrand))
        
    def y2x(self, y, t):
        psi1 = self.sde.s_psi1(t)
        return utils.sbmm(psi1, y)
    
    def x2y(self, x, t):
        inv_psi1 = self.sde.s_inv_psi1(t)
        return utils.sbmm(inv_psi1, x)
        
    @partial(jit, static_argnums=(0,))
    def s_psi(self, s, t):
        return self.s_psi2_fn(t) @ jnp.linalg.inv(self.s_psi2_fn(s))
    
    @partial(jit, static_argnums=(0,))
    def s_eps_integrand(self, s_t):
        cur_G = self.s_G(s_t)
        cur_inv_psi1 = self.sde.s_inv_psi1(s_t)
        
        integrand = 0.5 * cur_inv_psi1 @ cur_G @ cur_G.T @ self.sde.s_invR(s_t).T
        return integrand
    
    def get_deis_coef(self, order, rev_timesteps):
        x_coef = self.vv_psi(rev_timesteps[:-1], rev_timesteps[1:]) # [N, 2, 2]
        eps_coef = get_ab_eps_coef(self, order+1, rev_timesteps, order) # [N, order+1, 2, 2]
        return jnp.concatenate([x_coef[:, None], eps_coef], axis=1) # [N, order+2, 2,2]

def get_mldeis_sampler(sde, model, data_shape, nfe, inverse_scaler,deis_order, ts_order=2, denoising=False, is_p=False):
    num_step = nfe - 1 if denoising else nfe
    if denoising:
        step_fn = get_denoising_step(sde, model, sde.sampling_eps, sde.sampling_eps)
    else:
        step_fn = lambda _, u:u
    rev_ts = jnp.power(
        jnp.linspace(
            jnp.power(sde.T, 1.0 / ts_order),
            jnp.power(sde.sampling_eps, 1.0 / ts_order),
            num_step + 1
        ),
        ts_order
    )
    mlsde = MLCLD(sde)
    deis_coef = mlsde.get_deis_coef(deis_order, rev_ts)
    def sampler(rng, state, batch_size, u=None):
        rng, step_rng = random.split(rng)
        if u is None:
            u = sde.prior_sampling(step_rng, (batch_size,) + data_shape)
        u = mlsde.x2y(u, sde.T) # new

        _eps_fn = mutils.get_eps_fn(sde, model, state.params_ema, state.model_state, train=False, continuous=True)
        _ones = jnp.ones(batch_size)
        def eps_fn(y_u, _s_t): # new
            x_u = mlsde.y2x(y_u, _s_t)
            return _eps_fn(x_u, _s_t * _ones)

        def body_fn(i, val):
            cur_u, eps_pred = val
            cur_eps_pred = eps_fn(cur_u, rev_ts[i])
            return multistep_ab_step(
                cur_u, deis_coef[i], cur_eps_pred, eps_pred
            )
        eps_pred = jnp.stack([u,] * (deis_order+1))
        u, _ = jax.lax.fori_loop(0, num_step, body_fn, (u, eps_pred))
        u = step_fn(state, u)

        u = mlsde.y2x(u, sde.sampling_eps / 2)
        x, v = u[...,0], u[...,1]
        x = inverse_scaler(x)
        return x, v, nfe

    def psampler(prng, pstate, batch_size, u=None):
        rng = flax.jax_utils.unreplicate(prng)
        if u is None:
            u = sde.prior_sampling(rng, (jax.local_device_count(),batch_size, ) + data_shape)
        xs, vs, nfes = jax.pmap(sampler, static_broadcasted_argnums=2)(prng, pstate, batch_size, u)
        return xs, vs, nfes[0]

    return psampler if is_p else sampler

def _impl_sdeis_update_fn(sde, model, data_shape, nfe, inverse_scaler,deis_order, rev_ts, deis_coef, denoising=False, is_p=False):
    num_step = nfe - 1 if denoising else nfe
    if denoising:
        step_fn = get_denoising_step(sde, model, sde.sampling_eps, sde.sampling_eps)
    else:
        step_fn = lambda _, u:u
    def sampler(rng, state, batch_size, u=None):
        rng, step_rng = random.split(rng)
        if u is None:
            u = sde.prior_sampling(step_rng, (batch_size,) + data_shape)
        eps_fn = mutils.get_eps_fn(sde, model, state.params_ema, state.model_state, train=False, continuous=True)
        _ones = jnp.ones(batch_size)

        def body_fn(i, val):
            cur_u, eps_pred, rng = val
            rng, cur_rng = random.split(rng, 2)
            cur_eps_pred = eps_fn(cur_u, _ones * rev_ts[i])
            mean, eps_pred = multistep_ab_step(
                cur_u, deis_coef[i][:-1], cur_eps_pred, eps_pred
            ) # (B, D, 2)
            noise = random.multivariate_normal(cur_rng, jnp.zeros(2), deis_coef[i][-1], shape=mean.shape[:-1], method="svd")
            return mean + noise, eps_pred, rng
        eps_pred = jnp.stack([u,] * (deis_order+1))
        u, _, _ = jax.lax.fori_loop(0, num_step, body_fn, (u, eps_pred, rng))
        u = step_fn(state, u)
        x, v = u[...,0], u[...,1]
        x = inverse_scaler(x)
        return x, v, nfe

    def psampler(prng, pstate, batch_size, u=None):
        rng = flax.jax_utils.unreplicate(prng)
        if u is None:
            u = sde.prior_sampling(rng, (jax.local_device_count(),batch_size, ) + data_shape)
        xs, vs, nfes = jax.pmap(sampler, static_broadcasted_argnums=2)(prng, pstate, batch_size, u)
        return xs, vs, nfes[0]

    return psampler if is_p else sampler

def _impl_sdeis_sampler(sde, model, data_shape, nfe, inverse_scaler,deis_order, rev_ts, denoising=False, is_p=False):
    deis_coef = sde.get_deis_coef(deis_order, rev_ts)
    # ! avoid numerical error, if we have accurate integral, it should be zero
    deis_coef = deis_coef.at[-1,-1].set(0.0)
    return _impl_sdeis_update_fn(sde, model, data_shape, nfe, inverse_scaler,deis_order, rev_ts, deis_coef, denoising, is_p)

def get_sdeis_sampler(sde, model, data_shape, nfe, inverse_scaler, deis_order, lambda_coef=0, use_order0=True, ts_order=2, denoising=False, is_p=False):
    rev_ts = get_rev_ts(sde, ts_order, nfe - 1 if denoising else nfe)
    new_sde = LambdaSDE(sde, lambda_coef, use_order0)
    return _impl_sdeis_sampler(new_sde, model, data_shape, nfe, inverse_scaler, deis_order, rev_ts, denoising, is_p)

from models.utils import from_flattened_numpy, to_flattened_numpy
from scipy import integrate

def get_ode_sampler(sde, model, data_shape, inverse_scaler,denoising=False, rtol=1e-5, atol=1e-5, method='RK45', is_p=False):
    if denoising:
        step_fn = get_denoising_step(sde, model, sde.sampling_eps, sde.sampling_eps)
    else:
        step_fn = lambda _, u:u
    def sampler(rng, state, batch_size, u=None):
        d_shape = (batch_size, *data_shape, 2)
        rng, step_rng = random.split(rng)
        if u is None:
            u = sde.prior_sampling(step_rng, (batch_size,) + data_shape)
        score_fn = mutils.get_score_fn(sde, model, state.params_ema, state.model_state, train=False, continuous=True)
        _ones = jnp.ones(batch_size)
        @jit
        def drift_fn(x, s_t):
            vec_t = _ones * s_t
            score = score_fn(x, vec_t)
            cur_F, cur_G = sde.s_F(s_t), sde.s_G(s_t)
            grad = utils.sbmm(cur_F, x) - 0.5 * utils.sbmm(cur_G @ cur_G, score)
            return grad
        def ode_func(t, x):
            x = from_flattened_numpy(x, d_shape)
            drift = drift_fn(x, t)
            return to_flattened_numpy(drift)
        solution = integrate.solve_ivp(ode_func, (sde.T, sde.sampling_eps), to_flattened_numpy(u),
                                   rtol=rtol, atol=atol, method=method)
        nfe = solution.nfev
        u = jnp.asarray(solution.y[:, -1]).reshape(d_shape)
        u = step_fn(state, u)
        x, v = u[...,0], u[...,1]
        x = inverse_scaler(x)
        return x, v, nfe

    def psampler(prng, pstate, batch_size, u=None):
        rng = flax.jax_utils.unreplicate(prng)
        batch_size = batch_size // jax.local_device_count()
        d_shape = (jax.local_device_count(), batch_size, *data_shape, 2)
        rng, step_rng = random.split(rng)
        if u is None:
            u = sde.prior_sampling(step_rng, (jax.local_device_count(), batch_size,) + data_shape)
        _ones = jnp.ones(batch_size)
        
        @partial(jax.pmap, static_broadcasted_argnums=2)
        def drift_fn(state, x, s_t):
            score_fn = mutils.get_score_fn(sde, model, state.params_ema, state.model_state, train=False, continuous=True)
            vec_t = _ones * s_t
            score = score_fn(x, vec_t)
            cur_F, cur_G = sde.s_F(s_t), sde.s_G(s_t)
            grad = utils.sbmm(cur_F, x) - 0.5 * utils.sbmm(cur_G @ cur_G, score)
            return grad

        def ode_func(t, x):
            x = from_flattened_numpy(x, d_shape)
            drift = drift_fn(pstate, x, t)
            return to_flattened_numpy(drift)
        solution = integrate.solve_ivp(ode_func, (sde.T, sde.sampling_eps), to_flattened_numpy(u),
                                   rtol=rtol, atol=atol, method=method)
        nfe = solution.nfev
        u = jnp.asarray(solution.y[:, -1]).reshape(d_shape)
        # u = step_fn(state, u)
        x, v = u[...,0], u[...,1]
        x = inverse_scaler(x)
        return x, v, nfe

    return psampler if is_p else sampler

def _impl_Ldeis_sampler(sde, model, data_shape, nfe, inverse_scaler,deis_order, rev_ts, denoising=False, is_p=False):
    num_step = nfe - 1 if denoising else nfe
    if denoising:
        step_fn = get_denoising_step(sde, model, sde.sampling_eps, sde.sampling_eps)
    else:
        step_fn = lambda _, u:u

    deis_coef = sde.get_deis_coef(deis_order, rev_ts)
    def sampler(rng, state, batch_size, u=None):
        rng, step_rng = random.split(rng)
        if u is None:
            u = sde.prior_sampling(step_rng, (batch_size,) + data_shape)
        eps_fn = mutils.get_eps_fn(sde, model, state.params_ema, state.model_state, train=False, continuous=True)
        _ones = jnp.ones(batch_size)

        def body_fn(i, val):
            cur_u, eps_pred = val
            cur_eps_pred = eps_fn(cur_u, _ones * rev_ts[i])
            cur_eps_pred = sde.epsR2epsL(rev_ts[i], cur_eps_pred)
            return multistep_ab_step(
                cur_u, deis_coef[i], cur_eps_pred, eps_pred
            )
        eps_pred = jnp.stack([u,] * (deis_order+1))
        u, _ = jax.lax.fori_loop(0, num_step, body_fn, (u, eps_pred))
        u = step_fn(state, u)
        x, v = u[...,0], u[...,1]
        x = inverse_scaler(x)
        return x, v, nfe

    def psampler(prng, pstate, batch_size, u=None):
        rng = flax.jax_utils.unreplicate(prng)
        if u is None:
            u = sde.prior_sampling(rng, (jax.local_device_count(),batch_size, ) + data_shape)
        xs, vs, nfes = jax.pmap(sampler, static_broadcasted_argnums=2)(prng, pstate, batch_size, u)
        return xs, vs, nfes[0]

    return psampler if is_p else sampler

from sde_lib import LSDE

def get_L_deis_sampler(sde, model, data_shape, nfe, inverse_scaler,deis_order, ts_order=2, denoising=False, is_p=False):
    rev_ts = get_rev_ts(sde, ts_order, nfe - 1 if denoising else nfe)
    new_sde = LSDE(sde)
    return _impl_Ldeis_sampler(new_sde, model, data_shape, nfe, inverse_scaler, deis_order, rev_ts, denoising, is_p)

def get_sscs_ou_fn(sde):
    def _fn(rng, u, s_t, s_t_next):
        # due to the time convenction used in the paper, eq 2
        beta_int = sde.beta_int(1 - s_t_next) - sde.beta_int(1 - s_t)
        beta_int = -1 * beta_int

        coeff = jnp.exp(-2. * beta_int / sde.Gamma)
        mean_matrix = jnp.asarray(
            [
                [1 + 2 * beta_int / sde.Gamma, -4 * beta_int / sde.Gamma / sde.Gamma],
                [beta_int, 1 - 2 * beta_int / sde.Gamma]
            ]
        ) * coeff
        mean = utils.sbmm(mean_matrix, u)

        coeff = jnp.exp(-4 * beta_int/ sde.Gamma)
        cov_xx = jnp.exp(4 * beta_int / sde.Gamma) - 1 - 4 * beta_int / sde.Gamma - 8 * beta_int**2 / sde.Gamma / sde.Gamma
        cov_xv = -4 * beta_int**2 / sde.Gamma
        cov_vv = (sde.Gamma / 2)**2 * (jnp.exp(4 * beta_int / sde.Gamma) - 1) + beta_int * sde.Gamma - 2 * beta_int**2
        cov = jnp.asarray(
            [
                [cov_xx, cov_xv],
                [cov_xv, cov_vv]
            ]
        ) * coeff
        return mean + random.multivariate_normal(rng, jnp.zeros(2), cov, shape=mean.shape[:-1], method="svd")

    return _fn

def get_sscs_score_fn(sde, model, state):
    score_fn = mutils.get_score_fn(sde, model, state.params_ema, state.model_state, train=False, continuous=True)
    def _fn(u, s_t, s_t_next):
        x, v = u[...,0], u[...,1]
        v_score = score_fn(u, (sde.T - s_t) * jnp.ones(u.shape[0]))[...,1]
        v_dot = 2 * sde.beta(s_t) * sde.Gamma * (
            v_score + sde.m_inv * v
        )
        v = v + v_dot * (s_t_next - s_t)
        return jnp.stack([x, v], axis=-1)
    return _fn

def get_sscs_sampler(sde, model, data_shape, nfe, inverse_scaler, ts_order=2, denoising=False, is_p=False):
    # due to the time convenction used in the paper, eq 2
    rev_ts = get_rev_ts(sde, ts_order, nfe - 1 if denoising else nfe)
    ts = 1 - rev_ts
    num_step = nfe - 1 if denoising else nfe
    if denoising:
        step_fn = get_denoising_step(sde, model, sde.sampling_eps, sde.sampling_eps)
    else:
        step_fn = lambda _, u:u

    def sampler(rng, state, batch_size, u=None):
        rng, step_rng = random.split(rng)
        if u is None:
            u = sde.prior_sampling(step_rng, (batch_size,) + data_shape)
        sscs_ou_fn = get_sscs_ou_fn(sde)
        sscs_score_fn = get_sscs_score_fn(sde, model, state)
        def body_fn(i, val):
            cur_u, rng = val
            next_rng, rng1, rng2 = random.split(rng, 3)
            del val
            cur_t, next_t = ts[i], ts[i+1]
            cur_u = sscs_ou_fn(rng1, cur_u, cur_t, (cur_t + next_t) / 2.0)
            cur_u = sscs_score_fn(cur_u, cur_t, next_t)
            cur_u = sscs_ou_fn(rng2, cur_u, (cur_t + next_t) / 2.0, next_t)
            return cur_u, next_rng
        u, _ = jax.lax.fori_loop(0, num_step, body_fn, (u, rng))
        u = step_fn(state, u)
        x, v = u[...,0], u[...,1]
        x = inverse_scaler(x)
        return x, v, nfe


    def psampler(prng, pstate, batch_size, u=None):
        rng = flax.jax_utils.unreplicate(prng)
        if u is None:
            u = sde.prior_sampling(rng, (jax.local_device_count(),batch_size, ) + data_shape)
        xs, vs, nfes = jax.pmap(sampler, static_broadcasted_argnums=2)(prng, pstate, batch_size, u)
        return xs, vs, nfes[0]

    return psampler if is_p else sampler

def get_em_sampler(sde, model, data_shape, nfe, inverse_scaler, lambda_coef=0, ts_order=2, denoising=False, is_p=False):
    rev_ts = get_rev_ts(sde, ts_order, nfe - 1 if denoising else nfe)
    num_step = nfe - 1 if denoising else nfe
    if denoising:
        step_fn = get_denoising_step(sde, model, sde.sampling_eps, sde.sampling_eps)
    else:
        step_fn = lambda _, u:u

    def sampler(rng, state, batch_size, u=None):
        rng, step_rng = random.split(rng)
        if u is None:
            u = sde.prior_sampling(step_rng, (batch_size,) + data_shape)
        
        score_fn = mutils.get_score_fn(sde, model, state.params_ema, state.model_state, train=False, continuous=True)
        def body_fn(i, val):
            cur_u, rng = val
            next_rng, rng = random.split(rng, 2)
            cur_t, next_t = rev_ts[i], rev_ts[i+1]
            delta_t = next_t - cur_t
            cur_G = sde.s_G(cur_t)
            
            cur_score = score_fn(cur_u, cur_t * jnp.ones(cur_u.shape[0]))

            grad = utils.sbmm(sde.s_F(cur_t), cur_u) - (1.0 + lambda_coef) / 2.0 * \
                utils.sbmm(cur_G @ cur_G.T, cur_score)

            noise = random.normal(rng, cur_u.shape) * jnp.sqrt(jnp.abs(delta_t))

            next_u = cur_u + grad * delta_t + utils.sbmm(cur_G, noise) * lambda_coef
            return next_u, next_rng

        u, _ = jax.lax.fori_loop(0, num_step, body_fn, (u, rng))
        u = step_fn(state, u)
        x, v = u[...,0], u[...,1]
        x = inverse_scaler(x)
        return x, v, nfe


    def psampler(prng, pstate, batch_size, u=None):
        rng = flax.jax_utils.unreplicate(prng)
        if u is None:
            u = sde.prior_sampling(rng, (jax.local_device_count(),batch_size, ) + data_shape)
        xs, vs, nfes = jax.pmap(sampler, static_broadcasted_argnums=2)(prng, pstate, batch_size, u)
        return xs, vs, nfes[0]

    return psampler if is_p else sampler