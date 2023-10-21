import jax
# jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax import vmap, jit
from jax._src.numpy.lax_numpy import _promote_dtypes_inexact

from deis import get_ab_eps_coef, runge_kutta

from jammy.utils.git import git_rootdir
import jammy.io as jio
import os.path as osp
from hashlib import sha1
import utils

@jit
def inv_2x2(matrix):
    a,b,c,d = matrix[0,0], matrix[0,1], matrix[1,0], matrix[1,1]
    coef = 1.0 / (a * d - b * c)
    return jnp.asarray(
        [
            [d, -b],
            [-c ,a ]
        ]
    ) * coef

@jit
def inv_2x2s(matrix):
    return vmap(inv_2x2)(matrix)

def get_interp_fn(_xp, _fp):
  @jax.jit
  def _fn(x):
      x, xp, fp = _promote_dtypes_inexact(x, _xp, _fp)

      i = jnp.clip(jnp.searchsorted(xp, x, side='right'), 1, len(xp) - 1)
      df = fp[i] - fp[i - 1]
      dx = xp[i] - xp[i - 1]
      delta = x - xp[i - 1]
      f = jnp.where((dx == 0), fp[i], fp[i - 1] + (delta / dx) * df)
      return f
  return _fn

class CLD:
    def __init__(
        self, m_inv=4.0, beta_0=4.0, beta_1=0.0, vv_gamma=0.04, 
        numerical_eps=1e-6, mixed_score=False, used_cache=True, x64=False, is_R_rk=False, R_dt = 1e-5,
    ):
        self.mixed_score = mixed_score
        self.used_cache = used_cache
        self.x64 = x64
        self.f_prefix = git_rootdir(f"data/cached/{'x64' if x64 else 'x32'}/")
        jio.mkdir(self.f_prefix)

        self.m_inv = m_inv
        self.Gamma = 2. / np.sqrt(m_inv)
        self.beta_0 = beta_0
        self.beta_1 = beta_1

        self.R_0 = jnp.asarray(
            [
                [jnp.sqrt(numerical_eps), 0],
                [0, jnp.sqrt(vv_gamma / self.m_inv + numerical_eps)]
            ]
        )
        
        # helper fns
        self.s_R = self._get_s_R_fn(is_R_rk, R_dt)
        self.v_R = jit(vmap(self.s_R))
        self.v_invR = jit(vmap(self.s_invR))
        self.v_cov = jit(vmap(self.s_cov))
        self.vv_psi = jit(jax.vmap(self.s_psi, in_axes=(0,0)))
        self.vs_psi = jit(jax.vmap(self.s_psi, in_axes=(0,None)))
        self.v_eps_integrand = jit(jax.vmap(self.s_eps_integrand))
        self.sampling_eps = 1e-3
        self.T = 1.0

    @partial(jit, static_argnums=(0,))
    def _beta(self, t):
        return self.beta_0 + self.beta_1 * t

    @partial(jit, static_argnums=(0,))
    def _beta_int(self, t):
        return self.beta_0 * t + 0.5 * self.beta_1 * t**2

    def beta(self, t):
        return self._beta(t)

    def beta_int(self, t):
        return self._beta_int(t)

    def _get_s_R_fn(self, is_rk, R_dt=1e-5):
        def _ode_fn(_x, _t):
            cur_F, cur_G = self.s_F(_t), self.s_G(_t)
            grad = cur_F @ _x + 0.5 * cur_G @ cur_G.T @ inv_2x2(_x).T
            return grad
        def scan_body_fn(carry, cur_t):
            if is_rk:
                new_r = runge_kutta(carry, cur_t, R_dt, _ode_fn)
                return new_r, carry
            else:
                # mid point integral
                cur_F = ( self.s_F(cur_t) + self.s_F(cur_t + R_dt)) / 2.0
                cur_G = ( self.s_G(cur_t) + self.s_G(cur_t + R_dt)) / 2.0
                new_r = carry + R_dt * (cur_F @ carry + 0.5 * cur_G @ cur_G @ jnp.linalg.inv(carry).T)
                return new_r, carry

        f_path = osp.join(self.f_prefix, f"{'rk' if is_rk else 'euler'}_rs_{int(1.0 / R_dt)}.pkl")
        if self.used_cache and osp.exists(f_path):
            Rs = jnp.asarray(jio.load(f_path))
        else:
            assert jax.config.read("jax_enable_x64") == self.x64
            _, Rs = jax.lax.scan(scan_body_fn, self.R_0, jnp.linspace(0, 1.0 + R_dt, int(1.0 / R_dt) + 1, endpoint=False))
            jio.dump(f_path, Rs)
        ts = jnp.linspace(0, 1.0 + R_dt, int(1.0 / R_dt) + 1, endpoint=False)
        idx = jnp.linspace(0, Rs.shape[0]-1, 100_000, dtype=int) # 100_000 is accurate enough
        return get_interp_fn(ts[idx], Rs[idx])

    @partial(jit, static_argnums=(0,))
    def s_f1_psi(self, s, t):
        # system read
        # \dot{x} = F_1 x + F_2 x - \frac{1}{2}G_t G_t^T R^{-T} NN
        # F_1 = {{0, \[Beta]/M}, {-\[Beta], 0}}
        # F_2 = {{0, 0}, {0, -\[Beta]/M}}
        # expm{-F_1} = {{Cos[-\[Beta]t/Sqrt[M]], 1/ Sqrt[M] Sin[-\[Beta]t/Sqrt[M]]}, {-Sqrt[M] Sin[-\[Beta]t/Sqrt[M]], Cos[-\[Beta]t/Sqrt[M]]}}
        # return expm{-\int_s^t F_1 dt}
        # y = this_term @ x
        beta_int = self._beta_int(t) - self._beta_int(s)
        sqrt_m = jnp.sqrt(1.0 / self.m_inv)
        inv_sqrt_m = jnp.sqrt(self.m_inv)
        # return jnp.asarray(
        #     [
        #         [jnp.cos(-beta_int * inv_sqrt_m), inv_sqrt_m * jnp.sin(-beta_int * inv_sqrt_m)],
        #         [-sqrt_m * jnp.sin(-beta_int * inv_sqrt_m), jnp.cos(-beta_int * inv_sqrt_m)]
        #     ]
        # )
        return jnp.asarray(
            [
                [jnp.cos(beta_int * inv_sqrt_m), inv_sqrt_m * jnp.sin(beta_int * inv_sqrt_m)],
                [-sqrt_m * jnp.sin(beta_int * inv_sqrt_m), jnp.cos(beta_int * inv_sqrt_m)]
            ]
        )

    @partial(jit, static_argnums=(0,))
    def s_psi1(self, t):
        # return expm{\int_s^t F_1 dt}
        # x = this_term @ y
        return self.s_f1_psi(0, t)

    @partial(jit, static_argnums=(0,))
    def s_inv_psi1(self, t):
        # return expm{\int_s^t F_1 dt}
        # x = this_term @ y
        return self.s_f1_psi(t, 0)

    @partial(jit, static_argnums=(0,))
    def s_F1(self, t):
        # [[0 , beta * m_inv],
        #  [-beta, 0 ]]
        cur_beta = self._beta(t)
        return jnp.asarray(
            [
                [0.0, cur_beta * self.m_inv],
                [-cur_beta, 0.0]
            ]
        )
    @partial(jit, static_argnums=(0,))
    def s_F2(self, t):
        # [[0 , 0],
        #  [0,  -Gamma * beta * m_inv]]
        cur_beta = self._beta(t)
        return jnp.asarray(
            [
                [0.0, 0.0],
                [0.0, -self.Gamma * cur_beta * self.m_inv]
            ]
        )



    @partial(jit, static_argnums=(0,))
    def s_psi(self, s, t):
        beta_int = self._beta_int(t) - self._beta_int(s)
        # F_multi_delta = jnp.asarray(
        #     [
        #         [0.0, beta_int * self.m_inv],
        #         [-beta_int, -self.Gamma * beta_int * self.m_inv]
        #     ]
        # )
        # MatrixExp[{{0, a*a / 4 * t}, {-1*t, -a*t}}]
        # {
        # {E^(-((a t)/2)) (1 + (a t)/2), 1/4 a^2 E^(-((a t)/2)) t}, 
        # {-E^(-((a t)/2)) t, E^(-((a t)/2)) (1 - (a t)/2)}
        # }

        a = 2*jnp.sqrt(self.m_inv)
        t = beta_int
        coef = jnp.exp(- a * t / 2) # E^(-((a t)/2))
        return jnp.asarray(
            [
                [1 + a * t / 2, 0.25 * a * a * t],
                [-t , 1 - a * t / 2]
            ]
        ) * coef

    @partial(jit, static_argnums=(0,))
    def s_eps_integrand(self, s_t):
        cur_G = self.s_G(s_t)
        
        integrand = 0.5 * cur_G @ cur_G @ inv_2x2(self.s_R(s_t)).T
        return integrand

    @partial(jit, static_argnums=(0,))
    def s_F(self, t):
        # [[0 , beta * m_inv],
        #  [-beta, -Gamma * beta * m_inv]]
        cur_beta = self._beta(t)
        return jnp.asarray(
            [
                [0.0, cur_beta * self.m_inv],
                [-cur_beta, -self.Gamma * cur_beta * self.m_inv]
            ]
        )

    @partial(jit, static_argnums=(0,))
    def s_G(self, t):
        cur_beta = self._beta(t)
        return jnp.asarray(
            [
                [0. , 0.],
                [0. , jnp.sqrt(2 * self.Gamma * cur_beta)]
            ]
        )

    @partial(jit, static_argnums=(0,))
    def s_invR(self, t):
        cur_R = self.s_R(t)
        return inv_2x2(cur_R)

    @partial(jit, static_argnums=(0,))
    def s_cov(self, t):
        cur_R = self.s_R(t)
        return cur_R @ cur_R.T

    @partial(jit, static_argnums=(0,))
    def eps2score(self, eps, ts):
        # eps (B, ... c, 2,)
        # score = -R^{-T}@eps
        rs = self.v_R(ts) # (B, 2, 2)
        inv_rs = inv_2x2s(rs)
        score = jnp.einsum("bji,b...dj->b...di", -inv_rs, eps) # -R^{-T}@eps
        return score

    @partial(jit, static_argnums=(0,))
    def mean(self, batch, ts):
        # batch (B, ..., d, 2)
        psis = vmap(self.s_psi, in_axes=(None,0))(0., ts) # (B, 2, 2)
        return jnp.einsum("bij,b...dj->b...di", psis, batch)

    @partial(jit, static_argnums=(0,))
    def perturb_data(self, batch, ts, rng):
        mean = self.mean(batch, ts) # (B, ..., d, 2)
        rs = self.v_R(ts) # (B, 2, 2)
        raw_noise = jax.random.normal(rng, mean.shape) # (B, ..., d, 2)
        perb_noise = jnp.einsum("bij,b...dj->b...di", rs, raw_noise)
        perturbed_data = mean + perb_noise
        return perturbed_data, mean, raw_noise

    def prior_sampling(self, rng, shape):
        x_rng, v_rng = jax.random.split(rng, 2)
        xs = jax.random.normal(x_rng, shape=shape)
        vs = jax.random.normal(v_rng, shape=shape) / jnp.sqrt(self.m_inv)
        return jnp.stack([xs, vs], axis=-1)

    def prepare_naive_coef(self, rev_ts):
        def s_navie_psi(cur_t, next_t):
            return jnp.eye(2) + self.s_F(cur_t) * (next_t - cur_t)

        def s_naive_eps_mat(cur_t, next_t):
            G = self.s_G(cur_t)
            return 0.5 * G @ G @ inv_2x2(self.s_R(cur_t)).T * (next_t - cur_t)

        mean_matrix = jax.vmap(s_navie_psi, in_axes=(0,0))(rev_ts[:-1], rev_ts[1:])
        eps_matrix = jax.vmap(s_naive_eps_mat)(rev_ts[:-1], rev_ts[1:])
        return mean_matrix, eps_matrix


    def prepare_order0_coef(self, rev_ts):
        def s_eps_fn(cur_t, next_t, num_item=1000):
            dt = (next_t - cur_t) / num_item
            t_inter = jnp.linspace(cur_t, next_t, num_item, endpoint=False)
            
            psi_coef = self.vs_psi(t_inter, next_t)
            integrand = self.v_eps_integrand(t_inter)
            return jnp.einsum("bij,bjk->bik", psi_coef, integrand * dt).sum(axis=0)

        @jit
        def vec_eps_fn(cur_ts, next_ts, num_item=1000):
            return jax.vmap(
                s_eps_fn, in_axes=(0, 0, None)
            )(cur_ts, next_ts, num_item)

        mean_matrix = self.vv_psi(rev_ts[:-1], rev_ts[1:])
        eps_matrix = vec_eps_fn(rev_ts[:-1], rev_ts[1:])
        return mean_matrix, eps_matrix

    def get_deis_coef(self, order, rev_timesteps, used_cache=True):
        hexinfo = sha1(np.asarray(rev_timesteps)).hexdigest()
        fpath = osp.join(self.f_prefix, f"cld_deis_coef_oder{order}_TLen_{len(rev_timesteps)}_{hexinfo}.pkl")
        if self.used_cache and osp.exists(fpath) and used_cache:
            return jnp.asarray(jio.load(fpath))
        # return [x_coef, eps_coef]
        assert jax.config.read("jax_enable_x64") == self.x64
        x_coef = self.vv_psi(rev_timesteps[:-1], rev_timesteps[1:]) # [N, 2, 2]
        eps_coef = get_ab_eps_coef(self, order+1, rev_timesteps, order) # [N, order+1, 2, 2]
        rtn = jnp.concatenate([x_coef[:, None], eps_coef], axis=1) # [N, order+2, 2,2]
        jio.dump(fpath, rtn)
        return rtn

def from_config(config):
    return CLD(m_inv=config.model.m_inv, 
                beta_0=config.model.beta_0, 
                beta_1=config.model.beta_1, 
                vv_gamma=config.model.vv_gamma,
                mixed_score=config.model.mixed_score,
                is_R_rk=config.model.is_R_rk,
                used_cache=config.model.used_cache,
                R_dt = config.model.R_dt,
                x64 = config.model.x64,
                )


class LambdaSDE:
    def __init__(self, sde, lambda_coef=0.1, use_order0=True, used_cache=True):
        self.sde = sde
        self.mixed_score = sde.mixed_score
        self.prior_sampling = sde.prior_sampling
        self.v_invR = sde.v_invR
        self.used_cache = sde.used_cache and used_cache
        self.x64 = sde.x64

        self.use_order0 = use_order0
        self.lambda_coef = lambda_coef
        self.f_prefix = f"{sde.f_prefix}/{lambda_coef}"
        jio.mkdir(self.f_prefix)

        self.s_hat_psi_02t = self._get_hat_psi_02t()

    @partial(jit, static_argnums=(0,))
    def s_hat_F(self, t):
        # \hat{\mF}_\tau = \mF_\tau + \frac{1+\lambda^2}{2}\mG_\tau \mG_\tau^T\Sigma_\tau^{-1}
        cur_G  = self.sde.s_G(t)
        inv_cov = inv_2x2(self.sde.s_cov(t))
        return self.sde.s_F(t) + 0.5 * (1 + self.lambda_coef**2) * cur_G @ cur_G.T @ inv_cov

    def _get_hat_psi_02t(self, used_cache=True):
        dt = 1e-5
        def _ode_fn(_x, _t):
            # inv_cov = inv_2x2(self.sde.s_cov(_t))
            # cur_F, cur_G = self.sde.s_F(_t), self.sde.s_G(_t)
            # grad = cur_F @ _x + 0.5 * (1 + self.lambda_coef**2) * cur_G @ cur_G.T @ inv_cov @ _x
            # return grad
            return self.s_hat_F(_t) @ _x
        def scan_body_fn(carry, cur_t):
            new_r = runge_kutta(carry, cur_t, dt, _ode_fn)
            return new_r, carry
        f_path = osp.join(self.f_prefix, f"02t_hat_psi_{int(1.0 / dt)}.pkl")
        if self.used_cache and osp.exists(f_path) and used_cache:
            Rs = jnp.asarray(jio.load(f_path))
        else:
            _, Rs = jax.lax.scan(scan_body_fn, jnp.eye(2), jnp.linspace(0, 1.0 +dt, int(1.0 /dt) + 1, endpoint=False))
            jio.dump(f_path, Rs)
        ts = jnp.linspace(0, 1.0 + dt, int(1.0 / dt) + 1, endpoint=False)
        return get_interp_fn(ts, Rs)

    @partial(jit, static_argnums=(0,))
    def s_hat_psi(self, s ,t):
        return self.s_hat_psi_02t(t) @ inv_2x2(self.s_hat_psi_02t(s))

    @partial(jit, static_argnums=(0,))
    def cond_rev_cov(self, s, t):
        # \frac{d \mP_{s\tau}}{d\tau} = \hat{\mF}_\tau \mP_{s\tau} + \mP_{s\tau} \hat{\mF}_\tau^T + \lambda^2 \mG_\tau \mG_\tau^T \quad \mP_{ss} = 0
        dir_sign = jax.lax.cond(t > s, lambda _: 1, lambda _: -1, None)
        n_step = 10_000
        dt = (t - s) * 1.0 / n_step
        ts = jnp.linspace(s, t, n_step + 1, endpoint=False)
        def _ode_fn(_x, _t):
            cur_hat_F = self.s_hat_F(_t)
            cur_G = self.sde.s_G(_t)
            #! we run backward, make it negetive
            grad = cur_hat_F @ _x + _x @ cur_hat_F + dir_sign * self.lambda_coef**2 * cur_G @ cur_G.T
            return grad
        def body_fn(i, val):
            cur_t = ts[i]
            new_val = runge_kutta(val, cur_t, dt, _ode_fn)
            return new_val
        cov = jax.lax.fori_loop(0, n_step, body_fn, jnp.zeros((2,2)))
        return cov
    
    def update_coef(self, s, t):
        x_coef = self.sde.s_psi(s, t) #[2, 2]
        eps_coef = (
            self.s_hat_psi(s, t) - x_coef
        ) @ self.sde.s_R(s) # [2, 2]
        cov_coef = self.cond_rev_cov(s, t) # [2, 2]
        return jnp.stack([x_coef, eps_coef, cov_coef])

    def get_poly_eps_coef(self, order, rev_timesteps):
        def s_eps_integrand(_t):
            cur_G = self.sde.s_G(_t)
            inv_cov = inv_2x2(self.sde.s_cov(_t))
            return 0.5 * (1 + self.lambda_coef**2) * cur_G @ cur_G.T @ inv_cov @ self.sde.s_psi(0, _t)
        class _sde:
            def vs_psi(ss, t):
                return jit(
                    vmap(self.s_hat_psi, (0, None))
                )(ss, t)
            def v_eps_integrand(ts):
                return jit(
                    vmap(s_eps_integrand)
                )(ts)
        ab_eps_coef = get_ab_eps_coef(_sde, order+1, rev_timesteps, order) # [N, order+1, 2, 2]

        @jit
        def s_last_term(s):
            return self.sde.s_psi(s, 0) @ self.sde.s_R(s)
        last_term = jit(
            vmap(s_last_term)
        )(rev_timesteps[:-1]) # [N, 2, 2]
        return jnp.einsum(
            "b...ij,bjk->b...ik", ab_eps_coef, last_term
        )
    
    def get_deis_coef(self, order, rev_timesteps, used_cache=True):
        if self.use_order0 and order == 0:
            order0_coef = self.get_order0_coef(rev_timesteps, used_cache=True)
            aug_order0_coef = jnp.stack(
                [
                    order0_coef[:,0], order0_coef[:,1], jnp.zeros(order0_coef[:,0].shape), order0_coef[:,2]
                ],
                axis=1
            )
            return aug_order0_coef
        hexinfo = sha1(np.asarray(rev_timesteps)).hexdigest()
        fpath = osp.join(self.f_prefix, f"cld_deis_coef_oder{order}_TLen_{len(rev_timesteps)}_{hexinfo}.pkl")
        if self.used_cache and osp.exists(fpath) and used_cache:
            return jnp.asarray(jio.load(fpath))
        assert jax.config.read("jax_enable_x64") == self.x64
        x_coef = self.sde.vv_psi(rev_timesteps[:-1], rev_timesteps[1:]) # [N, 2, 2]
        eps_coef = self.get_poly_eps_coef(order, rev_timesteps) # [N, order+1, 2, 2]
        covs = vmap(self.cond_rev_cov)(rev_timesteps[:-1], rev_timesteps[1:]) # [N, 2, 2]
        rtn = jnp.concatenate([x_coef[:,None], eps_coef, covs[:,None]], axis=1) # [N, order+3, 2,2]
        jio.dump(fpath, rtn)
        return rtn


    def get_order0_coef(self, rev_timesteps, used_cache=True):
        hexinfo = sha1(np.asarray(rev_timesteps)).hexdigest()
        fpath = osp.join(self.f_prefix, f"order0_coef_TLen_{len(rev_timesteps)}_{hexinfo}.pkl")
        if self.used_cache and osp.exists(fpath) and used_cache:
            return jnp.asarray(jio.load(fpath))
        assert jax.config.read("jax_enable_x64") == self.x64
        coef = vmap(self.update_coef)(rev_timesteps[:-1], rev_timesteps[1:]) #[N, 3, 2, ,2]
        jio.dump(fpath, coef)
        return coef


class LSDE:
    def __init__(self, sde, used_cache=True):
        self.sde = sde
        self.mixed_score = sde.mixed_score
        self.prior_sampling = sde.prior_sampling
        self.v_invR = sde.v_invR
        self.s_G = sde.s_G
        self.used_cache = sde.used_cache and used_cache
        self.x64 = sde.x64

        self.vs_psi = sde.vs_psi
        self.vv_psi = sde.vv_psi
        self.v_eps_integrand = jit(vmap(
            self.s_eps_integrand
        ))

        self.f_prefix = f"{sde.f_prefix}/L_SDE"
        jio.mkdir(self.f_prefix)

    @partial(jit, static_argnums=(0,))
    def s_L(self, s_t):
        cur_cov = self.sde.s_cov(s_t)
        return jnp.linalg.cholesky(cur_cov)

    @partial(jit, static_argnums=(0,))
    def epsR2epsL(self, s_t, eps):
        # L^{T} R^{-T}
        cur_L = self.s_L(s_t)
        cur_R = self.sde.s_R(s_t)
        coef = cur_L.T @ inv_2x2(cur_R.T)
        return utils.sbmm(coef, eps)


    @partial(jit, static_argnums=(0,))
    def s_eps_integrand(self, s_t):
        cur_G = self.s_G(s_t)
        
        integrand = 0.5 * cur_G @ cur_G @ inv_2x2(self.s_L(s_t)).T
        return integrand

    def get_deis_coef(self, order, rev_timesteps, used_cache=True):
        hexinfo = sha1(np.asarray(rev_timesteps)).hexdigest()
        fpath = osp.join(self.f_prefix, f"deis_coef_oder{order}_TLen_{len(rev_timesteps)}_{hexinfo}.pkl")
        if self.used_cache and osp.exists(fpath) and used_cache:
            return jnp.asarray(jio.load(fpath))
        # return [x_coef, eps_coef]
        assert jax.config.read("jax_enable_x64") == self.x64
        x_coef = self.vv_psi(rev_timesteps[:-1], rev_timesteps[1:]) # [N, 2, 2]
        eps_coef = get_ab_eps_coef(self, order+1, rev_timesteps, order) # [N, order+1, 2, 2]
        rtn = jnp.concatenate([x_coef[:, None], eps_coef], axis=1) # [N, order+2, 2,2]
        jio.dump(fpath, rtn)
        return rtn