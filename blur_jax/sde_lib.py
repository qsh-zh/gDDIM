import jax
import jax.numpy as jnp
from jax import lax
from jax import vmap, jit, random
from functools import partial
import blur

@jax.jit
def batch_mul(a, b):
  return vmap(lambda a, b: a * b)(a, b)


def linear_t2alpha_fn(t):
    beta_0, beta_1 = 0.01, 20
    log_mean_coef = -0.25 * t ** 2 * (beta_1 - beta_0) - 0.5 * t * beta_0
    return jnp.exp(2 * log_mean_coef)

class SDE:
    def __init__(self, min_scale=0.001, sigma_blur_max=10.0, sampling_eps=1e-5):
        self.min_scale = min_scale
        self.sigma_blur_max = sigma_blur_max
        self.sampling_eps = sampling_eps

        img_dim = 32
        freqs = jnp.pi * jnp.linspace(0, img_dim  - 1, img_dim) / img_dim
        self.labda = freqs[None, :, None, None]**2  + freqs[None, None, :,  None]**2
        self.alpha_start = self.t2alpha_fn(0.0)

    @property
    def T(self):
        return 1.0

    @property
    def sampling_T(self):
        return self.rho2t(80.0)

    def t2alpha_fn(self, t):
        return jnp.cos(
            ((t+0.004) / 1.008 * jnp.pi / 2)
        )**2

    def alpha2t_fn(self, alpha):
        return jnp.arccos(
            jnp.sqrt(alpha)
        ) * 2 / jnp.pi * 1.008 - 0.004

    def rho2t(self, rho):
        num = self.alpha_start
        denum = (rho + jnp.sqrt(1 - self.alpha_start))**2 + self.alpha_start
        cur_alpha = num / denum
        return self.alpha2t_fn(cur_alpha)

    def psi(self, t_start, t_end):
        alpha_sq_ratio = jnp.sqrt(self.t2alpha_fn(t_end) / self.t2alpha_fn(t_start))
        scaling_ratio = self.get_frequency_scaling(t_end) / self.get_frequency_scaling(t_start)
        return batch_mul(alpha_sq_ratio, scaling_ratio) # (B, H , W, C)

    def G(self, ts):
        dalpha_dt = self.dalpha_dt_fn(ts) #(B,)
        alpha_t = self.t2alpha_fn(ts)
        D_t = self.get_frequency_scaling(ts) #(B, H, W, C=1)
        return jnp.sqrt(
            batch_mul(
                dalpha_dt,
                (
                    -1.0 + 
                    batch_mul(1 - 1.0 / alpha_t, D_t)
                )
            )
        )

    def eps_integrand(self, vec_t):
        cur_G = self.G(vec_t)
        return batch_mul(
            0.5 * cur_G*cur_G,
            1.0 / jnp.sqrt(1 - self.t2alpha_fn(vec_t))
        ) # (B, H, W, C=1)

    def get_frequency_scaling(self, t):
        sigma_blur = self.sigma_blur_max * jnp.sin( t * jnp.pi / 2)**2
        dissipation_time = sigma_blur**2 / 2

        # compute scaling for frequencies
        labda = self.labda
        logits = dissipation_time[:,None, None, None] * labda
        scaling = jnp.exp( -logits) * (1 - self.min_scale)
        scaling = scaling + self.min_scale
        return scaling

    def y_mean_coef(self, ts):
        freq_scaling = self.get_frequency_scaling(ts) # (B, D, D, 1)
        alphas = self.t2alpha_fn(ts)
        return batch_mul(jnp.sqrt(alphas), freq_scaling)

    def y_std_coef(self, ts):
        alphas = self.t2alpha_fn(ts)
        return jnp.sqrt(1 - alphas)

    @partial(jit, static_argnums=(0,))
    def perturb_data(self, batch, ts, rng, noise_ratio=1.0):
        eps = random.normal(rng, batch.shape)
        ys = blur.batch_img_dct(batch)

        mean_coef, std_coef = self.y_mean_coef(ts), self.y_std_coef(ts)

        mean = blur.batch_img_idct(
            batch_mul(mean_coef, ys)
        )
        x_t = mean + batch_mul(std_coef, eps) * noise_ratio
        return x_t, mean, eps

    def cos_perturb_data(self, batch, ts, rng, noise_ratio=1.0):
        eps = random.normal(rng, batch.shape)
        alpha = self.t2alpha_fn(ts)
        mean = batch_mul(
            jnp.sqrt(alpha), batch
        )
        x_t = mean + batch_mul(
            jnp.sqrt(1-alpha), eps
        ) * noise_ratio
        return x_t, mean, eps

    def linear_perturb_data(self, batch, ts, rng, noise_ratio=1.0):
        eps = random.normal(rng, batch.shape)
        alpha = linear_t2alpha_fn(ts)
        mean = batch_mul(
            jnp.sqrt(alpha), batch
        )
        x_t = mean + batch_mul(
            jnp.sqrt(1-alpha), eps
        ) * noise_ratio
        return x_t, mean, eps

    def prior_sampling(self, rng, shape):
        return jax.random.normal(rng, shape=shape)

    def x2y(self, xs):
        return blur.batch_img_dct(xs)

    def y2x(self, ys):
        return blur.batch_img_idct(ys)

    def sample_t(self, shape, rng):
        return random.uniform(rng, shape, minval=1e-5, maxval=self.T)

    def encode_t(self, t):
        return 999 * t

    def encode_x(self, xs):
        return xs

    def model2eps(self, xs, ts, model_output):
        del xs, ts
        return model_output

    def xeps2x0(self, xt, ts, xeps):
        xt_clean_pred = xt - batch_mul(
            jnp.sqrt(1 - self.t2alpha_fn(ts)),
            xeps
        )
        yt = self.x2y(xt_clean_pred)
        y0 = 1.0 / self.y_mean_coef(ts) * yt
        return self.y2x(y0)

def from_config(config):
    return SDE(
        sigma_blur_max = config.model.sigma_blur_max,
        sampling_eps = config.sampling.t0
    )