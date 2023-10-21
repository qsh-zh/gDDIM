import jax
import jax.numpy as jnp
from jax import lax
import fft

__all__ = [
    "img_dct",
    "img_idct",
]

def _impl_dct(x):
  N = len(x)
  axis = 0
  v0 = lax.slice_in_dim(x, None, None, 2, axis)
  v1 = lax.rev(lax.slice_in_dim(x, 1, None, 2, axis), (axis,))
  v = lax.concatenate([v0, v1], axis)

  fft_value = fft.fft(v)
  fft_real = fft_value.real
  fft_img = fft_value.imag

  k =  - jnp.arange(N, dtype=x.dtype) * jnp.pi / (2 * N)
  W_r, W_i = jnp.cos(k), jnp.sin(k)

  V = fft_real * W_r - fft_img * W_i

  factor = lax.concatenate(
      [
          lax.full((1,), jnp.sqrt(N) * 2, x.dtype),
          lax.full((N - 1,), jnp.sqrt(N * 2), x.dtype)
      ], 
    0
  )

  V = 2 * V / factor

  return V

jax_dct = jax.vmap(_impl_dct)

def _img_dct(img):
  w_img = jax_dct(img)
  h_img = jax_dct(w_img.T)
  return h_img.T

img_dct = jax.jit(
    jax.vmap(jax.vmap(_img_dct))
)


def _impl_idct(X):
  N = X.shape[-1]
  X_v = X / 2

  factor = lax.concatenate(
      [
          lax.full((1,), jnp.sqrt(N) * 2, X.dtype),
          lax.full((N - 1,), jnp.sqrt(N * 2), X.dtype)
      ], 
    0
  )
  X_v = X_v * factor

  k = jnp.arange(N, dtype=X.dtype) * jnp.pi / (2 * N)
  W_r, W_i = jnp.cos(k), jnp.sin(k)

  V_t_r = X_v
  V_t_i = jnp.concatenate([
      X_v[:1] * 0, 
      -jnp.flip(X_v)[:-1]
      ], axis=0
  )

  V_r = V_t_r * W_r - V_t_i * W_i
  V_i = V_t_r * W_i + V_t_i * W_r

  v = fft.irfft(
        lax.complex(V_r, V_i), n=N,
  )
  
  value = jnp.stack([
      v[:N - (N // 2)],
      jnp.flip(v)[:N // 2]
  ]).T
  return value.flatten()

jax_idct = jax.vmap(_impl_idct)


def _img_idct(img):
  w_img = jax_idct(img)
  h_img = jax_idct(w_img.T)
  return h_img.T

img_idct = jax.jit(
    jax.vmap(jax.vmap(_img_idct))
)

def batch_img_dct(xs):
    xs = jnp.transpose(xs, [0, 3, 1, 2])
    ys = img_dct(xs)
    return jnp.transpose(ys, [0, 2, 3, 1])

def batch_img_idct(ys):
    ys = jnp.transpose(ys, [0, 3, 1, 2])
    xs = img_idct(ys)
    return jnp.transpose(xs, [0, 2, 3, 1])