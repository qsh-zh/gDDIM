# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import operator
import numpy as np

from jax import lax
from jax.lib import xla_client
from jax._src.util import safe_zip
from jax._src.numpy.util import _wraps
# from .util import _wraps
# from . import lax_numpy as jnp
from jax._src.numpy import lax_numpy as jnp
from jax import ops as jaxops


def _fft_core(func_name, fft_type, a, s, axes, norm):
  full_name = "jax.numpy.fft." + func_name

  if s is not None:
    s = tuple(map(operator.index, s))
    if np.any(np.less(s, 0)):
      raise ValueError("Shape should be non-negative.")
  if norm is not None:
    raise NotImplementedError("%s only supports norm=None, got %s" % (full_name, norm))
  if s is not None and axes is not None and len(s) != len(axes):
    # Same error as numpy.
    raise ValueError("Shape and axes have different lengths.")

  orig_axes = axes
  if axes is None:
    if s is None:
      axes = range(a.ndim)
    else:
      axes = range(a.ndim - len(s), a.ndim)

  if len(axes) != len(set(axes)):
    raise ValueError(
        "%s does not support repeated axes. Got axes %s." % (full_name, axes))

  if len(axes) > 3:
    # XLA does not support FFTs over more than 3 dimensions
    raise ValueError(
        "%s only supports 1D, 2D, and 3D FFTs. "
        "Got axes %s with input rank %s." % (full_name, orig_axes, a.ndim))

  # XLA only supports FFTs over the innermost axes, so rearrange if necessary.
  if orig_axes is not None:
    axes = tuple(range(a.ndim - len(axes), a.ndim))
    a = jnp.moveaxis(a, orig_axes, axes)

  if s is not None:
    a = jnp.asarray(a)
    in_s = list(a.shape)
    for axis, x in safe_zip(axes, s):
      in_s[axis] = x
    if fft_type == xla_client.FftType.IRFFT:
      in_s[-1] = (in_s[-1] // 2 + 1)
    # Cropping
    a = a[tuple(map(slice, in_s))]
    # Padding
    a = jnp.pad(a, [(0, x-y) for x, y in zip(in_s, a.shape)])
  else:
    if fft_type == xla_client.FftType.IRFFT:
      s = [a.shape[axis] for axis in axes[:-1]]
      if axes:
        s += [max(0, 2 * (a.shape[axes[-1]] - 1))]
    else:
      s = [a.shape[axis] for axis in axes]

  transformed = lax.fft(a, fft_type, s)

  if orig_axes is not None:
    transformed = jnp.moveaxis(transformed, axes, orig_axes)
  return transformed


@_wraps(np.fft.fftn)
def fftn(a, s=None, axes=None, norm=None):
  return _fft_core('fftn', xla_client.FftType.FFT, a, s, axes, norm)


@_wraps(np.fft.ifftn)
def ifftn(a, s=None, axes=None, norm=None):
  return _fft_core('ifftn', xla_client.FftType.IFFT, a, s, axes, norm)


@_wraps(np.fft.rfftn)
def rfftn(a, s=None, axes=None, norm=None):
  return _fft_core('rfftn', xla_client.FftType.RFFT, a, s, axes, norm)


@_wraps(np.fft.irfftn)
def irfftn(a, s=None, axes=None, norm=None):
  return _fft_core('irfftn', xla_client.FftType.IRFFT, a, s, axes, norm)


def _axis_check_1d(func_name, axis):
  full_name = "jax.numpy.fft." + func_name
  if isinstance(axis, (list, tuple)):
    raise ValueError(
        "%s does not support multiple axes. Please use %sn. "
        "Got axis = %r." % (full_name, full_name, axis)
    )

def _fft_core_1d(func_name, fft_type, a, n, axis, norm):
  _axis_check_1d(func_name, axis)
  axes = None if axis is None else [axis]
  s = None if n is None else [n]
  return _fft_core(func_name, fft_type, a, s, axes, norm)


@_wraps(np.fft.fft)
def fft(a, n=None, axis=-1, norm=None):
  return _fft_core_1d('fft', xla_client.FftType.FFT, a, n=n, axis=axis,
                      norm=norm)

@_wraps(np.fft.ifft)
def ifft(a, n=None, axis=-1, norm=None):
  return _fft_core_1d('ifft', xla_client.FftType.IFFT, a, n=n, axis=axis,
                      norm=norm)

@_wraps(np.fft.rfft)
def rfft(a, n=None, axis=-1, norm=None):
  return _fft_core_1d('rfft', xla_client.FftType.RFFT, a, n=n, axis=axis,
                      norm=norm)

@_wraps(np.fft.irfft)
def irfft(a, n=None, axis=-1, norm=None):
  return _fft_core_1d('irfft', xla_client.FftType.IRFFT, a, n=n, axis=axis,
                      norm=norm)

@_wraps(np.fft.hfft)
def hfft(a, n=None, axis=-1, norm=None):
  conj_a = jnp.conj(a)
  _axis_check_1d('hfft', axis)
  nn = (a.shape[axis] - 1) * 2 if n is None else n
  return _fft_core_1d('hfft', xla_client.FftType.IRFFT, conj_a, n=n, axis=axis,
                      norm=norm) * nn

@_wraps(np.fft.ihfft)
def ihfft(a, n=None, axis=-1, norm=None):
  _axis_check_1d('ihfft', axis)
  nn = a.shape[axis] if n is None else n
  output = _fft_core_1d('ihfft', xla_client.FftType.RFFT, a, n=n, axis=axis,
                      norm=norm)
  return jnp.conj(output) * (1 / nn)


def _fft_core_2d(func_name, fft_type, a, s, axes, norm):
  full_name = "jax.numpy.fft." + func_name
  if len(axes) != 2:
    raise ValueError(
        "%s only supports 2 axes. Got axes = %r."
        % (full_name, axes)
    )
  return _fft_core(func_name, fft_type, a, s, axes, norm)


@_wraps(np.fft.fft2)
def fft2(a, s=None, axes=(-2,-1), norm=None):
  return _fft_core_2d('fft2', xla_client.FftType.FFT, a, s=s, axes=axes,
                      norm=norm)

@_wraps(np.fft.ifft2)
def ifft2(a, s=None, axes=(-2,-1), norm=None):
  return _fft_core_2d('ifft2', xla_client.FftType.IFFT, a, s=s, axes=axes,
                      norm=norm)

@_wraps(np.fft.rfft2)
def rfft2(a, s=None, axes=(-2,-1), norm=None):
  return _fft_core_2d('rfft2', xla_client.FftType.RFFT, a, s=s, axes=axes,
                      norm=norm)

@_wraps(np.fft.irfft2)
def irfft2(a, s=None, axes=(-2,-1), norm=None):
  return _fft_core_2d('irfft2', xla_client.FftType.IRFFT, a, s=s, axes=axes,
                      norm=norm)


@_wraps(np.fft.fftfreq)
def fftfreq(n, d=1.0):
  if isinstance(n, (list, tuple)):
    raise ValueError(
          "The n argument of jax.numpy.fft.fftfreq only takes an int. "
          "Got n = %s." % list(n))

  elif isinstance(d, (list, tuple)):
    raise ValueError(
          "The d argument of jax.numpy.fft.fftfreq only takes a single value. "
          "Got d = %s." % list(d))

  k = jnp.zeros(n)
  if n % 2 == 0:
    # k[0: n // 2 - 1] = jnp.arange(0, n // 2 - 1)
    k = jaxops.index_update(k, jaxops.index[0: n // 2], jnp.arange(0, n // 2))

    # k[n // 2:] = jnp.arange(-n // 2, -1)
    k = jaxops.index_update(k, jaxops.index[n // 2:], jnp.arange(-n // 2, 0))

  else:
    # k[0: (n - 1) // 2] = jnp.arange(0, (n - 1) // 2)
    k = jaxops.index_update(k, jaxops.index[0: (n - 1) // 2 + 1],
                            jnp.arange(0, (n - 1) // 2 + 1))

    # k[(n - 1) // 2 + 1:] = jnp.arange(-(n - 1) // 2, -1)
    k = jaxops.index_update(k, jaxops.index[(n - 1) // 2 + 1:],
                            jnp.arange(-(n - 1) // 2, 0))

  return k / (d * n)


@_wraps(np.fft.rfftfreq)
def rfftfreq(n, d=1.0):
  if isinstance(n, (list, tuple)):
    raise ValueError(
          "The n argument of jax.numpy.fft.rfftfreq only takes an int. "
          "Got n = %s." % list(n))

  elif isinstance(d, (list, tuple)):
    raise ValueError(
          "The d argument of jax.numpy.fft.rfftfreq only takes a single value. "
          "Got d = %s." % list(d))

  if n % 2 == 0:
    k = jnp.arange(0, n // 2 + 1)

  else:
    k = jnp.arange(0, (n - 1) // 2 + 1)

  return k / (d * n)


@_wraps(np.fft.fftshift)
def fftshift(x, axes=None):
  x = jnp.asarray(x)
  if axes is None:
    axes = tuple(range(x.ndim))
    shift = [dim // 2 for dim in x.shape]
  elif isinstance(axes, int):
    shift = x.shape[axes] // 2
  else:
    shift = [x.shape[ax] // 2 for ax in axes]

  return jnp.roll(x, shift, axes)


@_wraps(np.fft.ifftshift)
def ifftshift(x, axes=None):
  x = jnp.asarray(x)
  if axes is None:
    axes = tuple(range(x.ndim))
    shift = [-(dim // 2) for dim in x.shape]
  elif isinstance(axes, int):
    shift = -(x.shape[axes] // 2)
  else:
    shift = [-(x.shape[ax] // 2) for ax in axes]

  return jnp.roll(x, shift, axes)