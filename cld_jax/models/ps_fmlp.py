import jax
from jax import random
import jax.numpy as jnp
import flax.linen as nn
from . import utils
import ml_collections

class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""  
  embed_dim: int
  scale: float = 30.
  @nn.compact
  def __call__(self, x):    
    # Randomly sample weights during initialization. These weights are fixed 
    # during optimization and are not trainable.
    W = self.param('W', jax.nn.initializers.normal(stddev=self.scale), 
                 (self.embed_dim // 2, ))
    W = jax.lax.stop_gradient(W)
    x_proj = x[:, None] * W[None, :] * 2 * jnp.pi
    return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)

class TimeEmb(nn.Module):
  embed_dim: int
  @nn.compact
  def __call__(self, x):
    w = self.param('w', lambda _: jnp.linspace(0.1, 100, self.embed_dim)[None,:])
    b = self.param('b', lambda rng: random.normal(rng, (self.embed_dim,))[None,:])
    angle = x[:, None] * w + b
    return jnp.concatenate([jnp.sin(angle), jnp.cos(angle)], axis=-1)

@utils.register_model(name="ps_fmlp")
class PSFMLP(nn.Module):
  config: ml_collections
  
  @nn.compact
  def __call__(self, x, t, train=False): 
    del train
    act = nn.swish
    nf = self.config.model.nf
    out_dim = x.shape[-1]
    # Obtain the Gaussian random feature embedding for t (B,embed_dim)
    t_embed = act(
        nn.Dense(2 * nf)(
          GaussianFourierProjection(embed_dim=nf)(t)
          # TimeEmb(embed_dim=nf)(t)
        )
    )
    t_embed = nn.Dense(nf)(t_embed)

    x_ebmbed = nn.Dense(nf)(x)
    all_ebmbed = act(t_embed + x_ebmbed)
    feat = act(nn.Dense(nf)(all_ebmbed))
    feat = act(nn.Dense(nf)(feat))
    return nn.Dense(out_dim)(feat)