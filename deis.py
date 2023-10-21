from functools import partial
import jax
import jax.numpy as jnp

@partial(jax.jit, static_argnums=(3,))
def runge_kutta(x, t, dt, fn):
    grad_1 = fn(x, t)
    x_2 = x + grad_1 * dt / 2
    
    grad_2 = fn(x_2, t + dt / 2)
    x_3 = x + grad_2 * dt / 2
    
    grad_3 = fn(x_3, t + dt / 2)
    x_4 = x + grad_3 * dt
    
    grad_4 = fn(x_4, t + dt)
    return x + dt / 6 * (grad_1 + 2 * grad_2 + 2 * grad_3 + grad_4)

def get_eps_coef_worker_fn(sde):
    def _worker(t_start, t_end, num_item):
        dt = (t_end - t_start) / num_item

        t_inter = jnp.linspace(t_start, t_end, num_item, endpoint=False)
        psi_coef = sde.vs_psi(t_inter, t_end) #(n, 2, 2)
        integrand = sde.v_eps_integrand(t_inter) #(n, 2, 2)

        return jnp.einsum("bij,bjk->bik", psi_coef, integrand), t_inter, dt #(n, 2,2), #(n, 2, 2), dt
    return _worker

@jax.jit
def single_poly_coef(t_val, ts_poly, coef_idx=0):
    num = t_val - ts_poly
    denum = ts_poly[coef_idx] - ts_poly
    num = num.at[coef_idx].set(1.0)
    denum = denum.at[coef_idx].set(1.0)
    return jnp.prod(num) / jnp.prod(denum)

vec_poly_coef = jax.jit(jax.vmap(single_poly_coef, (0, None, None), 0))


def get_eps_single_coef_fn(sde):
    _eps_coef_worker_fn = get_eps_coef_worker_fn(sde)
    def _worker(t_start, t_end, ts_poly, coef_idx=0,num_item=10000):
        integrand, t_inter, dt = _eps_coef_worker_fn(t_start, t_end, num_item)
        poly_coef = vec_poly_coef(t_inter, ts_poly, coef_idx) #(N, )
        return jnp.sum(integrand * poly_coef[:,None,None], axis=0) * dt #(2,2)
    return _worker

def get_eps_coef_fn(sde, highest_order, order):
    eps_coef_fn = get_eps_single_coef_fn(sde)
    @jax.jit
    def _worker(t_start, t_end, ts_poly, num_item=10000):
        rtn = jnp.zeros((highest_order+1, 2, 2), dtype=float)
        ts_poly = ts_poly[:order+1]
        coef = jax.vmap(eps_coef_fn, (None, None, None, 0, None))(t_start, t_end, ts_poly, jnp.flip(jnp.arange(order+1)), num_item)
        assert coef.shape == (order+1, 2, 2)
        rtn = rtn.at[:order+1].set(coef)
        return rtn
    return _worker

def get_ab_eps_coef_order0(sde, highest_order, timesteps):
    _worker = get_eps_coef_fn(sde, highest_order, 0)
    col_idx = jnp.arange(len(timesteps)-1)[:,None]
    idx = col_idx + jnp.arange(1)[None, :]
    vec_ts_poly = timesteps[idx]
    return jax.vmap(
        _worker,
        (0, 0, 0), 0
    )(timesteps[:-1], timesteps[1:], vec_ts_poly) # (N, order, 2, 2)

def get_ab_eps_coef(sde, highest_order, timesteps, order):
    if order == 0:
        return get_ab_eps_coef_order0(sde, highest_order, timesteps)
    
    prev_coef = get_ab_eps_coef(sde, highest_order, timesteps[:order+1], order=order-1)

    cur_coef_worker = get_eps_coef_fn(sde, highest_order, order)

    col_idx = jnp.arange(len(timesteps)-order-1)[:,None]
    idx = col_idx + jnp.arange(order+1)[None, :]
    vec_ts_poly = timesteps[idx]
    

    cur_coef = jax.vmap(
        cur_coef_worker,
        (0, 0, 0), 0
    )(timesteps[order:-1], timesteps[order+1:], vec_ts_poly) #[3, 4, (0,1,2,3)]

    return jnp.concatenate(
        [
            prev_coef,
            cur_coef
        ],
        axis=0
    )

def get_am_eps_coef_order1(sde, highest_order, timesteps):
    # ts -> [0, ..., n-1]
    # eg n = 2
    _worker = get_eps_coef_fn(sde, highest_order, 1)
    col_idx = jnp.arange(len(timesteps)-1)[:,None] # [0, n-2]
    idx = col_idx + jnp.arange(2)[None, :] #[(0,1), (n-2, n-1)]
    vec_ts_poly = timesteps[idx]
    return jax.vmap(
        _worker,
        (0, 0, 0), 0
    )(timesteps[:-1], timesteps[1:], vec_ts_poly)

def get_am_eps_coef(sde, highest_order, timesteps, order):
    # ts -> [0, ..., n-1]
    if order == 0:
        raise RuntimeError("Using Adams-Moulton with order 0")
    if order == 1:
        return get_am_eps_coef_order1(sde, highest_order, timesteps)

    prev_coef = get_am_eps_coef(sde, highest_order, timesteps[:order], order=order-1)

    cur_coef_worker = get_eps_coef_fn(sde, highest_order, order)

    # eg: order = 2
    col_idx = jnp.arange(len(timesteps)-order)[:,None]
    # [0, 1, 2]
    idx = col_idx + jnp.arange(order+1)[None, :]
    # ([0,1,2], ... [n-3,n-2,n-1])
    vec_ts_poly = timesteps[idx]

    # eg: n = 3, order=2
    cur_coef = jax.vmap(
        cur_coef_worker,
        (0, 0, 0), 0
    )(timesteps[order-1:-1], timesteps[order:], vec_ts_poly) #[1, 2, (0,1,2)]

    return jnp.concatenate(
        [
            prev_coef,
            cur_coef
        ],
        axis=0
    )

@jax.jit
def multistep_ab_step(x, deis_coef, new_eps, eps_pred):
    # x -- (B, ...., d, 2)
    # deis_coef (order + 1, 2, 2)
    # new_eps   (B, ...., d, 2)
    # eps_pred  (order-1, B, ...., d, 2)
    x_coef, eps_coef = deis_coef[0], deis_coef[1:] #(2,2), (order,2,2)
    full_eps = jnp.concatenate([new_eps[None], eps_pred])
    linear_term = jnp.einsum("ij,b...j->b...i", x_coef, x)
    eps_term = jnp.einsum("oij,o...j->...i", eps_coef, full_eps)
    return linear_term + eps_term, full_eps[:-1]