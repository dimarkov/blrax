import jax
import chex
import equinox as eqx
from collections.abc import Callable
from typing import Optional
from blrax.states import ScaleByIvonState
from jax import lax


def precision(h, ess, weight_decay):
    return ess * (h + weight_decay)

def sigma(h, ess, weight_decay):
    return lax.rsqrt(precision(h, ess, weight_decay))

def get_scale(state):
    return jax.tree.map(
        lambda v: sigma(v, state.ess, state.weight_decay), state.hess
    )

def sample_posterior(key, params, state, shape=(), mask=None):
    ess = state.ess
    weight_decay = state.weight_decay

    pleaves, paux = jax.tree.flatten(params)
    hleaves = jax.tree.leaves(state.hess)

    if mask is not None:
        mleaves = jax.tree.leaves(mask)
        if len(mleaves) < len(pleaves):
            mleaves = jax.tree.leaves(mask, is_leaf=lambda x: x is None)
    else:
        mleaves = [None] * len(pleaves)

    samples = []
    noise = []
    for p, h, m in zip(pleaves, hleaves, mleaves):
        key, rng = jax.random.split(key)
        n = p + jax.random.normal(rng, shape=shape + p.shape)
        val = p + n * sigma(h, ess, weight_decay)

        samples.append( 
            val if m is None else jax.numpy.where(m, val, 0.0)
        )

        noise.append(
            n if m is None else jax.numpy.where(m, n, 0.0)
        )

    sampled_params = jax.tree.unflatten(paux, samples)
    noise = jax.tree.unflatten(paux, noise)

    return sampled_params, state._replace(noise=noise)

def tree_stack(pytree_list, axis=0):
    def stack(*args):
        return jax.numpy.stack(args, axis=axis)

    return jax.tree.map(stack, *pytree_list)

def sequential_sampling(key, loss_fn, params, state, mc_samples, *args, mask=None, **kwargs):
    keys = jax.random.split(key, mc_samples)

    output = []
    for i, rn in enumerate(keys):
        # draw IVON weight posterior sample
        k1, k2 = jax.random.split(rn)
        psample, optstate = sample_posterior(k1, params, state, mask=mask)

        # get gradient and loss for this MC sample from
        out, grad = jax.value_and_grad(loss_fn, **kwargs)(psample, *args, k2)
        output.append(out if isinstance(out, tuple) else (out,))
        if i == 0:
            g_bar = grad
            h_bar = jax.tree.map(
                lambda g, n: g * n, grad, optstate.noise
            )
        else:
            g_bar = jax.tree.map(lambda a, g: a + g, g_bar, grad)
            h_bar = jax.tree.map(
                lambda h, g, n: h + g * n, h_bar, grad, optstate.noise
            )
    
    if mc_samples > 1:
        g_bar = jax.tree.map(lambda g: g / mc_samples, g_bar)
        h_bar = jax.tree.map(lambda h: h / mc_samples, h_bar)

    output = tree_stack(output, axis=0)
    output = output[0] if len(output) == 1 else output

    return output, g_bar, state._replace(h_bar=h_bar)

def parallel_sampling(key, loss_fn, params, state, mc_samples, *args, mask=None, **kwargs):
    key, _key = jax.random.split(key)
    sampled_params, state = sample_posterior(_key, params, state, shape=(mc_samples,), mask=mask)
    parallel_value_and_grad = jax.vmap(
        jax.value_and_grad(loss_fn, **kwargs), in_axes=(0,) + tuple([None] * len(args)) + (0,)
    )
    keys = jax.random.split(key, mc_samples)
    out, grads = parallel_value_and_grad(sampled_params, *args, keys)

    g_bar = jax.tree.map(lambda g: jax.numpy.mean(g, 0), grads)
    h_bar = jax.tree.map(lambda g, n: jax.numpy.mean(g * n, 0), grads, state.noise)

    return out, g_bar, state._replace(h_bar=h_bar)

@eqx.filter_jit
def noisy_value_and_grad(loss_fn, opt_state, params, key, *args, mc_samples=1, mask=None, method='sequential', **kwargs):
    """Estimate loss value and gradients. If state has noisy values, they are added
    to the parameters, weighted by posterior scale, and the gradients and values are 
    computed around these monte carlo samples. Only the mean estimate of the loss function
    is returned.

    Args:
        loss_fn: Function to be differentiated. It should return a scalar (which includes 
        arrays with shape ``()`` but not arrays with shape ``(1,)`` etc.)
        opt_state: optax type state.
        params: PyTree or Array, over which we are computing gradients and values.
        args: Arguments to the loss function
        key: RNG key to be passed to the loss function. It has to be vmap-able over the number of mc samples.
        mask: PyTree or Array of the same structure as params, used to mask gradients and random samples.

    Returns:
        loss_value: Array, 
        updates: Gradient estimates
        state: optax type state.
    """
    ivon_state = opt_state[0]
    if isinstance(ivon_state, ScaleByIvonState):
        if method == 'parallel':
            out, updates, ivon_state = parallel_sampling(key, loss_fn, params, ivon_state, mc_samples, *args, mask=mask, **kwargs)
        else:
            out, updates, ivon_state = sequential_sampling(key, loss_fn, params, ivon_state, mc_samples, *args, mask=mask, **kwargs)
        
        opt_state = (ivon_state,) + opt_state[1:]
    else:
        out, updates = jax.value_and_grad(loss_fn, **kwargs)(params, *args, key)

    return out, updates, opt_state