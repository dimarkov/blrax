import jax
import chex
import equinox as eqx
from collections.abc import Callable
from typing import Optional
from blrax.states import ScaleByIvonState

def tree_split_key_like(
    rng_key: chex.PRNGKey, target_tree: chex.ArrayTree
) -> chex.ArrayTree:
  """Split keys to match structure of target tree.

  Args:
    rng_key: the key to split.
    target_tree: the tree whose structure to match.

  Returns:
    a tree of rng keys.
  """
  tree_def = jax.tree.structure(target_tree)
  keys = jax.random.split(rng_key, tree_def.num_leaves)
  return jax.tree.unflatten(tree_def, keys)


def tree_random_like(
    rng_key: chex.PRNGKey,
    target_tree: chex.ArrayTree,
    sampler: Callable[
        [chex.PRNGKey, chex.Shape, chex.ArrayDType], chex.Array
    ] = jax.random.normal,
    add_shape: Optional[tuple]=(),
    dtype: Optional[chex.ArrayDType] = None,
) -> chex.ArrayTree:
  """Create tree with random entries of the same shape as target tree.

  Args:
    rng_key: the key for the random number generator.
    target_tree: the tree whose structure to match. Leaves must be arrays.
    sampler: the noise sampling function, by default ``jax.random.normal``.
    dtype: the desired dtype for the random numbers, passed to ``sampler``. If
      None, the dtype of the target tree is used if possible.

  Returns:
    a random tree with the same structure as ``target_tree``, whose leaves have
    distribution ``sampler``.
  """
  keys_tree = tree_split_key_like(rng_key, target_tree)
  return jax.tree_util.tree_map(
      lambda leaf, key: sampler(key, add_shape + leaf.shape, dtype or leaf.dtype),
      target_tree,
      keys_tree,
  )

def get_sigma(h, ess, weight_decay):
    return 1 / jax.numpy.sqrt(ess * (h + weight_decay))

def get_scale(state):
    return jax.tree_util.tree_map(
        lambda v: get_sigma(v, state.ess, state.weight_decay), state.hess
    )

def add_noise_to_params(params, state, mask=None):
    ess = state.ess
    delta = state.weight_decay

    if mask is not None:
        params = jax.tree_util.tree_map(
            lambda m, h, e, t: m + t * e * get_sigma(h, ess, delta), params, state.hess, state.noise, mask
        )
    else:
        params = jax.tree_util.tree_map(
            lambda m, h, e: m + e * get_sigma(h, ess, delta), params, state.hess, state.noise
        )
    
    return params

def sample_posterior(key, params, state, shape=(), mask=None):
    ess = state.ess
    weight_decay = state.weight_decay

    pleaves, paux = jax.tree_util.tree_flatten(params)
    hleaves = jax.tree_util.tree_flatten(state.hess)[0]

    mleaves = jax.tree_util.tree_flatten(mask)[0] if mask is not None else [None] * len(pleaves)
    samples = []
    noise = []
    for p, h, m in zip(pleaves, hleaves, mleaves):
        key, rng = jax.random.split(key)
        n = p + jax.random.normal(rng, shape=shape + p.shape)
        val = p + n * get_sigma(h, ess, weight_decay)

        samples.append( 
            val if m is None else jnp.where(m, val, 0.0)
        )

        noise.append(
            n if m is None else jnp.where(m, n, 0.0)
        )

    sampled_params = jax.tree_util.tree_unflatten(paux, samples)
    noise = jax.tree_util.tree_unflatten(paux, noise)

    return sampled_params, state._replace(noise=noise)


def sequential_sampling(key, loss_fn, params, state, mc_samples, *args, mask=None, **kwargs):
    keys = jax.random.split(key, mc_samples)

    total_loss = 0.0
    for i, rn in enumerate(keys):
        # draw IVON weight posterior sample
        k1, k2 = jax.random.split(rn)
        psample, optstate = sample_posterior(k1, params, state, mask=mask)

        # get gradient and loss for this MC sample from
        loss_value, grad = jax.value_and_grad(loss_fn, **kwargs)(psample, *args, k2)
        total_loss += loss_value
        if i == 0:
            hat_g = grad
            hat_h = jax.tree_util.tree_map(
                lambda g, n: g * n, grad, optstate.noise
            )
        else:
            hat_g = jax.tree_util.tree_map(lambda a, g: a + g, hat_g, grad)
            hat_h = jax.tree_util.tree_map(
                lambda h, g, n: h + g * n, hat_g, grad, optstate.noise
            )
    
    if mc_samples > 1:
        hat_g = jax.tree_util.tree_map(lambda g: g / mc_samples, hat_g)
        hat_h = jax.tree_util.tree_map(lambda h: h / mc_samples, hat_h)

    return total_loss / mc_samples, (hat_g, hat_h)

def parallel_sampling(key, loss_fn, params, state, mc_samples, *args, mask=None, **kwargs):
    key, _key = jax.random.split(key)
    sampled_params, state = sample_posterior(_key, params, state, shape=(mc_samples,), mask=mask)
    parallel_value_and_grad = jax.vmap(
        jax.value_and_grad(loss_fn, **kwargs), in_axes=(0,) + tuple([None] * len(args)) + (0,)
    )
    keys = jax.random.split(key, mc_samples)
    loss_value, grads = parallel_value_and_grad(sampled_params, *args, keys)

    hat_g = jax.tree_util.tree_map(lambda g: jax.numpy.mean(g, 0), grads)
    hat_h = jax.tree_util.tree_map(lambda g, n: jax.numpy.mean(g * n, 0), grads, state.noise)

    return jax.numpy.mean(loss_value), (hat_g, hat_h)

@eqx.filter_jit
def noisy_value_and_grad(loss_fn, state, params, key, *args, mc_samples=1, mask=None, method='sequential', **kwargs):
    """Estimate loss value and gradients. If state has noisy values, they are added
    to the parameters, weighted by posterior scale, and the gradients and values are 
    computed around these monte carlo samples. Only the mean estimate of the loss function
    is returned.

    Args:
        loss_fn: Function to be differentiated. It should return a scalar (which includes 
        arrays with shape ``()`` but not arrays with shape ``(1,)`` etc.)
        state: optax type state.
        params: PyTree or Array, over which we are computing gradients and values.
        args: Arguments to the loss function
        key: RNG key to be passed to the loss function. It has to be vmap-able over the number of mc samples.
        mask: PyTree or Array of the same structure as params, used to mask gradients and random samples.

    Returns:
        loss_value: Array, 
        grads: Noisy gradient estimates
    """
    if isinstance(state, ScaleByIvonState):
        if method == 'parallel':
            loss_value, updates = parallel_sampling(key, loss_fn, params, state, mc_samples, *args, mask=mask, **kwargs)
        else:
            loss_value, updates = sequential_sampling(key, loss_fn, params, state, mc_samples, *args, mask=mask, **kwargs)
    else:
        loss_value, updates = jax.value_and_grad(loss_fn, **kwargs)(params, *args, key)

    return loss_value, updates
