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

  .. warning::
    The possible dtypes may be limited by the sampler, for example
    ``jax.random.rademacher`` only supports integer dtypes and will raise an
    error if the dtype of the target tree is not an integer or if the dtype
    is not of integer type.

  .. versionadded:: 0.2.1
  """
  keys_tree = tree_split_key_like(rng_key, target_tree)
  return jax.tree_util.tree_map(
      lambda leaf, key: sampler(key, add_shape + leaf.shape, dtype or leaf.dtype),
      target_tree,
      keys_tree,
  )

def get_sigma(h, lam, delta):
    return 1 / jax.numpy.sqrt(lam * (h + delta))

def get_scale(state):
    return jax.tree_util.tree_map(
        lambda v: get_sigma(v, state.num_datapoints, state.weight_decay), state.h
    )

def add_noise_to_params(params, state, mask=None):
    lam = state.num_datapoints
    delta = state.weight_decay

    if mask is not None:
        params = jax.tree_util.tree_map(
            lambda m, h, e, t: m + t * e * get_sigma(h, lam, delta), params, state.h, state.eps, mask
        )
    else:
        params = jax.tree_util.tree_map(
            lambda m, h, e: m + e * get_sigma(h, lam, delta), params, state.h, state.eps
        )
    
    return params

def sample_posterior(key, params, state, n_samples=1):
    lam = state.num_datapoints
    delta = state.weight_decay
    keys_tree = tree_split_key_like(key, params)
    params = jax.tree_util.tree_map(
        lambda m, h, k: m + jax.random.normal(k, shape=(n_samples,) + m.shape) * get_sigma(h, lam, delta), params, state.h, keys_tree
    )

    return params

@eqx.filter_jit
def noisy_value_and_grad(loss_fn, state, params, *args, key=None, mask=None, **kwargs):
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
        key: RNG key to be passed to the loss function
        mask: PyTree or Array of the same structure as params, used to mask gradients and random samples.

    Returns:
        loss_value: Array, 
        grads: Noisy gradient estimates
    """
    if isinstance(state, ScaleByIvonState):
        _params = add_noise_to_params(params, state, mask=mask)
        if key is None:
            loss_value, grads = jax.vmap(
                jax.value_and_grad(loss_fn, **kwargs), in_axes=(0,) + tuple([None] * len(args)))(_params, *args)
        else:
            n_samples = state.mc_samples  
            keys = jax.random.split(key, n_samples)
            loss_value, grads = jax.vmap(
                jax.value_and_grad(loss_fn, **kwargs), in_axes=(0,) + tuple([None] * len(args)) + (0,))(_params, *args, key=keys)

        loss_value = jax.numpy.mean(loss_value)
    else:
        loss_value, grads = jax.value_and_grad(loss_fn, **kwargs)(params, *args) if key is None else jax.value_and_grad(loss_fn)(params, *args, key=key)

    return loss_value, grads