import jax

def random_split_like_tree(rng_key, target=None, treedef=None):
    if treedef is None:
        treedef = jax.tree_structure(target)
    keys = jax.random.split(rng_key, treedef.num_leaves + 1)
    return jax.tree_unflatten(treedef, keys[:-1]), keys[-1]

def get_sigma(h, lam, delta):
    return 1 / jax.numpy.sqrt(lam * (h + delta))

def add_noise_to_params(params, state):
    if hasattr(state, "sample_params"):
        lam = state.num_datapoints
        delta = state.weight_decay
        params = jax.tree_util.tree_map(
            lambda m, h, e: m + e * get_sigma(h, lam, delta), params, state.h, state.eps
        )
    
    return params

def parallel_value_and_grad(loss_fn, state, params, x, y):
    if hasattr(state, "sample_params"):
        _params = add_noise_to_params(params, state)
        loss_value, grads = jax.vmap(
            jax.value_and_grad(loss_fn), in_axes=(0, None, None))(_params, x, y)
    else:
        loss_value, grads = jax.value_and_grad(loss_fn)(params, x, y)

    return loss_value, grads