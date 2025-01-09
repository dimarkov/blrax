import jax

def random_split_like_tree(rng_key, target=None, treedef=None):
    if treedef is None:
        treedef = jax.tree_structure(target)
    keys = jax.random.split(rng_key, treedef.num_leaves + 1)
    return jax.tree_unflatten(treedef, keys[:-1]), keys[-1]

def get_sigma(h, lam, delta):
    return 1 / jax.numpy.sqrt(lam * (h + delta))

def sample_value_and_grad(fun, params, state, *args, **kwargs):
    if hasattr(state, "sample_params"):
        lam = state.num_datapoints
        delta = state.weight_decay
        if state.sample_params:
            x = jax.tree_util.tree_map(lambda m, h, e: m + e * get_sigma(h, lam, delta), params, state.h, state.eps)
        else:
            x = params
    else:
        x = params
    
    return jax.value_and_grad(fun, *args, **kwargs)(x)