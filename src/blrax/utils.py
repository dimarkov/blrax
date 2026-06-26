import jax
import chex
import equinox as eqx
from collections.abc import Callable
from typing import Optional
from blrax.states import ScaleByIvonState, ScaleByEvonState, MatrixEvonLeaf, DiagEvonLeaf, _is_evon_leaf
from jax import lax


def _project(QL, QR, X):
    """G° = QLᵀ X QR, skipping any None side. Leading batch dims on X are OK."""
    if QL is not None:
        X = jax.numpy.swapaxes(QL, -1, -2) @ X
    if QR is not None:
        X = X @ QR
    return X


def _project_back(QL, QR, U):
    """Δ = QL U QRᵀ, skipping any None side. Leading batch dims on U are OK."""
    if QL is not None:
        U = QL @ U
    if QR is not None:
        U = U @ jax.numpy.swapaxes(QR, -1, -2)
    return U


def precision(h, ess, weight_decay):
    return ess * (h + weight_decay)

def sigma(h, ess, weight_decay):
    return lax.rsqrt(precision(h, ess, weight_decay))

def get_scale(state):
    if isinstance(state, ScaleByEvonState):
        ess, wd = state.ess, state.weight_decay
        def leaf_scale(s):
            V = 1.0 / (ess * (s.H + wd))           # (d, o)
            if isinstance(s, DiagEvonLeaf):
                return jax.numpy.sqrt(V)
            left = (s.QL * s.QL) if s.QL is not None else None
            right = (s.QR * s.QR) if s.QR is not None else None
            var = V
            if left is not None:
                var = left @ var
            if right is not None:
                var = var @ right.T
            return jax.numpy.sqrt(var)
        return jax.tree.map(leaf_scale, state.leaves, is_leaf=_is_evon_leaf)
    # existing IVON body:
    return jax.tree.map(
        lambda v: sigma(v, state.ess, state.weight_decay), state.hess
    )

def sample_posterior(key, params, state, shape=(), mask=None):
    if isinstance(state, ScaleByEvonState):
        samples, _ = _evon_sample_leaves(key, params, state, shape=shape, mask=mask)
        return samples
    # existing IVON body unchanged below
    ess = state.ess
    weight_decay = state.weight_decay

    pleaves, paux = jax.tree.flatten(params, is_leaf=lambda x: x is None)
    hleaves = jax.tree.leaves(state.hess, is_leaf=lambda x: x is None)

    if mask is not None:
        mleaves = jax.tree.leaves(mask, is_leaf=lambda x: x is None)
    else:
        mleaves = [None] * len(pleaves)

    samples = []
    noise = []
    for p, h, m in zip(pleaves, hleaves, mleaves):
        if p is not None:
            key, rng = jax.random.split(key)
            n = sigma(h, ess, weight_decay) * jax.random.normal(rng, shape=shape + p.shape)
            val = p + n

            samples.append( 
                val if m is None else jax.numpy.where(m, val, 0.0)
            )

            noise.append(
                n if m is None else jax.numpy.where(m, n, 0.0)
            )
        else:
            samples.append(None)
            noise.append(None)

    sampled_params = jax.tree.unflatten(paux, samples)
    noise = jax.tree.unflatten(paux, noise)

    return sampled_params, noise

def _evon_sample_leaves(key, params, state, shape=(), mask=None):
    p_leaves, treedef = jax.tree.flatten(params)
    s_leaves = jax.tree.leaves(state.leaves, is_leaf=_is_evon_leaf)
    if mask is not None:
        m_leaves = treedef.flatten_up_to(mask)
    else:
        m_leaves = [None] * len(p_leaves)

    samples, noises = [], []
    for p, s, m in zip(p_leaves, s_leaves, m_leaves):
        key, rk = jax.random.split(key)
        V = 1.0 / (state.ess * (s.H + state.weight_decay))   # eigenbasis variance
        E = jax.numpy.sqrt(V) * jax.random.normal(rk, shape + s.H.shape, p.dtype)
        if isinstance(s, MatrixEvonLeaf):
            pert = _project_back(s.QL, s.QR, E).reshape(shape + p.shape)
        else:
            pert = E.reshape(shape + p.shape)
        if m is not None:
            pert = jax.numpy.where(m, pert, 0.0)
        samples.append(p + pert)
        noises.append(E)
    return jax.tree.unflatten(treedef, samples), noises


def evon_sample_and_grad(key, loss_fn, params, state, *args, mask=None, **kwargs):
    k_sample, k_loss = jax.random.split(key)
    samples, noises = _evon_sample_leaves(k_sample, params, state, mask=mask)
    out, grads = jax.value_and_grad(loss_fn, **kwargs)(samples, *args, k_loss)

    # stash transient noise back onto each leaf, in flattened order
    s_leaves = jax.tree.leaves(state.leaves, is_leaf=_is_evon_leaf)
    new_leaves = [s._replace(noise=n) for s, n in zip(s_leaves, noises)]
    # rebuild the leaves pytree using the params treedef (same structure)
    _, treedef = jax.tree.flatten(params)
    leaves = jax.tree.unflatten(treedef, new_leaves)
    return out, grads, state._replace(leaves=leaves)


def evon_hutchinson_and_grad(key, loss_fn, params, state, *args, mask=None, **kwargs):
    """Mode + Hutchinson estimator for EVON.

    Computes the gradient at the posterior mean ``params`` every step and the
    diagonal eigenbasis Hessian via one Rademacher probe and one Hessian-vector
    product (``jax.jvp`` of ``jax.value_and_grad``). The HVP runs only on steps
    where ``(count + 1) % hess_every == 0`` -- aligned with the eigenbasis refresh
    in ``scale_by_evon`` (which fires on the post-increment ``count``). On the
    intervening steps the HVP is skipped and ``h_hat`` is set to the current ``H``
    so the downstream EMA and quadratic correction leave ``H`` unchanged.

    The probe ``E`` is drawn in eigenbasis coordinates (shape ``leaf.H.shape``),
    projected to parameter space for the HVP, and the HVP is projected back, so
    ``h_hat = E ⊙ (Q_Lᵀ ∇²ℓ(M) Q_L E Q_Rᵀ Q_R)`` estimates ``diag(H°)`` in actual
    Hessian units (no ``pi(h)`` rescaling).
    """
    estimate_step = ((state.count + 1) % state.hess_every) == 0
    k_probe, k_loss = jax.random.split(key)

    def f(p):
        return loss_fn(p, *args, k_loss)

    p_leaves, treedef = jax.tree.flatten(params)
    s_leaves = jax.tree.leaves(state.leaves, is_leaf=_is_evon_leaf)
    if mask is not None:
        m_leaves = treedef.flatten_up_to(mask)
    else:
        m_leaves = [None] * len(p_leaves)

    def estimate():
        probes, tangents = [], []
        k = k_probe
        for p, s, m in zip(p_leaves, s_leaves, m_leaves):
            k, rk = jax.random.split(k)
            E = jax.random.rademacher(rk, shape=s.H.shape, dtype=p.dtype)
            if isinstance(s, MatrixEvonLeaf):
                Vp = _project_back(s.QL, s.QR, E).reshape(p.shape)
            else:
                Vp = E.reshape(p.shape)
            if m is not None:
                Vp = jax.numpy.where(m, Vp, 0.0)
            probes.append(E)
            tangents.append(Vp)
        Vp_tree = jax.tree.unflatten(treedef, tangents)
        (out, grad), (_, hvp) = jax.jvp(
            jax.value_and_grad(f, **kwargs), (params,), (Vp_tree,))
        hvp_leaves = treedef.flatten_up_to(hvp)
        h_hats = []
        for E, s, hv in zip(probes, s_leaves, hvp_leaves):
            if isinstance(s, MatrixEvonLeaf):
                d, o = s.H.shape
                HVo = _project(s.QL, s.QR, hv.reshape(d, o))
                h_hats.append(E * HVo)
            else:
                h_hats.append(E * hv)
        return out, grad, h_hats

    def reuse():
        out, grad = jax.value_and_grad(f, **kwargs)(params)
        h_hats = [s.H for s in s_leaves]
        return out, grad, h_hats

    out, grad, h_hats = lax.cond(estimate_step, estimate, reuse)

    new_leaves = [s._replace(noise=None, h_hat=h) for s, h in zip(s_leaves, h_hats)]
    leaves = jax.tree.unflatten(treedef, new_leaves)
    return out, grad, state._replace(leaves=leaves)


def tree_stack(pytree_list, axis=0):
    def stack(*args):
        return jax.numpy.stack(args, axis=axis)

    return jax.tree.map(stack, *pytree_list)

def sequential_sampling(key, loss_fn, params, state, mc_samples, *args, mask=None, **kwargs):
    g_bar_init = jax.tree.map(jax.numpy.zeros_like, params)
    h_bar_init = jax.tree.map(jax.numpy.zeros_like, params)

    def scan_body(carry, _):
        g_bar, h_bar, rn = carry
        rn, k1, k2 = jax.random.split(rn, 3)
        psample, noise = sample_posterior(k1, params, state, mask=mask)

        out, grad = jax.value_and_grad(loss_fn, **kwargs)(psample, *args, k2)
        
        g_bar = jax.tree.map(lambda a, g: a + g / mc_samples, g_bar, grad)
        h_bar = jax.tree.map(
            lambda h, g, n: h + g * n / mc_samples, h_bar, grad, noise
        )
        
        # The loss 'out' is the value we want to carry over and stack
        return (g_bar, h_bar, rn), out

    init_carry = (g_bar_init, h_bar_init, key)
    (g_bar, h_bar, _), output = lax.scan(scan_body, init_carry, jax.numpy.arange(mc_samples))

    h_bar = jax.tree.map(
        lambda hb, h: precision(h, state.ess, state.weight_decay) * hb, h_bar, state.hess
    )

    # Outside an estimation step, reuse the previous hessian (frozen EMA).
    estimate_step = (state.count % state.hess_every) == 0
    h_bar = jax.tree.map(
        lambda hb, h: jax.numpy.where(estimate_step, hb, h), h_bar, state.hess
    )

    return output, g_bar, state._replace(h_bar=h_bar)

def parallel_sampling(key, loss_fn, params, state, mc_samples, *args, mask=None, **kwargs):
    key, _key = jax.random.split(key)
    sampled_params, noise = sample_posterior(_key, params, state, shape=(mc_samples,), mask=mask)
    parallel_value_and_grad = jax.vmap(
        jax.value_and_grad(loss_fn, **kwargs), in_axes=(0,) + tuple([None] * len(args)) + (0,)
    )
    keys = jax.random.split(key, mc_samples)
    out, grads = parallel_value_and_grad(sampled_params, *args, keys)

    g_bar = jax.tree.map(lambda g: jax.numpy.mean(g, 0), grads)
    h_bar = jax.tree.map(lambda g, n: jax.numpy.mean(g * n, 0), grads, noise)

    h_bar = jax.tree.map(
        lambda hb, h: precision(h, state.ess, state.weight_decay) * hb, h_bar, state.hess
    )

    # Outside an estimation step, reuse the previous hessian (frozen EMA).
    estimate_step = (state.count % state.hess_every) == 0
    h_bar = jax.tree.map(
        lambda hb, h: jax.numpy.where(estimate_step, hb, h), h_bar, state.hess
    )

    return out, g_bar, state._replace(h_bar=h_bar)

def rademacher_like(key, params, mask=None):
    """Draw a Rademacher (+/-1) probe with the same structure as ``params``.

    Masked entries are zeroed and ``None`` leaves are preserved, mirroring the
    noise construction in ``sample_posterior``.
    """
    pleaves, paux = jax.tree.flatten(params, is_leaf=lambda x: x is None)

    if mask is not None:
        mleaves = jax.tree.leaves(mask, is_leaf=lambda x: x is None)
    else:
        mleaves = [None] * len(pleaves)

    probes = []
    for p, m in zip(pleaves, mleaves):
        if p is not None:
            key, rng = jax.random.split(key)
            u = jax.random.rademacher(rng, shape=p.shape, dtype=p.dtype)
            probes.append(u if m is None else jax.numpy.where(m, u, 0.0))
        else:
            probes.append(None)

    return jax.tree.unflatten(paux, probes)

def hutchinson_estimator(key, loss_fn, params, state, *args, mask=None, **kwargs):
    """Estimate loss value, gradient, and diagonal Hessian at the mean ``params``.

    The gradient ``∇L(μ)`` and the Hessian-vector product ``∇²L(μ)·u`` are
    produced by a single forward-over-reverse pass (``jax.jvp`` of
    ``jax.value_and_grad``). The diagonal Hessian is Hutchinson's estimator
    ``u ⊙ (∇²L·u)`` with one Rademacher probe ``u``; it is already in actual
    Hessian units and stored in ``h_bar`` without ``pi(h)`` rescaling.

    The Hessian is only (re)estimated on steps where ``count`` is a multiple of
    ``state.hess_every``. On the intervening steps the extra HVP pass is skipped
    -- only the gradient at the mean is computed -- and the previous hessian is
    returned so the optimizer's EMA leaves it unchanged.

    ``mc_samples`` and the sequential/parallel ``method`` axis do not apply here:
    a single probe is used and the gradient is deterministic at the mean.
    """
    k_probe, k_loss = jax.random.split(key)

    def f(p):
        return loss_fn(p, *args, k_loss)

    def estimate():
        u = rademacher_like(k_probe, params, mask=mask)
        (out, grad), (_, hvp) = jax.jvp(jax.value_and_grad(f, **kwargs), (params,), (u,))
        h_bar = jax.tree.map(lambda ui, hi: ui * hi, u, hvp)
        return out, grad, h_bar

    def reuse():
        # Skip the HVP pass: compute only the gradient at the mean and hand back
        # the previous hessian (the EMA then reduces to the identity).
        out, grad = jax.value_and_grad(f, **kwargs)(params)
        h_bar = jax.tree.map(lambda h, g: h.astype(g.dtype), state.hess, grad)
        return out, grad, h_bar

    estimate_step = (state.count % state.hess_every) == 0
    out, grad, h_bar = lax.cond(estimate_step, estimate, reuse)

    return out, grad, state._replace(h_bar=h_bar)

@eqx.filter_jit
def noisy_value_and_grad(loss_fn, opt_state, params, key, *args, mc_samples=1, mask=None, method='sequential', estimator='sampling', **kwargs):
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
        mc_samples: Number of posterior samples drawn for ``estimator='sampling'``;
          ignored when ``estimator='hutchinson'`` (a single Rademacher probe is used).
        mask: PyTree or Array of the same structure as params, used to mask gradients and random samples.
        method: For ``estimator='sampling'``, selects how the monte carlo samples are
          drawn: ``'sequential'`` (scan) or ``'parallel'`` (vmap).
        estimator: ``'sampling'`` (default) for the reparameterization estimates
          around posterior samples, or ``'hutchinson'`` for a variational-Laplace
          estimate at the mean using Hutchinson's diagonal Hessian estimator. With
          ``'hutchinson'`` the ``mc_samples`` and ``method`` arguments are ignored.
          Applies to both IVON and EVON; for EVON the Hutchinson HVP is gated by
          ``hess_every`` (estimated in the eigenbasis, frozen on other steps).

    Returns:
        loss_value: Array,
        updates: Gradient estimates
        state: optax type state.
    """
    ivon_state = opt_state[0]
    if isinstance(ivon_state, ScaleByEvonState):
        if estimator == 'hutchinson':
            out, updates, ivon_state = evon_hutchinson_and_grad(
                key, loss_fn, params, ivon_state, *args, mask=mask, **kwargs)
        else:
            out, updates, ivon_state = evon_sample_and_grad(
                key, loss_fn, params, ivon_state, *args, mask=mask, **kwargs)
        opt_state = (ivon_state,) + opt_state[1:]
    elif isinstance(ivon_state, ScaleByIvonState):
        if estimator == 'hutchinson':
            out, updates, ivon_state = hutchinson_estimator(key, loss_fn, params, ivon_state, *args, mask=mask, **kwargs)
        elif method == 'parallel':
            out, updates, ivon_state = parallel_sampling(key, loss_fn, params, ivon_state, mc_samples, *args, mask=mask, **kwargs)
        else:
            out, updates, ivon_state = sequential_sampling(key, loss_fn, params, ivon_state, mc_samples, *args, mask=mask, **kwargs)
        opt_state = (ivon_state,) + opt_state[1:]
    else:
        out, updates = jax.value_and_grad(loss_fn, **kwargs)(params, *args, key)

    return out, updates, opt_state