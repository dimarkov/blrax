from typing import Any, NamedTuple, Optional, Union
import math
import functools

import chex
import jax
import optax
import jax.numpy as jnp

from jax import lax, random as jr
from optax._src import utils
from optax import tree_utils as otu

from blrax.states import (
    ScaleByIvonState,
    ScaleByEvonState, MatrixEvonLeaf, DiagEvonLeaf, _is_evon_leaf,
)
from blrax.utils import _project, _project_back, precision

ScalarOrSchedule = Union[float, chex.Array, optax.Schedule]

def update_hessian(hess, t, decay, wd):
  """Compute the exponential moving average of the hessian while maintaing positivity contraint.

  ``t`` is the per-step diagonal-Hessian estimate already expressed in actual
  Hessian units (the sampling estimators apply the ``pi(h)`` rescaling before
  calling this; the Hutchinson estimator returns it directly).
  """

  def func(h, _t):
    v = decay * h + (1 - decay) * _t
    v += 0.5 * jnp.square( (1 - decay) * (h - _t) ) / (h + wd)

    return v

  return jax.tree.map(func, hess, t)

def scale_by_ivon(
    ess: float,
    hess_init: float,
    b1: float = 0.9,
    b2: float = 0.99999,
    weight_decay: float = 1e-4,
    hess_every: int = 1,
    m_dtype: Optional[chex.ArrayDType] = None
) -> optax.GradientTransformation:
  """Rescale updates according to the IVON algorithm.

  References:
    [Shen et al, 2024](https://arxiv.org/abs/2402.17641)

  Args:
    ess: Effective sample size.
    hess_init: Initial value of the hessian for every parameter.
    b1: Decay rate for the exponentially weighted average of grads.
    b2: Decay rate for the exponentially weighted average of squared grads.
    weight_decay: Weight decay coefficient.
    hess_every: Refresh the hessian every ``hess_every`` steps (``1`` = every
      step). On the intervening steps the estimator returns the previous hessian
      (so the EMA below is an exact no-op) and, for the Hutchinson estimator,
      skips the extra Hessian-vector-product pass entirely.
    m_dtype: Optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.

  Returns:
    A `GradientTransformation` object.
  """

  m_dtype = utils.canonicalize_dtype(m_dtype)
  def init_fn(params):
    momentum = otu.tree_zeros_like(params, dtype=m_dtype)
    hessian = otu.tree_full_like(params, hess_init, dtype=m_dtype)
    return ScaleByIvonState(
      count=jnp.zeros([], jnp.int32),
      momentum=momentum,
      hess=hessian,
      ess=ess,
      weight_decay=weight_decay,
      hess_every=jnp.asarray(hess_every, jnp.int32),
    )

  def update_fn(updates, state, params):
    g_bar = updates
    h_bar = state.h_bar
    momentum = otu.tree_update_moment(g_bar, state.momentum, b1, 1.)
    # On skipped steps the estimator hands back the previous hessian, so this
    # EMA reduces to the identity and the hessian is reused unchanged.
    hess = update_hessian(state.hess, h_bar, b2, state.weight_decay)
    count = optax.safe_increment(state.count)
    bias_correction = 1 - b1**count
    updates = jax.tree_util.tree_map(
        lambda m, p, h: (m / bias_correction + state.weight_decay * p) / (h + state.weight_decay), momentum, params, hess)

    momentum = otu.tree_cast(momentum, m_dtype)

    return updates, ScaleByIvonState(
      count=count,
      momentum=momentum,
      hess=hess,
      ess=state.ess,
      weight_decay=state.weight_decay,
      hess_every=state.hess_every,
    )

  return optax.GradientTransformation(init_fn, update_fn)

def ivon(
    learning_rate: ScalarOrSchedule,
    ess: float = 1.,
    hess_init: float = 1.,
    clip_radius: float = float("inf"),
    b1: float = 0.9,
    b2: float = 0.99999,
    weight_decay: float = 1e-4,
    hess_every: int = 1,
    rescale_lr: bool = False,
    m_dtype: Optional[Any] = None,
) -> optax.GradientTransformation:
  r"""The improved variational online Newton (IVON) optimizer.

  IVON is a stochastic natural gradient variant with gradient scaling adaptation. The scaling
  used for each parameter is computed from estimates of first and second-order
  moments of the gradients (using suitable exponential moving averages).

  Let :math:`\alpha_t` represent the learning rate and :math:`\beta_1, \beta_2`, represent the arguments
  ``b1``, ``b2``, respectievly. The learning rate is indexed by :math:`t` since the learning rate may also be provided by a
  schedule function.

  The ``init`` function of this optimizer initializes an internal state
  :math:`S_0 := (g_0, h_0) = (0, h0)`, representing initial estimates for the first and second moments. In practice these values are stored as pytrees
  , with the same shape as the model updates.
  At step :math:`t`, the ``update`` function of this optimizer takes as
  arguments the incoming (sample of) gradients :math:`g_t` and optimizer state :math:`S_t`
  and computes updates :math:`u_t` and new state :math:`S_{t+1}`. Thus, for
  :math:`t > 0`, we have,

  .. math::
    \begin{align*}
      \delta &\leftarrow s0 / N \\
      \bar{g} &\leftarrow \sum_s g_s / S \\
      \bar{h} &\leftarrow \sum_s g_s \epsilon_s / S \\
      g &\leftarrow b1 \cdot g + ( 1 - b1 ) \cdot \bar{g} \\
      h &\leftarrow \beta_2 \cdot h + (1-\beta_2) \cdot \bar{h} + 0.5 \cdot (1-\beta_2)^2 (h - \bar{h})^2 /(h + \delta) \\
      \hat{g} &\leftarrow g / {(1-\beta_1^t)} \\
      mu &\leftarrow mu - \alpha_t \cdot ( \hat{g} + \delta \mu ) / (h + \delta) \\
      \sigma &\leftarrow 1 / \sqrt{N h + s0} \\
      S_t &\leftarrow (g_t, h_t).
    \end{align*}

  References:
    Shen et al, 2024: https://arxiv.org/abs/2402.17641

  Args:
    learning_rate: A fixed global scaling factor.
    ess: Effective sample size.
    hess_init: Initial value of the hessian for every parameter.
    clip_radius: Maximal absolute value of the change in the first moment.
    b1: Exponential decay rate to track the first moment of past gradients.
    b2: Exponential decay rate to track the second moment of past gradients.
    weight_decay: Weight decay coefficient.
    hess_every: Estimate the hessian every ``hess_every`` steps (``1`` = every
      step). Larger values reuse the previous hessian on intervening steps,
      speeding up the Hutchinson estimator by skipping its extra Hessian-vector
      product on those steps. The gradient is still computed every step.
    rescale_lr: Optional bool to switch rescaling of learning rate by a sum of initial hessian and weight decay.
    m_dtype: Optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.

  Returns:
    The corresponding `GradientTransformation`.
  """

  ivon_trans = scale_by_ivon(ess, hess_init, b1=b1, b2=b2, weight_decay=weight_decay, hess_every=hess_every, m_dtype=m_dtype)

  if rescale_lr:
    lr_scale = (
      optax.scale_by_learning_rate(learning_rate),
      optax.scale(hess_init + weight_decay),
    )
  else:
    lr_scale = (optax.scale_by_learning_rate(learning_rate),)

  if clip_radius < float("inf"):
    transform = optax.chain(
      ivon_trans,
      optax.clip(clip_radius),
      *lr_scale,
    )
  else:
    transform = optax.chain(ivon_trans, *lr_scale)

  return transform


def _qr_power_iter(M32, Q32):
    """One warm-started power-iteration step; float32 in, float32 out."""
    eig_est = jnp.diag(Q32.T @ M32 @ Q32)
    idx = jnp.argsort(eig_est, descending=True)
    Qs = Q32[:, idx]
    Qn, _ = jnp.linalg.qr(M32 @ Qs)
    return Qn


def _refresh_side(M, Q, do_refresh, first):
    """Return the (possibly refreshed) eigenbasis for one side. M, Q non-None."""
    M32 = M.astype(jnp.float32)
    Q32 = Q.astype(jnp.float32)

    def refresh():
        def via_eigh():
            _, vecs = jnp.linalg.eigh(M32)
            return vecs[:, ::-1]              # descending eigenvalue order
        return lax.cond(first, via_eigh, lambda: _qr_power_iter(M32, Q32))

    Qn32 = lax.cond(do_refresh, refresh, lambda: Q32)
    return Qn32.astype(Q.dtype)


def _update_diag_leaf(g, p, leaf, ess, wd, b1, b2):
    Hhat = leaf.h_hat if leaf.h_hat is not None else (leaf.noise * g * precision(leaf.H, ess, wd))
    G_bar = b1 * leaf.G_bar + (1 - b1) * g
    H = update_hessian(leaf.H, Hhat, b2, wd)
    U = (G_bar + wd * p) / (H + wd)
    return U, DiagEvonLeaf(H=H, G_bar=G_bar, noise=None, h_hat=None)


def _update_matrix_leaf(g, p, leaf, ess, wd, b1, b2, b3, count, precond_every):
    orig_shape = g.shape
    d, o = leaf.H.shape
    G = g.reshape(d, o)
    M = p.reshape(d, o)

    Go = _project(leaf.QL, leaf.QR, G)
    if leaf.h_hat is not None:
        Hhat = leaf.h_hat
    else:
        Hhat = leaf.noise * Go * precision(leaf.H, ess, wd)
    G_bar = b1 * leaf.G_bar + (1 - b1) * Go
    H = update_hessian(leaf.H, Hhat, b2, wd)
    Mo = _project(leaf.QL, leaf.QR, M)
    U = (G_bar + wd * Mo) / (H + wd)
    delta = _project_back(leaf.QL, leaf.QR, U).reshape(orig_shape)

    # preconditioner EMAs (raw G), present sides only
    L = (b3 * leaf.L + (1 - b3) * (G @ G.T)) if leaf.L is not None else None
    R = (b3 * leaf.R + (1 - b3) * (G.T @ G)) if leaf.R is not None else None

    # eigenbasis refresh + momentum rotation (after the param update)
    do_refresh = (count % precond_every) == 0
    first = count == precond_every
    QL_new = _refresh_side(L, leaf.QL, do_refresh, first) if leaf.QL is not None else None
    QR_new = _refresh_side(R, leaf.QR, do_refresh, first) if leaf.QR is not None else None
    if leaf.QL is not None:
        G_bar = (QL_new.T @ leaf.QL) @ G_bar
    if leaf.QR is not None:
        G_bar = G_bar @ (leaf.QR.T @ QR_new)

    return delta, MatrixEvonLeaf(L=L, R=R, QL=QL_new, QR=QR_new,
                                 H=H, G_bar=G_bar, noise=None, h_hat=None)


def _make_leaf(p, hess_init, max_precond_dim, one_sided, m_dtype=None):
    """Static per-leaf construction from the parameter array ``p``.

    The first-moment accumulator ``G_bar`` is stored in ``m_dtype`` (falling
    back to ``p.dtype``); the Hessian and the eigenbasis factors stay in
    ``p.dtype`` (the eigenbasis refresh runs in float32 regardless).
    """
    g_dtype = m_dtype if m_dtype is not None else p.dtype
    if p.ndim < 2:
        return DiagEvonLeaf(H=jnp.full(p.shape, hess_init, p.dtype),
                            G_bar=jnp.zeros(p.shape, g_dtype))
    d = math.prod(p.shape[:-1])
    o = p.shape[-1]
    left_ok = d <= max_precond_dim
    right_ok = o <= max_precond_dim
    if not left_ok and not right_ok:
        return DiagEvonLeaf(H=jnp.full(p.shape, hess_init, p.dtype),
                            G_bar=jnp.zeros(p.shape, g_dtype))
    if one_sided and left_ok and right_ok:
        # keep only the smaller axis (on a tie d == o either side is equally
        # cheap; we keep the left one)
        if d <= o:
            right_ok = False
        else:
            left_ok = False
    L = jnp.zeros((d, d), p.dtype) if left_ok else None
    QL = jnp.eye(d, dtype=p.dtype) if left_ok else None
    R = jnp.zeros((o, o), p.dtype) if right_ok else None
    QR = jnp.eye(o, dtype=p.dtype) if right_ok else None
    return MatrixEvonLeaf(L=L, R=R, QL=QL, QR=QR,
                          H=jnp.full((d, o), hess_init, p.dtype),
                          G_bar=jnp.zeros((d, o), g_dtype))


def scale_by_evon(
    ess: float,
    hess_init: float,
    b1: float = 0.9,
    b2: float = 0.95,
    b3: float = 0.95,
    weight_decay: float = 1e-4,
    precond_every: int = 10,
    hess_every: int = None,
    max_precond_dim: int = 10000,
    one_sided: bool = False,
    m_dtype=None,
) -> optax.GradientTransformation:
    """Rescale updates according to the EVON algorithm (Minut et al., 2026)."""
    m_dtype = utils.canonicalize_dtype(m_dtype)
    if hess_every is None:
        hess_every = precond_every

    def init_fn(params):
        leaves = jax.tree.map(
            functools.partial(_make_leaf, hess_init=hess_init,
                              max_precond_dim=max_precond_dim, one_sided=one_sided,
                              m_dtype=m_dtype),
            params,
        )
        return ScaleByEvonState(
            count=jnp.zeros([], jnp.int32),
            ess=ess,
            weight_decay=weight_decay,
            precond_every=jnp.asarray(precond_every, jnp.int32),
            hess_every=jnp.asarray(hess_every, jnp.int32),
            leaves=leaves,
        )

    def update_fn(updates, state, params):
        count = optax.safe_increment(state.count)
        p_leaves, treedef = jax.tree.flatten(params)
        g_leaves = treedef.flatten_up_to(updates)
        s_leaves = jax.tree.leaves(state.leaves, is_leaf=_is_evon_leaf)

        new_updates, new_s = [], []
        for g, p, s in zip(g_leaves, p_leaves, s_leaves):
            if isinstance(s, MatrixEvonLeaf):
                delta, s_new = _update_matrix_leaf(
                    g, p, s, state.ess, state.weight_decay, b1, b2, b3,
                    count, state.precond_every)
            else:
                delta, s_new = _update_diag_leaf(
                    g, p, s, state.ess, state.weight_decay, b1, b2)
            # keep the first-moment accumulator in m_dtype (computed in the
            # promoted dtype above); mirrors IVON's momentum cast.
            if m_dtype is not None:
                s_new = s_new._replace(G_bar=s_new.G_bar.astype(m_dtype))
            new_updates.append(delta)
            new_s.append(s_new)

        updates = jax.tree.unflatten(treedef, new_updates)
        leaves = jax.tree.unflatten(treedef, new_s)
        return updates, ScaleByEvonState(
            count=count, ess=state.ess, weight_decay=state.weight_decay,
            precond_every=state.precond_every, hess_every=state.hess_every,
            leaves=leaves)

    return optax.GradientTransformation(init_fn, update_fn)


def evon(
    learning_rate: ScalarOrSchedule,
    ess: float = 1.,
    hess_init: float = 1.,
    clip_radius: float = float("inf"),
    b1: float = 0.9,
    b2: float = 0.95,
    b3: float = 0.95,
    weight_decay: float = 1e-4,
    precond_every: int = 10,
    hess_every: int = None,
    max_precond_dim: int = 10000,
    one_sided: bool = False,
    m_dtype=None,
) -> optax.GradientTransformation:
    """The Eigenspace Variational Online Newton (EVON) optimizer.

    EVON runs IVON inside SOAP's eigenbasis per 2-D weight matrix, yielding a
    structured (non-diagonal) Gaussian posterior. See Minut et al., 2026
    (arXiv:2606.23357).
    """
    evon_trans = scale_by_evon(
        ess, hess_init, b1=b1, b2=b2, b3=b3, weight_decay=weight_decay,
        precond_every=precond_every, hess_every=hess_every,
        max_precond_dim=max_precond_dim, one_sided=one_sided, m_dtype=m_dtype)
    lr_scale = (optax.scale_by_learning_rate(learning_rate),)
    if clip_radius < float("inf"):
        return optax.chain(evon_trans, optax.clip(clip_radius), *lr_scale)
    return optax.chain(evon_trans, *lr_scale)
