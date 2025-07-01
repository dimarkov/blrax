from typing import Any, NamedTuple, Optional, Union

import chex
import jax
import optax
import jax.numpy as jnp

from jax import random as jr
from optax._src import utils
from optax import tree_utils as otu

from blrax.utils import tree_random_like
from blrax.states import ScaleByIvonState

ScalarOrSchedule = Union[float, chex.Array, optax.Schedule]

def update_hessian(hess, bar_hess, ess, decay, delta):
  """Compute the exponential moving average of the hessian while maintaing positivity contraint."""
  hat_hess = jax.tree_util.tree_map(lambda a, h: ess * a * (h + delta), bar_hess, hess)

  func = lambda h, t: decay * h + (1 - decay) * t + .5 * jnp.square( (1 - decay) * (h - t) ) / (h + delta)
  return jax.tree_util.tree_map(func, hess, hat_hess)

def scale_by_ivon(
    ess: float,
    hess_init: float,
    b1: float = 0.9,
    b2: float = 0.99999,
    weight_decay: float = 1e-4,
    m_dtype: Optional[chex.ArrayDType] = None
) -> optax.GradientTransformation:
  """Rescale updates according to the IVON algorithm.

  References:
    [Shen et al, 2024](https://arxiv.org/abs/2402.17641)

  Args:
    key: Random number seed.
    ess: Effective sample size.
    hess_init: Initial value of the hessian for every parameter.
    mc_samples: Number of monte carlo samples from approximate posterior.
    b1: Decay rate for the exponentially weighted average of grads.
    b2: Decay rate for the exponentially weighted average of squared grads.
    weight_decay: Weight decay coefficient.
    m_dtype: Optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.

  Returns:
    A `GradientTransformation` object.
  """

  m_dtype = utils.canonicalize_dtype(m_dtype)
  def init_fn(params):
    m = otu.tree_zeros_like(params, dtype=m_dtype)  # First moment
    h = otu.tree_full_like(params, hess_init, dtype=m_dtype)  # Second moment (hessian)
    h_bar = otu.tree_zeros_like(params, dtype=m_dtype)
    return ScaleByIvonState(
      count=jnp.zeros([], jnp.int32),
      momentum=m,
      hess=h,
      ess=ess,
      weight_decay=weight_decay,
      h_bar=h_bar
    )

  def update_fn(updates, state, params):
    g_bar = updates
    h_bar = state.h_bar
    momentum = otu.tree_update_moment(g_bar, state.momentum, b1, 1.)
    hess = update_hessian(state.hess, h_bar, state.ess, b2, state.weight_decay)
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
      h_bar=otu.tree_zeros_like(h_bar, dtype=m_dtype)
    )

  return optax.GradientTransformation(init_fn, update_fn)

def ivon(
    learning_rate: ScalarOrSchedule,
    ess: float = 1.,
    hess_init: float = 1.,
    clip_radius: float = float("inf"),
    b1: float = 0.9,
    b2: float = 0.99999,
    weight_decay: float = 1e-3,
    rescale_lr: bool = True,
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
    rescale_lr: Optional bool to switch rescaling of learning rate by a sum of initial hessian and weight decay.
    m_dtype: Optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.

  Returns:
    The corresponding `GradientTransformation`.
  """

  ivon_trans = scale_by_ivon(ess, hess_init, b1=b1, b2=b2, weight_decay=weight_decay, m_dtype=m_dtype)

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
