import functools
from typing import Any, NamedTuple, Optional, Union

import chex
import jax
import jax.numpy as jnp

from jax import random as jr
from jaxtyping import Array, PRNGKeyArray

from optax._src import base
from optax._src import numerics
from optax._src import utils
from optax import scale_by_learning_rate, chain

from .utils import random_split_like_tree

_abs_sq = numerics.abs_sq

ScalarOrSchedule = Union[float, Array, base.Schedule]

@functools.partial(jax.jit, inline=True)
def bias_correction(moment, decay, count):
  """Performs bias correction. It becomes a no-op as count goes to infinity."""
  # The conversion to the data type of the moment ensures that bfloat16 remains
  # bfloat16 in the optimizer state. This conversion has to be done after
  # `bias_correction_` is calculated as calculating `decay**count` in low
  # precision can result in it being rounded to 1 and subsequently a
  # "division by zero" error.
  bias_correction_ = 1 - decay**count

  # Perform division in the original precision.
  return jax.tree_util.tree_map(
    lambda t: t / bias_correction_.astype(t.dtype), moment)

def update_grad(updates, moments, decay):
  """Compute the exponential moving average of the `order`-th moment."""
  return jax.tree_util.tree_map(
      lambda g, t: decay * g  + (1 - decay) * t, updates, moments)

def update_hessian(moments, updates, decay, delta):
  """Compute the exponential moving average of the hessian while maintaing positivity contraint."""
  func = lambda h, t: decay * h + (1 - decay) * t + .5 * jnp.square( (1 - decay) * (h - t) ) / (h + delta)
  return jax.tree_util.tree_map(func, moments, updates)

class ScaleByState(NamedTuple):
  """State for the algorithms."""
  key: chex.PRNGKey
  count: chex.Array  # shape=(), dtype=jnp.int32.
  g: base.Updates
  h: base.Updates
  eps: base.Updates
  sample_params: bool = True


def scale_by_ivon(
    key: PRNGKeyArray,
    s0: float,
    h0: float,
    num_data: int,
    mc_samples: int = 1,
    clip_radius: float = jnp.inf,
    b1: float = 0.9,
    b2: float = 0.99999,
    m_dtype: Optional[chex.ArrayDType] = None
) -> base.GradientTransformation:
  """Rescale updates according to the IVON algorithm.

  References:
    [Shen et al, 2024](https://arxiv.org/abs/2402.17641)

  Args:
    s0: prior precision.
    h0: initial hessian.
    num_data: number of data points.
    b1: Decay rate for the exponentially weighted average of grads.
    b2: Decay rate for the exponentially weighted average of squared grads.
    m_dtype: Optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.

  Returns:
    A `GradientTransformation` object.
  """

  m_dtype = utils.canonicalize_dtype(m_dtype)
  wd = s0 / num_data  # weight decay

  def init_fn(key, params):
    g = jax.tree_util.tree_map(lambda t:  jnp.zeros_like(t), params)  # Second moment
    h = jax.tree_util.tree_map(lambda t: h0 * jnp.ones_like(t, dtype=m_dtype), params)  # First moment
    keys, key = random_split_like_tree(key, target=params)
    eps = jax.tree_util.tree_map(lambda t, key: jr.normal(key, shape=(mc_samples,) + t.shape), params, keys)
    return ScaleByState(key=key, count=jnp.zeros([], jnp.int32), h=h, g=g, eps=eps)

  def update_fn(updates, state, params):
    g_bar = jax.tree_util.tree_map(
      lambda g: jnp.mean(g, 0), updates
    )
    g = update_grad(g_bar, state.g, b1)
    hat_h = jax.tree_util.tree_map(
      lambda g, noise: jnp.mean(noise * g, 0), updates, state.eps
    )

    h = update_hessian(state.h, hat_h, b2, wd)
    count_inc = numerics.safe_int32_increment(state.count)
    g_hat = bias_correction(g, b1, count_inc)
    updates = jax.tree_util.tree_map(
        lambda m, mu, v: jnp.clip(m + wd * mu, min=-clip_radius, max=clip_radius) / (v + wd), g_hat, params, h)
    
    g = utils.cast_tree(g, m_dtype)

    keys, key = random_split_like_tree(state.key, target=params)
    eps = jax.tree_util.tree_map(lambda t, key: jr.normal(key, shape=(mc_samples,) + t.shape), params, keys)
    return updates, ScaleByState(key=key, count=count_inc, g=g, h=h, eps=eps)

  return base.GradientTransformation(functools.partial(init_fn, key), update_fn)

def ivon(
    key: PRNGKeyArray,
    learning_rate: ScalarOrSchedule,
    s0: float = 1.,
    h0: float = 1.,
    num_data: int = 1e4,
    mc_samples: int = 1,
    clip_radius: float = jnp.inf,
    b1: float = 0.9,
    b2: float = 0.99999,
    m_dtype: Optional[Any] = None,
) -> base.GradientTransformation:
  r"""The improved variational online Newton (IVON) optimizer.

  IVON is a stochastic natural gradient variant with gradient scaling adaptation. The scaling
  used for each parameter is computed from estimates of first and second-order
  moments of the gradients (using suitable exponential moving averages).

  Let :math:`\alpha_t` represent the learning rate and :math:`\beta_1, \beta_2`, represent the arguments
  ``b1``, ``b2``, respectievly. The learning rate is indexed by :math:`t` since the learning rate may also be provided by a
  schedule function.

  The ``init`` function of this optimizer initializes an internal state
  :math:`S_0 := (m_0, h_0) = (0, h0)`, representing initial estimates for the first and second moments. In practice these values are stored as pytrees
  , with the same shape as the model updates.
  At step :math:`t`, the ``update`` function of this optimizer takes as
  arguments the incoming (sample of) gradients :math:`g_t` and optimizer state :math:`S_t`
  and computes updates :math:`u_t` and new state :math:`S_{t+1}`. Thus, for
  :math:`t > 0`, we have,

  .. math::
    \begin{align*}
      m_t &\leftarrow m_{t-1} - \alpha_t \cdot ( \bar{g}_t + \delta \mu ) \\
      s_t &\leftarrow \beta_2 \cdot s_{t-1} + (1-\beta_2) \cdot {g_t}^2 \\
      \hat{m}_t &\leftarrow m_t / {(1-\beta_1^t)} \\
      \hat{s}_t &\leftarrow s_t / {(1-\beta_2^t)} \\
      u_t &\leftarrow \alpha_t \cdot \hat{m}_t / \left({\sqrt{\hat{v}_t +
      \bar{\varepsilon}} + \tilde{\lambda}} \right)\\
      S_t &\leftarrow (m_t, s_t).
    \end{align*}

  References:
    Shen et al, 2024: https://arxiv.org/abs/2402.17641

  Args:
    key: Random number seed.
    learning_rate: A fixed global scaling factor.
    t_lam: Prior precision devided by the number of data points in the training set.
    t_init: Initial precision of the variational posterior q devided by the number of data points in the training set.
    N: Number of data points in the training set.
    b1: Exponential decay rate to track the first moment of past gradients.
    b2: Exponential decay rate to track the second moment of past gradients.
    eps_root: A small constant applied to denominator inside the square root (as
      in RMSProp), to avoid dividing by zero when rescaling. This is needed for
      example when computing (meta-)gradients through Vadam.
    m_dtype: Optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.

  Returns:
    The corresponding `GradientTransformation`.
  """
  return chain(
      scale_by_ivon(key, s0, h0, num_data, mc_samples, b1=b1, b2=b2, m_dtype=m_dtype),
      scale_by_learning_rate(learning_rate),
  )