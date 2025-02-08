from functools import partial
from typing import Any, NamedTuple, Optional, Union

import chex
import jax
import jax.numpy as jnp

from jax import random as jr
from optax._src import base
from optax._src import numerics
from optax._src import utils
from optax import scale_by_learning_rate, chain, tree_utils as otu

from blrax.utils import tree_random_like
from blrax.states import ScaleByIvonState

ScalarOrSchedule = Union[float, chex.Array, base.Schedule]

def update_hessian(moments, updates, decay, delta):
  """Compute the exponential moving average of the hessian while maintaing positivity contraint."""
  func = lambda h, t: decay * h + (1 - decay) * t + .5 * jnp.square( (1 - decay) * (h - t) ) / (h + delta)
  return jax.tree_util.tree_map(func, moments, updates)  

def scale_by_ivon(
    key: chex.PRNGKey,
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
    mc_samples: number of monte carlo samples from approximate posterior
    b1: Decay rate for the exponentially weighted average of grads.
    b2: Decay rate for the exponentially weighted average of squared grads.
    m_dtype: Optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.

  Returns:
    A `GradientTransformation` object.
  """

  m_dtype = utils.canonicalize_dtype(m_dtype)
  wd = s0 / num_data  # weight decay

  def init_fn(rng_key, params):
    g = otu.tree_zeros_like(params, dtype=m_dtype)  # First moment
    h = otu.tree_full_like(params, h0, dtype=m_dtype)  # Second moment (hessian)
    key1, key2 = jr.split(rng_key)
    eps = tree_random_like(key1, target_tree=params, add_shape=(mc_samples,), sampler=jr.normal)  # noise
    return ScaleByIvonState(
      key=key2,
      count=jnp.zeros([], jnp.int32),
      h=h,
      g=g,
      eps=eps,
      num_datapoints=num_data,
      mc_samples=mc_samples,
      weight_decay=wd * jnp.ones([])
    )

  def update_fn(updates, state, params):
    otu.tree_update_moment
    g_bar = jax.tree_util.tree_map(
      lambda g: jnp.mean(g, 0), updates
    )
    g = otu.tree_update_moment(g_bar, state.g, 1 - b1, 1.)
    hat_h = jax.tree_util.tree_map(
      lambda g, noise: jnp.mean(noise * g, 0), updates, state.eps
    )

    h = update_hessian(state.h, hat_h, b2, wd)
    count_inc = numerics.safe_int32_increment(state.count)
    g_hat = otu.tree_bias_correction(g, b1, count_inc)
    updates = jax.tree_util.tree_map(
        lambda g, mu, h: jnp.clip(g + wd * mu, min=-clip_radius, max=clip_radius) / (h + wd), g_hat, params, h)
    
    g = otu.tree_cast(g, m_dtype)

    _key, key = jr.split(state.key)
    eps = tree_random_like(_key, target_tree=params, add_shape=(mc_samples,), sampler=jr.normal)
    return updates, ScaleByIvonState(key=key, count=count_inc, g=g, h=h, eps=eps, weight_decay=wd, num_datapoints=num_data, mc_samples=mc_samples)

  return base.GradientTransformation(partial(init_fn, key), update_fn)

def ivon(
    key: chex.PRNGKey,
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
    key: Random number seed.
    learning_rate: A fixed global scaling factor.
    s0: Prior precision.
    h0: Initial value of the hessian for every parameter
    num_data: Number of data points in the training set.
    mc_samples: Number of monte carlo samples from the posterior.
    clip_radius: Maximal absolute value of the change in the first moment.
    b1: Exponential decay rate to track the first moment of past gradients.
    b2: Exponential decay rate to track the second moment of past gradients.
    m_dtype: Optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.

  Returns:
    The corresponding `GradientTransformation`.
  """
  return chain(
      scale_by_ivon(key, s0, h0, num_data, mc_samples, clip_radius=clip_radius, b1=b1, b2=b2, m_dtype=m_dtype),
      scale_by_learning_rate(learning_rate),
  )