import chex
from typing import NamedTuple, Optional
import optax

class ScaleByIvonState(NamedTuple):
  """State for the algorithm."""
  count: chex.Array  # shape=(), dtype=jnp.int32
  momentum: optax.Updates
  hess: optax.Updates
  ess: chex.Array  # shape=(), dtype=jnp.float32
  weight_decay: chex.Array # shape=(), dtype=jnp.float32
  hess_every: chex.Array = 1  # estimate/refresh the hessian every k steps
  h_bar: Optional[optax.Updates] = None
