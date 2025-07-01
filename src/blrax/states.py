import chex
from typing import NamedTuple, Optional
import optax

class ScaleByIvonState(NamedTuple):
  """State for the algorithm."""
  count: chex.Array  # shape=(), dtype=jnp.int32
  momentum: optax.Updates
  hess: optax.Updates
  h_bar: optax.Updates
  ess: chex.Array  # shape=(), dtype=jnp.float32
  weight_decay: chex.Array # shape=(), dtype=jnp.float32
  noise: Optional[optax.Updates] = None
