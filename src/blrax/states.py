import chex
from typing import NamedTuple
from optax._src import base

class ScaleByIvonState(NamedTuple):
  """State for the algorithm."""
  key: chex.PRNGKey
  count: chex.Array  # shape=(), dtype=jnp.int32.
  g: base.Updates
  h: base.Updates
  eps: base.Updates
  num_datapoints: int
  mc_samples: int
  weight_decay: chex.Array # shape=(), dtype=jnp.float32.