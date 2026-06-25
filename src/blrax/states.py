import chex
from typing import NamedTuple, Optional, Any
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


class MatrixEvonLeaf(NamedTuple):
  """Per 2-D weight-matrix EVON state (shape (d, o) after any reshape).

  A side's preconditioner (L, QL) / (R, QR) is either present or ``None``; a
  ``None`` side is treated diagonally (identity projection on that axis). ``H``
  and ``G_bar`` are (d, o) in the (partial) eigenbasis. ``noise`` (the eigenbasis
  draw ``E``) is transient: ``None`` at rest, set by the sampler, cleared by
  ``update``.
  """
  L: Optional[chex.Array]
  R: Optional[chex.Array]
  QL: Optional[chex.Array]
  QR: Optional[chex.Array]
  H: chex.Array
  G_bar: chex.Array
  noise: Optional[chex.Array] = None


class DiagEvonLeaf(NamedTuple):
  """Diagonal-IVON fallback leaf (1-D / scalar / over-cutoff). The ``Q=I`` case."""
  H: chex.Array
  G_bar: chex.Array
  noise: Optional[chex.Array] = None


class ScaleByEvonState(NamedTuple):
  """State for the EVON algorithm.

  ``leaves`` mirrors ``params`` with each array leaf replaced by a
  ``MatrixEvonLeaf`` or ``DiagEvonLeaf``. The posterior mean ``M`` is the optax
  ``params`` and is not stored here.
  """
  count: chex.Array            # shape=(), dtype=jnp.int32
  ess: chex.Array              # zeta
  weight_decay: chex.Array     # delta
  precond_every: chex.Array    # T (eigenbasis refresh period)
  leaves: Any


def _is_evon_leaf(x) -> bool:
  return isinstance(x, (MatrixEvonLeaf, DiagEvonLeaf))
