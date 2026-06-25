import jax
import jax.numpy as jnp
import unittest
from jax import random as jr

from blrax.states import (
    MatrixEvonLeaf, DiagEvonLeaf, ScaleByEvonState, _is_evon_leaf,
)
from blrax.utils import _project, _project_back


class TestEvonState(unittest.TestCase):
    def test_leaf_containers_and_predicate(self):
        m = MatrixEvonLeaf(
            L=jnp.eye(3), R=jnp.eye(2), QL=jnp.eye(3), QR=jnp.eye(2),
            H=jnp.ones((3, 2)), G_bar=jnp.zeros((3, 2)),
        )
        d = DiagEvonLeaf(H=jnp.ones(4), G_bar=jnp.zeros(4))
        self.assertIsNone(m.noise)
        self.assertIsNone(d.noise)
        self.assertTrue(_is_evon_leaf(m))
        self.assertTrue(_is_evon_leaf(d))
        self.assertFalse(_is_evon_leaf(jnp.ones(3)))

    def test_one_sided_leaf_allows_none_side(self):
        # right side diagonal: R and QR are None
        m = MatrixEvonLeaf(
            L=jnp.eye(3), R=None, QL=jnp.eye(3), QR=None,
            H=jnp.ones((3, 5)), G_bar=jnp.zeros((3, 5)),
        )
        self.assertIsNone(m.R)
        self.assertIsNone(m.QR)
        # treated as a single pytree leaf via the predicate
        leaves = jax.tree.leaves({'w': m}, is_leaf=_is_evon_leaf)
        self.assertEqual(len(leaves), 1)
        self.assertTrue(_is_evon_leaf(leaves[0]))

    def test_state_is_pytree_with_leaf_containers(self):
        leaves = {'w': MatrixEvonLeaf(jnp.eye(3), jnp.eye(2), jnp.eye(3), jnp.eye(2),
                                      jnp.ones((3, 2)), jnp.zeros((3, 2))),
                  'b': DiagEvonLeaf(jnp.ones(2), jnp.zeros(2))}
        state = ScaleByEvonState(count=jnp.zeros([], jnp.int32), ess=1.0,
                                 weight_decay=1e-4, precond_every=10, leaves=leaves)
        # flatten treating containers as leaves -> exactly the two containers
        cont = jax.tree.leaves(state.leaves, is_leaf=_is_evon_leaf)
        self.assertEqual(len(cont), 2)


class TestEvonProjection(unittest.TestCase):
    def _orthonormal(self, key, n):
        a = jr.normal(key, (n, n))
        q, _ = jnp.linalg.qr(a)
        return q

    def test_roundtrip_two_sided(self):
        kL, kR, kX = jr.split(jr.PRNGKey(0), 3)
        QL = self._orthonormal(kL, 4)
        QR = self._orthonormal(kR, 3)
        X = jr.normal(kX, (4, 3))
        back = _project_back(QL, QR, _project(QL, QR, X))
        self.assertTrue(jnp.allclose(back, X, atol=1e-5))

    def test_none_side_is_identity(self):
        QL = self._orthonormal(jr.PRNGKey(1), 4)
        X = jr.normal(jr.PRNGKey(2), (4, 3))
        # right side None -> only left rotation
        self.assertTrue(jnp.allclose(_project(QL, None, X), QL.T @ X, atol=1e-6))
        self.assertTrue(jnp.allclose(_project(None, None, X), X))

    def test_batched_projection(self):
        QL = self._orthonormal(jr.PRNGKey(3), 4)
        QR = self._orthonormal(jr.PRNGKey(4), 3)
        X = jr.normal(jr.PRNGKey(5), (7, 4, 3))  # batch of 7
        out = _project_back(QL, QR, X)
        self.assertEqual(out.shape, (7, 4, 3))
        # matches per-sample application
        self.assertTrue(jnp.allclose(out[0], QL @ X[0] @ QR.T, atol=1e-5))
