import jax
import jax.numpy as jnp
import optax
import unittest
from jax import random as jr

from blrax.states import (
    MatrixEvonLeaf, DiagEvonLeaf, ScaleByEvonState, _is_evon_leaf,
)
from blrax.utils import _project, _project_back, precision
from blrax.optim import scale_by_evon, _update_diag_leaf, _update_matrix_leaf, update_hessian, evon
from blrax.utils import noisy_value_and_grad, _evon_sample_leaves
from blrax.utils import sample_posterior, get_scale


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
        # float32 round-trip through two orthonormal rotations
        self.assertTrue(jnp.allclose(back, X, atol=1e-4))

    def test_none_side_is_identity(self):
        QL = self._orthonormal(jr.PRNGKey(1), 4)
        X = jr.normal(jr.PRNGKey(2), (4, 3))
        # right side None -> only left rotation
        self.assertTrue(jnp.allclose(_project(QL, None, X), QL.T @ X, atol=1e-6))
        # left side None -> only right rotation
        QR = self._orthonormal(jr.PRNGKey(3), 3)
        self.assertTrue(jnp.allclose(_project(None, QR, X), X @ QR, atol=1e-6))
        self.assertTrue(jnp.allclose(_project(None, None, X), X))

    def test_batched_projection(self):
        QL = self._orthonormal(jr.PRNGKey(3), 4)
        QR = self._orthonormal(jr.PRNGKey(4), 3)
        X = jr.normal(jr.PRNGKey(5), (7, 4, 3))  # batch of 7
        out = _project_back(QL, QR, X)
        self.assertEqual(out.shape, (7, 4, 3))
        # matches per-sample application; batched matmul takes a different
        # float32 reduction path than a single matmul, hence the loose atol
        self.assertTrue(jnp.allclose(out[0], QL @ X[0] @ QR.T, atol=1e-3))


class TestEvonInit(unittest.TestCase):
    def test_init_leaf_kinds_two_sided(self):
        params = {
            'w':  jnp.ones((4, 3)),       # 2-D, both small -> two-sided matrix
            'b':  jnp.ones(3),            # 1-D -> diag
            's':  jnp.ones(()),           # scalar -> diag
            'wide': jnp.ones((2, 50)),    # right axis over cutoff -> one-sided (left only)
        }
        tx = scale_by_evon(ess=10., hess_init=1.0, max_precond_dim=8, one_sided=False)
        state = tx.init(params)
        self.assertIsInstance(state, ScaleByEvonState)
        L = state.leaves
        self.assertIsInstance(L['w'], MatrixEvonLeaf)
        self.assertEqual(L['w'].QL.shape, (4, 4))
        self.assertEqual(L['w'].QR.shape, (3, 3))
        self.assertEqual(L['w'].H.shape, (4, 3))
        self.assertIsInstance(L['b'], DiagEvonLeaf)
        self.assertIsInstance(L['s'], DiagEvonLeaf)
        # 'wide': left axis (2) <= 8 kept, right axis (50) > 8 -> None
        self.assertIsInstance(L['wide'], MatrixEvonLeaf)
        self.assertIsNotNone(L['wide'].QL)
        self.assertIsNone(L['wide'].QR)
        self.assertIsNone(L['wide'].R)
        self.assertEqual(L['wide'].H.shape, (2, 50))

    def test_init_one_sided_keeps_smaller_axis(self):
        params = {'w': jnp.ones((8, 3))}  # both <= cutoff
        tx = scale_by_evon(ess=10., hess_init=1.0, max_precond_dim=100, one_sided=True)
        leaf = tx.init(params).leaves['w']
        # smaller axis is the right one (3) -> keep QR, drop QL
        self.assertIsNone(leaf.QL)
        self.assertIsNotNone(leaf.QR)
        self.assertEqual(leaf.QR.shape, (3, 3))

    def test_init_one_sided_square_keeps_left_axis(self):
        # square matrix: both axes equal size, tie kept on the left side
        params = {'w': jnp.ones((4, 4))}
        tx = scale_by_evon(ess=10., hess_init=1.0, max_precond_dim=100, one_sided=True)
        leaf = tx.init(params).leaves['w']
        self.assertIsNotNone(leaf.QL)
        self.assertEqual(leaf.QL.shape, (4, 4))
        self.assertIsNone(leaf.QR)
        self.assertIsNone(leaf.R)

    def test_init_reshapes_higher_rank(self):
        params = {'k': jnp.ones((2, 3, 5))}  # -> (6, 5)
        tx = scale_by_evon(ess=10., hess_init=1.0, max_precond_dim=100)
        leaf = tx.init(params).leaves['k']
        self.assertIsInstance(leaf, MatrixEvonLeaf)
        self.assertEqual(leaf.H.shape, (6, 5))
        self.assertEqual(leaf.QL.shape, (6, 6))

    def test_init_identity_bases(self):
        params = {'w': jnp.ones((4, 3))}
        leaf = scale_by_evon(ess=1., hess_init=1.0).init(params).leaves['w']
        self.assertTrue(jnp.allclose(leaf.QL, jnp.eye(4)))
        self.assertTrue(jnp.allclose(leaf.L, jnp.zeros((4, 4))))


def _ref_diag_update(g, p, H, G_bar, noise, ess, wd, b1, b2):
    """Closed-form diagonal IVON-without-debias reference (quadratic correction kept)."""
    Hhat = noise * g * precision(H, ess, wd)
    G_bar2 = b1 * G_bar + (1 - b1) * g
    H2 = update_hessian(H, Hhat, b2, wd)
    U = (G_bar2 + wd * p) / (H2 + wd)
    return U, H2, G_bar2


class TestEvonLeafUpdate(unittest.TestCase):
    def test_diag_leaf_matches_reference(self):
        ess, wd, b1, b2 = 10., 0.5, 0.9, 0.5
        p = jnp.array([0.5, -1.0, 2.0])
        g = jnp.array([0.3, 0.7, -0.2])
        leaf = DiagEvonLeaf(H=jnp.array([1.0, 2.0, 3.0]),
                            G_bar=jnp.array([0.1, -0.1, 0.0]),
                            noise=jnp.array([0.4, -0.3, 0.2]))
        delta, new = _update_diag_leaf(g, p, leaf, ess, wd, b1, b2)
        U, H2, G2 = _ref_diag_update(g, p, leaf.H, leaf.G_bar, leaf.noise, ess, wd, b1, b2)
        self.assertTrue(jnp.allclose(delta, U))
        self.assertTrue(jnp.allclose(new.H, H2))
        self.assertTrue(jnp.allclose(new.G_bar, G2))
        self.assertIsNone(new.noise)

    def test_matrix_leaf_with_identity_Q_reduces_to_diagonal(self):
        # QL=QR=I and no refresh (precond_every huge) -> matrix path == diagonal path
        ess, wd, b1, b2, b3 = 10., 0.5, 0.9, 0.5, 0.95
        p = jnp.array([[0.5, -1.0], [2.0, 0.3]])
        g = jnp.array([[0.3, 0.7], [-0.2, 0.1]])
        H = jnp.array([[1.0, 2.0], [3.0, 1.5]])
        Gb = jnp.zeros((2, 2))
        E = jnp.array([[0.4, -0.3], [0.2, 0.5]])
        leaf = MatrixEvonLeaf(L=jnp.eye(2), R=jnp.eye(2), QL=jnp.eye(2), QR=jnp.eye(2),
                              H=H, G_bar=Gb, noise=E)
        delta, new = _update_matrix_leaf(g, p, leaf, ess, wd, b1, b2, b3,
                                         count=jnp.asarray(1, jnp.int32),
                                         precond_every=jnp.asarray(10**9, jnp.int32))
        # reference: elementwise diagonal IVON-no-debias (since G°=G, Δ=U)
        U, H2, G2 = _ref_diag_update(g, p, H, Gb, E, ess, wd, b1, b2)
        self.assertTrue(jnp.allclose(delta, U, atol=1e-6))
        self.assertTrue(jnp.allclose(new.H, H2, atol=1e-6))
        self.assertTrue(jnp.allclose(new.G_bar, G2, atol=1e-6))
        self.assertIsNone(new.noise)

    def test_refresh_rotates_momentum_preserving_represented_object(self):
        # Non-identity OLD basis so the rotation (QL_new^T @ QL_old) is genuinely
        # exercised: a missing or transposed rotation would change Q_L G_bar.
        QL_old, _ = jnp.linalg.qr(jr.normal(jr.PRNGKey(5), (4, 4)))
        A = jr.normal(jr.PRNGKey(0), (4, 4))
        L = A @ A.T + jnp.eye(4)         # SPD -> non-trivial refreshed basis
        leaf = MatrixEvonLeaf(L=L, R=None, QL=QL_old, QR=None,
                              H=jnp.ones((4, 3)), G_bar=jr.normal(jr.PRNGKey(1), (4, 3)),
                              noise=jnp.zeros((4, 3)))
        g = jr.normal(jr.PRNGKey(2), (4, 3))
        p = jnp.zeros((4, 3))
        old_repr = leaf.QL @ leaf.G_bar          # left-only: Q_L Ḡ (right identity)
        # count == precond_every triggers the first (eigh) refresh.
        # b1=1.0: no EMA mixing, so the only transform on G_bar is the rotation.
        _, new = _update_matrix_leaf(g, p, leaf, 10., 0.5, 1.0, 0.5, 0.95,
                                     count=jnp.asarray(10, jnp.int32),
                                     precond_every=jnp.asarray(10, jnp.int32))
        # refreshed basis is orthonormal and actually changed
        self.assertTrue(jnp.allclose(new.QL.T @ new.QL, jnp.eye(4), atol=1e-5))
        self.assertFalse(jnp.allclose(new.QL, QL_old, atol=1e-3))
        # the rotation preserves the represented object Q_L Ḡ exactly: this fails
        # for a missing rotation (new_repr = QL_new @ G_bar) or a wrong one.
        new_repr = new.QL @ new.G_bar
        self.assertTrue(jnp.allclose(new_repr, old_repr, atol=1e-4))

    def test_no_refresh_leaves_basis_unchanged(self):
        leaf = MatrixEvonLeaf(L=jnp.eye(3) * 2, R=None, QL=jnp.eye(3), QR=None,
                              H=jnp.ones((3, 2)), G_bar=jnp.zeros((3, 2)),
                              noise=jnp.zeros((3, 2)))
        g = jnp.ones((3, 2))
        p = jnp.zeros((3, 2))
        # count not a multiple of precond_every -> no refresh
        _, new = _update_matrix_leaf(g, p, leaf, 10., 0.5, 0.9, 0.5, 0.95,
                                     count=jnp.asarray(3, jnp.int32),
                                     precond_every=jnp.asarray(10, jnp.int32))
        self.assertTrue(jnp.allclose(new.QL, jnp.eye(3)))


class TestEvonUpdate(unittest.TestCase):
    def _set_noise(self, state, value=0.1):
        # fill transient noise on every leaf so update can run deterministically
        def fill(leaf):
            return leaf._replace(noise=jnp.full(leaf.H.shape, value, leaf.H.dtype))
        leaves = jax.tree.map(fill, state.leaves, is_leaf=_is_evon_leaf)
        return state._replace(leaves=leaves)

    def test_update_runs_and_increments_count(self):
        params = {'w': jnp.ones((4, 3)), 'b': jnp.ones(3)}
        tx = scale_by_evon(ess=10., hess_init=1.0, max_precond_dim=100)
        state = self._set_noise(tx.init(params))
        grads = {'w': jnp.ones((4, 3)) * 0.2, 'b': jnp.ones(3) * 0.1}
        updates, new_state = tx.update(grads, state, params)
        self.assertEqual(updates['w'].shape, (4, 3))
        self.assertEqual(int(new_state.count), 1)
        self.assertIsNone(jax.tree.leaves(new_state.leaves, is_leaf=_is_evon_leaf)[0].noise)
        self.assertTrue(jnp.all(jnp.isfinite(updates['w'])))

    def test_evon_wrapper_one_step(self):
        params = {'w': jnp.ones((4, 3)), 'b': jnp.ones(3)}
        tx = evon(learning_rate=1e-2, ess=10., hess_init=1.0, max_precond_dim=100)
        state = tx.init(params)
        # fill noise on the inner scale_by_evon state (index 0 of the chain)
        inner = state[0]
        def fill(leaf):
            return leaf._replace(noise=jnp.full(leaf.H.shape, 0.1, leaf.H.dtype))
        state = (inner._replace(leaves=jax.tree.map(fill, inner.leaves, is_leaf=_is_evon_leaf)),) + state[1:]
        grads = {'w': jnp.ones((4, 3)) * 0.2, 'b': jnp.ones(3) * 0.1}
        updates, _ = tx.update(grads, state, params)
        new_params = optax.apply_updates(params, updates)
        self.assertTrue(jnp.all(jnp.isfinite(new_params['w'])))

    def test_diagonal_evon_matches_reference_trajectory(self):
        # all-diagonal (max_precond_dim=0): EVON == diagonal IVON-no-debias, step by step.
        ess, wd, b1, b2 = 10., 0.5, 0.9, 0.5
        params = {'w': jnp.array([0.5, -1.0, 2.0, 0.3])}
        grads = {'w': jnp.array([0.3, 0.7, -0.2, 0.1])}
        tx = scale_by_evon(ess=ess, hess_init=1.0, b1=b1, b2=b2, weight_decay=wd,
                           max_precond_dim=0)
        state = self._set_noise(tx.init(params), value=0.4)
        noise = jnp.full(4, 0.4)
        updates, _ = tx.update(grads, state, params)
        U, _, _ = _ref_diag_update(grads['w'], params['w'], jnp.ones(4),
                                   jnp.zeros(4), noise, ess, wd, b1, b2)
        self.assertTrue(jnp.allclose(updates['w'], U))


class TestEvonSampler(unittest.TestCase):
    def test_sample_leaves_shapes_and_noise(self):
        params = {'w': jnp.zeros((4, 3)), 'b': jnp.zeros(3)}
        tx = scale_by_evon(ess=10., hess_init=1.0, max_precond_dim=100)
        state = tx.init(params)
        samples, noises = _evon_sample_leaves(jr.PRNGKey(0), params, state)
        self.assertEqual(samples['w'].shape, (4, 3))
        self.assertEqual(len(noises), 2)
        self.assertEqual(noises[1].shape, (4, 3))   # E for 'w' in the eigenbasis (JAX sorts dict keys: b < w)

    def test_sample_leaves_identity_basis_is_diagonal_draw(self):
        # at init QL=QR=I, so the matrix draw equals M + E elementwise
        params = {'w': jnp.full((4, 3), 5.0)}
        tx = scale_by_evon(ess=10., hess_init=1.0, max_precond_dim=100)
        state = tx.init(params)
        samples, noises = _evon_sample_leaves(jr.PRNGKey(1), params, state)
        self.assertTrue(jnp.allclose(samples['w'], 5.0 + noises[0], atol=1e-6))

    def test_noisy_value_and_grad_dispatches_to_evon(self):
        params = {'w': jnp.ones((4, 3)), 'b': jnp.ones(3)}
        tx = scale_by_evon(ess=10., hess_init=1.0, max_precond_dim=100)
        state = (tx.init(params),)   # tuple state like optax.chain
        def loss_fn(p, *args):
            return p['w'].sum() + p['b'].sum()
        out, grads, new_state = noisy_value_and_grad(loss_fn, state, params, jr.PRNGKey(2))
        self.assertEqual(grads['w'].shape, (4, 3))
        # noise was stashed
        stashed = jax.tree.leaves(new_state[0].leaves, is_leaf=_is_evon_leaf)[0].noise
        self.assertIsNotNone(stashed)

    def test_posterior_sample_mean_is_M(self):
        # many draws around M -> empirical mean close to M
        params = {'w': jnp.full((3, 2), 1.0)}
        tx = scale_by_evon(ess=2., hess_init=1.0, max_precond_dim=100)
        state = tx.init(params)
        samples, _ = _evon_sample_leaves(jr.PRNGKey(3), params, state, shape=(4000,))
        self.assertEqual(samples['w'].shape, (4000, 3, 2))
        self.assertTrue(jnp.allclose(samples['w'].mean(0), 1.0, atol=0.05))


class TestEvonPosterior(unittest.TestCase):
    def test_sample_posterior_dispatches(self):
        params = {'w': jnp.zeros((3, 2))}
        tx = scale_by_evon(ess=5., hess_init=1.0, max_precond_dim=100)
        state = tx.init(params)
        s = sample_posterior(jr.PRNGKey(0), params, state, shape=(10,))
        self.assertEqual(s['w'].shape, (10, 3, 2))

    def test_get_scale_matches_empirical_marginal(self):
        # rotate to a non-trivial basis, then compare get_scale to empirical std
        key = jr.PRNGKey(1)
        QL, _ = jnp.linalg.qr(jr.normal(key, (3, 3)))
        QR, _ = jnp.linalg.qr(jr.normal(jr.PRNGKey(2), (2, 2)))
        H = jnp.array([[1.0, 4.0], [2.0, 3.0], [5.0, 1.0]])
        leaf = MatrixEvonLeaf(L=jnp.eye(3), R=jnp.eye(2), QL=QL, QR=QR,
                              H=H, G_bar=jnp.zeros((3, 2)))
        state = ScaleByEvonState(count=jnp.zeros([], jnp.int32), ess=2.0,
                                 weight_decay=0.5, precond_every=10,
                                 leaves={'w': leaf})
        params = {'w': jnp.zeros((3, 2))}
        scale = get_scale(state)
        samples, _ = _evon_sample_leaves(jr.PRNGKey(3), params, state, shape=(20000,))
        emp = samples['w'].std(0)
        self.assertTrue(jnp.allclose(scale['w'], emp, atol=0.05))
