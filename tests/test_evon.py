import jax
import jax.numpy as jnp
import optax
import unittest
import inspect
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

    def test_m_dtype_controls_momentum_accumulator(self):
        # m_dtype sets the G_bar accumulator dtype at init and after an update;
        # H and the eigenbasis factors stay in the parameter dtype.
        params = {'w': jnp.ones((4, 3)), 'b': jnp.ones(3)}
        tx = scale_by_evon(ess=10., hess_init=1.0, max_precond_dim=100,
                           m_dtype=jnp.bfloat16)
        state = self._set_noise(tx.init(params))
        for leaf in jax.tree.leaves(state.leaves, is_leaf=_is_evon_leaf):
            self.assertEqual(leaf.G_bar.dtype, jnp.bfloat16)
            self.assertEqual(leaf.H.dtype, jnp.float32)   # stays param dtype
        grads = {'w': jnp.ones((4, 3)) * 0.2, 'b': jnp.ones(3) * 0.1}
        _, new_state = tx.update(grads, state, params)
        for leaf in jax.tree.leaves(new_state.leaves, is_leaf=_is_evon_leaf):
            self.assertEqual(leaf.G_bar.dtype, jnp.bfloat16)


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


class TestEvonIntegration(unittest.TestCase):
    def _train(self, tx, params, loss_fn, X, y, key, steps):
        state = tx.init(params)
        @jax.jit
        def step(params, state, key):
            key, k = jr.split(key)
            loss, grads, state = noisy_value_and_grad(loss_fn, state, params, k, X, y)
            updates, state = tx.update(grads, state, params)
            return optax.apply_updates(params, updates), state, key, loss
        losses = []
        for _ in range(steps):
            params, state, key, loss = step(params, state, key)
            losses.append(float(loss))
        return params, state, losses

    def test_end_to_end_linear_regression_converges(self):
        key = jr.PRNGKey(0)
        kx, kw, kp = jr.split(key, 3)
        X = jr.normal(kx, (64, 5))
        w_true = jr.normal(kw, (5, 1))
        y = X @ w_true
        params = {'w': jr.normal(kp, (5, 1)) * 0.1}
        def loss_fn(p, X, y, key):
            return jnp.mean((X @ p['w'] - y) ** 2)
        tx = evon(1e-1, ess=64., hess_init=1.0, max_precond_dim=100, precond_every=5)
        params, state, losses = self._train(tx, params, loss_fn, X, y, jr.PRNGKey(1), 300)
        self.assertLess(losses[-1], losses[0])
        self.assertTrue(losses[-1] < 0.5)
        self.assertTrue(all(map(lambda v: v == v, losses)))  # no NaNs

    def test_mixed_pytree_with_highrank_and_wide_leaf_jits(self):
        params = {
            'w': jnp.ones((6, 4)),
            'b': jnp.ones(4),
            'k': jnp.ones((2, 3, 4)),     # >2-D -> reshape (6,4)
            'wide': jnp.ones((3, 40)),    # right axis over cutoff -> one-sided
        }
        def loss_fn(p, key):
            return sum(jnp.sum(v ** 2) for v in jax.tree.leaves(p))
        tx = evon(1e-2, ess=10., hess_init=1.0, max_precond_dim=8, precond_every=3)
        params, state, losses = self._train(
            tx, params,
            lambda p, X, y, key: loss_fn(p, key),
            jnp.zeros(1), jnp.zeros(1), jr.PRNGKey(2), 20)
        self.assertTrue(all(v == v for v in losses))

    def test_precond_every_gates_basis_refresh(self):
        params = {'w': jnp.ones((4, 3))}
        def loss_fn(p, X, y, key):
            return jnp.sum((p['w'] - 0.5) ** 2)
        tx = evon(1e-2, ess=10., hess_init=1.0, max_precond_dim=100, precond_every=5)
        def one_step(params, state, c):
            inner = state[0]._replace(count=jnp.asarray(c, jnp.int32))
            state = (inner,) + state[1:]
            _, grads, state = noisy_value_and_grad(loss_fn, state, params,
                                                   jr.PRNGKey(c), jnp.zeros(1), jnp.zeros(1))
            _, state = tx.update(grads, state, params)
            return state
        # post-increment count in 1..4 -> no refresh, QL stays identity;
        # post-increment count 5 -> refresh (eigh). one_step sets count to c,
        # then update increments it to c+1 before the refresh gate is checked.
        for c in (0, 3):   # -> post-increment counts 1 and 4, both no-refresh
            s = one_step(params, tx.init(params), c)
            self.assertTrue(jnp.allclose(s[0].leaves['w'].QL, jnp.eye(4)))
        s = one_step(params, tx.init(params), 4)   # update increments to 5 -> refresh
        self.assertFalse(jnp.allclose(s[0].leaves['w'].QL, jnp.eye(4)))

    def test_structured_posterior_beats_diagonal_on_logistic(self):
        # Corollary 1 (qualitative): EVON's covariance is closer to the exact
        # full-Gaussian Laplace covariance than IVON's diagonal one. Sigma_evon
        # and Sigma_diag share the same V and differ only by the rotation QL, so
        # the gap below isolates the structural (off-diagonal) benefit. Features
        # are correlated (X = Z @ A) so the true covariance has strong
        # off-diagonals -- otherwise the rotation has almost nothing to capture.
        from jax import hessian
        key = jr.PRNGKey(0)
        n, dext = 80, 6
        Z = jr.normal(key, (n, dext))
        A = jr.normal(jr.PRNGKey(10), (dext, dext))   # mixing -> correlated columns
        X = Z @ A
        w_star = jr.normal(jr.PRNGKey(1), (dext,))
        probs = jax.nn.sigmoid(X @ w_star)
        y = (jr.uniform(jr.PRNGKey(2), (n,)) < probs).astype(jnp.float32)

        def nll(w):
            z = X @ w
            return jnp.sum(jax.nn.softplus(z) - y * z) + 0.5 * jnp.sum(w ** 2)

        # Refine the MAP point with a few Newton steps so the Laplace covariance
        # is evaluated at the true mode (fair comparison point).
        def newton_step(w):
            g = jax.grad(nll)(w)
            Hm = hessian(nll)(w)
            return w - jnp.linalg.solve(Hm, g)
        w_map = w_star
        for _ in range(25):
            w_map = newton_step(w_map)
        Sigma_full = jnp.linalg.inv(hessian(nll)(w_map))

        # EVON on the same problem; structured covariance = QL diag(V) QL^T (o=1)
        params = {'w': jnp.zeros((dext, 1))}
        def loss_fn(p, X, y, key):
            z = (X @ p['w'])[:, 0]
            return jnp.sum(jax.nn.softplus(z) - y * z)
        tx = evon(5e-2, ess=1.0, hess_init=1.0, weight_decay=1.0,
                  max_precond_dim=100, precond_every=5, one_sided=False)
        params, state, _ = self._train(tx, params, loss_fn, X, y, jr.PRNGKey(3), 800)
        leaf = state[0].leaves['w']
        V = 1.0 / (state[0].ess * (leaf.H + state[0].weight_decay))   # (dext, 1)
        Sigma_evon = leaf.QL @ (V[:, 0][:, None] * leaf.QL.T)         # (dext, dext)
        Sigma_diag = jnp.diag(V[:, 0])                                # IVON-style diagonal

        err_evon = jnp.linalg.norm(Sigma_evon - Sigma_full)
        err_diag = jnp.linalg.norm(Sigma_diag - Sigma_full)
        self.assertLess(float(err_evon), float(err_diag))


class TestEvonCoverage(unittest.TestCase):
    def test_sample_leaves_mask_reverts_to_mean(self):
        params = {'w': jnp.full((4, 3), 7.0)}
        tx = scale_by_evon(ess=10., hess_init=1.0, max_precond_dim=100)
        state = tx.init(params)
        mask = {'w': jnp.zeros((4, 3), bool).at[:2].set(True)}  # perturb only rows 0,1
        samples, _ = _evon_sample_leaves(jr.PRNGKey(0), params, state, mask=mask)
        # masked-out rows (2:) revert exactly to the mean
        self.assertTrue(jnp.allclose(samples['w'][2:], 7.0))
        # unmasked rows are perturbed away from the mean (w.h.p.)
        self.assertFalse(jnp.allclose(samples['w'][:2], 7.0))

    def test_get_scale_one_sided_matches_empirical(self):
        # right side None: covariance mixes only along the left axis
        QL, _ = jnp.linalg.qr(jr.normal(jr.PRNGKey(0), (3, 3)))
        H = jnp.array([[1.0, 4.0], [2.0, 3.0], [5.0, 1.0]])
        leaf = MatrixEvonLeaf(L=jnp.eye(3), R=None, QL=QL, QR=None,
                              H=H, G_bar=jnp.zeros((3, 2)))
        state = ScaleByEvonState(count=jnp.zeros([], jnp.int32), ess=2.0,
                                 weight_decay=0.5, precond_every=10, leaves={'w': leaf})
        params = {'w': jnp.zeros((3, 2))}
        samples, _ = _evon_sample_leaves(jr.PRNGKey(1), params, state, shape=(20000,))
        self.assertTrue(jnp.allclose(get_scale(state)['w'], samples['w'].std(0), atol=0.05))

    def test_get_scale_diag_leaf(self):
        H = jnp.array([1.0, 2.0, 3.0])
        leaf = DiagEvonLeaf(H=H, G_bar=jnp.zeros(3))
        state = ScaleByEvonState(count=jnp.zeros([], jnp.int32), ess=2.0,
                                 weight_decay=0.5, precond_every=10, leaves={'b': leaf})
        expected = jnp.sqrt(1.0 / (2.0 * (H + 0.5)))
        self.assertTrue(jnp.allclose(get_scale(state)['b'], expected, atol=1e-6))


class TestEvonHutchinsonConfig(unittest.TestCase):
    def test_hess_every_defaults_to_precond_every(self):
        opt = evon(1e-2, ess=1.0, hess_init=1.0, precond_every=7)
        state = opt.init({'w': jnp.zeros((3, 2))})[0]
        self.assertEqual(int(state.hess_every), 7)
        self.assertEqual(int(state.precond_every), 7)

    def test_explicit_hess_every_is_honored(self):
        opt = scale_by_evon(ess=1.0, hess_init=1.0, precond_every=10, hess_every=5)
        state = opt.init({'w': jnp.zeros((3, 2))})
        self.assertEqual(int(state.hess_every), 5)

    def test_b2_default_is_0_95_for_evon(self):
        self.assertEqual(inspect.signature(evon).parameters['b2'].default, 0.95)
        self.assertEqual(inspect.signature(scale_by_evon).parameters['b2'].default, 0.95)

    def test_leaves_accept_h_hat_transient(self):
        m = MatrixEvonLeaf(L=jnp.eye(3), R=jnp.eye(2), QL=jnp.eye(3), QR=jnp.eye(2),
                           H=jnp.ones((3, 2)), G_bar=jnp.zeros((3, 2)))
        d = DiagEvonLeaf(H=jnp.ones(4), G_bar=jnp.zeros(4))
        self.assertIsNone(m.h_hat)
        self.assertIsNone(d.h_hat)
        m2 = m._replace(h_hat=jnp.ones((3, 2)))
        self.assertIsNotNone(m2.h_hat)
