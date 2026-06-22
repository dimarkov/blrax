import jax
import jax.numpy as jnp
import unittest

from jax import random as jr
from blrax.optim import ivon
from blrax.utils import noisy_value_and_grad

class TestOptim(unittest.TestCase):
    def test_ivon(self):
        key = jr.PRNGKey(0)
        params = {'w': jnp.ones(10), 'b': jnp.ones(1)}
        loss_fn = lambda params, *args: params['w'].sum()
        
        # Test sequential and parallel variants
        for mc_samples in [1, 10]:
            with self.subTest(mc_samples=mc_samples):
                optimizer = ivon(learning_rate=1e-3)
                state = optimizer.init(params)
                mean_loss, updates, state = noisy_value_and_grad(loss_fn, state, params, key, mc_samples=mc_samples, method='sequential')
                
                # Test one step
                updates, _ = optimizer.update(updates, state, params)

                mean_loss, updates, state = noisy_value_and_grad(loss_fn, state, params, key, mc_samples=mc_samples, method='parallel')

                # Test one step
                updates, state = optimizer.update(updates, state, params)

    def test_ivon_with_aux(self):
        key = jr.PRNGKey(0)
        params = {'w': jnp.ones(10), 'b': jnp.ones(1)}
        def loss_fn(params, *args):
            return params['w'].sum(), 0.0
        
        # Test sequential and parallel variants
        for mc_samples in [1, 10]:
            with self.subTest(mc_samples=mc_samples):
                optimizer = ivon(learning_rate=1e-3)
                state = optimizer.init(params)
                out, updates, state = noisy_value_and_grad(loss_fn, state, params, key, mc_samples=mc_samples, method='sequential', has_aux=True)
                
                assert len(out) == 2
                # Test one step
                updates, _ = optimizer.update(updates, state, params)

                out, updates, state = noisy_value_and_grad(loss_fn, state, params, key, mc_samples=mc_samples, method='parallel', has_aux=True)
                assert len(out) == 2
                # Test one step
                updates, state = optimizer.update(updates, state, params)

    def test_hutchinson_exact_on_diagonal_quadratic(self):
        # L(theta) = 1/2 sum_i a_i theta_i^2  has Hessian diag(a).
        # A single Rademacher probe is exact: u (.) (diag(a) u) = a (.) u^2 = a.
        key = jr.PRNGKey(0)
        a = {'w': jnp.array([1., 2., 3., 4.]), 'b': jnp.array([5., 6.])}
        params = {'w': jnp.array([0.5, -1., 2., 0.]), 'b': jnp.array([1., -2.])}

        def loss_fn(params, *args):
            return 0.5 * (jnp.sum(a['w'] * params['w'] ** 2)
                          + jnp.sum(a['b'] * params['b'] ** 2))

        optimizer = ivon(learning_rate=1e-3)
        state = optimizer.init(params)
        value, grad, state = noisy_value_and_grad(
            loss_fn, state, params, key, estimator='hutchinson')

        # diagonal Hessian: exact for a single Rademacher probe
        h_bar = state[0].h_bar
        self.assertTrue(jnp.allclose(h_bar['w'], a['w']))
        self.assertTrue(jnp.allclose(h_bar['b'], a['b']))

        # value and gradient are evaluated at the mean mu (no sampling)
        self.assertTrue(jnp.allclose(value, loss_fn(params)))
        self.assertTrue(jnp.allclose(grad['w'], a['w'] * params['w']))
        self.assertTrue(jnp.allclose(grad['b'], a['b'] * params['b']))

    def test_hutchinson_estimate_feeds_hessian_ema_unscaled(self):
        # The Hutchinson estimate is the actual diagonal Hessian and must enter
        # IVON's Hessian EMA un-rescaled (no pi(h) factor). Non-trivial ess /
        # hess_init / wd and a low b2 make any spurious rescaling obvious.
        key = jr.PRNGKey(1)
        a = {'w': jnp.array([1., 2., 3., 4.]), 'b': jnp.array([5., 6.])}
        params = {'w': jnp.array([0.5, -1., 2., 0.]), 'b': jnp.array([1., -2.])}

        def loss_fn(params, *args):
            return 0.5 * (jnp.sum(a['w'] * params['w'] ** 2)
                          + jnp.sum(a['b'] * params['b'] ** 2))

        h0, wd, b2 = 2.0, 0.5, 0.5
        optimizer = ivon(learning_rate=1e-3, ess=10., hess_init=h0,
                         weight_decay=wd, b2=b2)
        state = optimizer.init(params)
        _, updates, state = noisy_value_and_grad(
            loss_fn, state, params, key, estimator='hutchinson')
        _, state = optimizer.update(updates, state, params)

        def expected_ema(t):  # IVON positivity-preserving EMA, t = diag(a)
            return b2 * h0 + (1 - b2) * t + 0.5 * ((1 - b2) * (h0 - t)) ** 2 / (h0 + wd)

        self.assertTrue(jnp.allclose(state[0].hess['w'], expected_ema(a['w'])))
        self.assertTrue(jnp.allclose(state[0].hess['b'], expected_ema(a['b'])))

    def test_hutchinson_with_float_aux(self):
        key = jr.PRNGKey(0)
        params = {'w': jnp.ones(10), 'b': jnp.ones(1)}
        def loss_fn(params, *args):
            return params['w'].sum(), 0.0  # float aux

        optimizer = ivon(learning_rate=1e-3)
        state = optimizer.init(params)
        out, updates, state = noisy_value_and_grad(
            loss_fn, state, params, key, estimator='hutchinson', has_aux=True)
        self.assertEqual(len(out), 2)
        updates, state = optimizer.update(updates, state, params)

    def test_hutchinson_with_integer_aux(self):
        # aux need not be a differentiable float; jvp over value_and_grad must
        # not choke on a non-float aux carried alongside the loss.
        key = jr.PRNGKey(0)
        params = {'w': jnp.ones(10), 'b': jnp.ones(1)}
        def loss_fn(params, *args):
            return params['w'].sum(), jnp.array(3, dtype=jnp.int32)  # int aux

        optimizer = ivon(learning_rate=1e-3)
        state = optimizer.init(params)
        out, updates, state = noisy_value_and_grad(
            loss_fn, state, params, key, estimator='hutchinson', has_aux=True)
        self.assertEqual(len(out), 2)
        self.assertEqual(int(out[1]), 3)

    def test_hutchinson_ignores_mc_samples_and_method(self):
        # mc_samples and method must not affect the hutchinson estimate.
        key = jr.PRNGKey(2)
        a = jnp.array([1., 2., 3., 4.])
        params = {'w': jnp.array([0.5, -1., 2., 0.])}
        def loss_fn(params, *args):
            return 0.5 * jnp.sum(a * params['w'] ** 2)

        optimizer = ivon(learning_rate=1e-3)
        state = optimizer.init(params)

        results = []
        for mc_samples, method in [(1, 'sequential'), (10, 'sequential'), (7, 'parallel')]:
            _, _, s = noisy_value_and_grad(
                loss_fn, state, params, key, estimator='hutchinson',
                mc_samples=mc_samples, method=method)
            results.append(s[0].h_bar['w'])

        for r in results:
            self.assertTrue(jnp.allclose(r, results[0]))
            self.assertTrue(jnp.allclose(r, a))  # exact, single probe


if __name__ == '__main__':
    unittest.main()
