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
                mean_loss, updates = noisy_value_and_grad(loss_fn, state[0], params, key, mc_samples=mc_samples, method='sequential')
                
                # Test one step
                updates, _ = optimizer.update(updates, state, params)

                mean_loss, updates = noisy_value_and_grad(loss_fn, state[0], params, key, mc_samples=mc_samples, method='parallel')

                # Test one step
                updates, state = optimizer.update(updates, state, params)

if __name__ == '__main__':
    unittest.main()
