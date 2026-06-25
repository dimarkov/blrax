# blrax
[Bayesian learning rule](https://arxiv.org/abs/2107.04562) based optimisation algorithms in Jax.

`blrax` provides [optax](https://optax.readthedocs.io)-style gradient transformations that
fit an approximate Gaussian posterior over the parameters while they train, so a single run
yields both a point estimate (the posterior mean) and a per-parameter uncertainty (the
posterior standard deviation).

## Installation
```bash
pip install .
```

## Implemented algorithms
* [Improved Variational Online Newton (IVON)](https://github.com/team-approx-bayes/ivon)
* [Eigenspace Variational Online Newton (EVON)](https://arxiv.org/html/2606.23357v1) — IVON run in
  SOAP's eigenbasis, giving a structured (non-diagonal) Gaussian posterior.

## Usage

IVON maintains a diagonal Gaussian posterior `N(μ, σ²)`. Each step needs an estimate of the
loss gradient **and** of the diagonal Hessian; `noisy_value_and_grad` produces both and stashes
the Hessian estimate in the optimizer state, then `optim.update` consumes it.

The loss function receives the parameters first and an RNG `key` as its **last** positional
argument (any other `*args` such as a data batch go in between):

```python
import jax, jax.numpy as jnp
import jax.random as jr
import optax
from blrax import ivon, noisy_value_and_grad, get_scale

def loss_fn(params, x, y, key):                 # key is the last positional argument
    logits = model(params, x)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

optim = ivon(learning_rate=1e-1, ess=N, hess_init=1.0, weight_decay=1e-3)  # N = #train points
opt_state = optim.init(params)

@jax.jit
def step(params, opt_state, key, x, y):
    key, k = jr.split(key)
    loss, grads, opt_state = noisy_value_and_grad(loss_fn, opt_state, params, k, x, y)
    updates, opt_state = optim.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss

# posterior standard deviation per parameter (e.g. for sampling predictions)
std = get_scale(opt_state[0])
```

`ess` (effective sample size) sets the scale of the posterior; the Bayesian-correct choice is
the number of training points `N`. Other knobs: `hess_init` (initial diagonal Hessian),
`weight_decay`, `clip_radius`, and the EMA decays `b1`/`b2`.

### Hessian estimators

`noisy_value_and_grad` supports two ways of estimating the diagonal Hessian, via the
`estimator` argument:

| `estimator`   | How the Hessian is estimated | Relevant knobs |
|---------------|------------------------------|----------------|
| `'sampling'` (default) | Reparameterization estimate: draw posterior samples `θ ~ N(μ, σ²)`, evaluate gradients at the samples, and form the diagonal Hessian from gradient–noise products. | `mc_samples`, `method` (`'sequential'` scan / `'parallel'` vmap) |
| `'hutchinson'` | [Sophia](https://arxiv.org/abs/2305.14342)-style estimate **at the mean**: the gradient `∇L(μ)` and the diagonal Hessian `u ⊙ (∇²L·u)` (one Rademacher probe `u`) are obtained from a single forward-over-reverse pass (`jvp` of `value_and_grad`). | — |

The `hutchinson` estimator is a variational-Laplace / delta-method variant: the gradient is
evaluated deterministically at the posterior mean and the Hessian uses a single probe, so it
ignores `mc_samples` and `method`. For a quadratic (diagonal Hessian) the single Rademacher
probe is exact.

```python
# at-the-mean Hutchinson estimate instead of posterior sampling
loss, grads, opt_state = noisy_value_and_grad(
    loss_fn, opt_state, params, k, x, y, estimator='hutchinson')
```

If your loss returns auxiliary data, pass `has_aux=True`; `mask` (a pytree matching `params`)
zeroes selected entries in both the gradient and the posterior noise / probe.

### Sparser Hessian estimates (`hess_every`)

The Hessian usually moves slowly, so it need not be re-estimated every step. The `hess_every`
knob (default `1`) refreshes the Hessian only every `k` steps and reuses the previous estimate
in between — the gradient is still computed on every step. For the `hutchinson` estimator this
skips the extra Hessian-vector-product pass on the intervening steps (the gradient at the mean
is computed alone), which is the bulk of its per-step overhead:

```python
# estimate the Hessian once every 10 steps
optim = ivon(learning_rate=1e-1, ess=N, hess_init=1.0, weight_decay=1e-3, hess_every=10)
```

(The `sampling` estimator derives gradient and Hessian from the same posterior samples, so
`hess_every` only freezes its Hessian EMA on the skipped steps without a compute saving.)

### EVON

EVON fits a non-diagonal Gaussian posterior `N(vec(M), (Q_R⊗Q_L) diag(vec(V)) (Q_R⊗Q_L)ᵀ)` per
2-D weight matrix — IVON run inside [SOAP](https://arxiv.org/abs/2409.11321)'s eigenbasis. The
training loop is identical to IVON; only the optimizer changes:

```python
from blrax import evon, noisy_value_and_grad, sample_posterior, get_scale

optim = evon(learning_rate=1e-1, ess=N, hess_init=1.0, weight_decay=1e-3,
             precond_every=10, max_precond_dim=10000, one_sided=False)
opt_state = optim.init(params)

@jax.jit
def step(params, opt_state, key, x, y):
    key, k = jr.split(key)
    loss, grads, opt_state = noisy_value_and_grad(loss_fn, opt_state, params, k, x, y)
    updates, opt_state = optim.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss

# test-time Bayesian model averaging: draw weight matrices from the structured posterior
models = sample_posterior(key, params, opt_state[0], shape=(num_mc,))
# per-parameter marginal std (drops correlations; use sampling for the full posterior)
std = get_scale(opt_state[0])
```

Knobs beyond IVON's: `b3` (preconditioner EMA decay), `precond_every` (eigenbasis refresh
period), `max_precond_dim` (axes larger than this are left diagonal), `one_sided` (precondition
only the smaller axis of each matrix). 1-D parameters (biases, norms) fall back to diagonal IVON
automatically.

## Examples
* [`examples/test_mnist.ipynb`](examples/test_mnist.ipynb) — IVON on MNIST.
* [`examples/compare_ivon_estimators.ipynb`](examples/compare_ivon_estimators.ipynb) — sampling vs. Hutchinson.
* [`examples/tune_ivon.py`](examples/tune_ivon.py) — Optuna hyper-parameter search (validation NLL).

### TODO
* [The Lie-Group Bayesian Learning Rule](https://github.com/team-approx-bayes/liegroups)
