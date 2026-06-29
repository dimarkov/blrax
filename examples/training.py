import equinox as eqx
import optax
from blrax import noisy_value_and_grad
from jax import lax, vmap, nn
import jax.random as jr
import jax.numpy as jnp

def compute_ece(logits, labels, num_bins=20):
    """Compute Expected Calibration Error"""
    probs = nn.softmax(logits, axis=-1)
    confidences = jnp.max(probs, axis=-1)
    predictions = jnp.argmax(probs, axis=-1)
    correct = predictions == labels

    bin_boundaries = jnp.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    def compute_bin_stats(bin_lower, bin_upper):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = jnp.sum(in_bin)
        accuracy_in_bin = jnp.where(prop_in_bin > 0, jnp.sum(correct * in_bin) / prop_in_bin, 0.)
        avg_confidence_in_bin = jnp.where(prop_in_bin > 0, jnp.sum(confidences * in_bin) / prop_in_bin, 0.)
        return prop_in_bin * jnp.abs(accuracy_in_bin - avg_confidence_in_bin) / len(in_bin)

    ce = [compute_bin_stats(bl, bu) for bl, bu in zip(bin_lowers, bin_uppers)]
    return jnp.stack(ce).mean()

def run_training(key, nnet, optim, train_ds, test_ds, mc_samples=1, num_epochs=10,
                 batch_size=128, method='sequential', estimator='sampling'):
    params, static = eqx.partition(nnet, eqx.is_array)
    opt_state = optim.init(params)
    n_samples = len(train_ds['image'])
    steps_per_epoch = n_samples // batch_size

    def loss_fn(params, x, y, *args):
        model = eqx.combine(params, static)
        logits = vmap(model)(x)
        return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

    @eqx.filter_jit
    def train_step(key, loss_fn, params, opt_state, x, y):
        loss_value, grads, opt_state = noisy_value_and_grad(
            loss_fn, opt_state, params, key, x, y,
            mc_samples=mc_samples, method=method, estimator=estimator)
        updates, opt_state = optim.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    @eqx.filter_jit
    def evaluate(params, images, labels):
        model = eqx.combine(params, static)
        logits = vmap(model)(images)
        predictions = jnp.argmax(logits, axis=1)
        acc = jnp.mean(predictions == labels)
        nll = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        ece = compute_ece(logits, labels)
        return acc, nll, ece

    def train_epoch(carry, key):
        params, opt_state = carry
        key, _key = jr.split(key)
        perm = jr.permutation(_key, n_samples)
        train_images = train_ds['image'][perm]
        train_labels = train_ds['label'][perm]

        def train_step_scan(carry, xs):
            params, opt_state, key = carry
            batch_images, batch_labels = xs
            key, _key = jr.split(key)
            params, opt_state, loss_value = train_step(
                _key, loss_fn, params, opt_state, batch_images, batch_labels)
            return (params, opt_state, key), loss_value

        data = (
            train_images[:steps_per_epoch * batch_size].reshape(steps_per_epoch, batch_size, -1),
            train_labels[:steps_per_epoch * batch_size].reshape(steps_per_epoch, batch_size))
        init_carry = (params, opt_state, key)
        (params, opt_state, _), losses = lax.scan(train_step_scan, init_carry, data)

        acc, nll, ece = evaluate(params, test_ds['image'].reshape(-1, 28*28), test_ds['label'])
        metrics = {'loss': losses.sum() / steps_per_epoch, 'acc': acc, 'ece': ece, 'nll': nll}
        return (params, opt_state), metrics

    keys = jr.split(key, num_epochs)
    init_carry = (params, opt_state)
    (params, final_opt_state), metrics = lax.scan(train_epoch, init_carry, keys)
    trained_model = eqx.combine(params, static)
    return trained_model, final_opt_state, metrics