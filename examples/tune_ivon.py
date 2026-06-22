"""Optuna hyper-parameter search for IVON (sampling vs. hutchinson).

Optimises **validation NLL** over the four knobs that influence behaviour
(`lr`, `hess_init`, `weight_decay`, `clip_radius`) with a TPE sampler and
Hyperband pruning, separately for each estimator, then reports the best config
and its held-out test metrics.

Assumptions / fixed choices (see also the discussion in the README/notebook):
  * ess = N (number of train-fit points) -- the Bayesian-correct value; for the
    hutchinson estimator it does not affect the trajectory at all, only the
    reported posterior scale.
  * b1/b2 left at the library defaults; for `sampling` we fix mc_samples=1 and
    method='parallel' so both estimators are compared on the same four knobs.
  * Objective = best per-epoch val NLL evaluated at the posterior mean (mu).
  * A single fixed seed per study (same init + noise stream across trials) keeps
    the objective low-variance for the sampler.

Run (GPU 0):
  cd examples
  CUDA_VISIBLE_DEVICES=0 uv run --project .. --with "jax[cuda12]" --with optuna \
      python tune_ivon.py --n-trials 40 --max-epochs 20
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import argparse
import gzip
import json

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
from jax import lax, vmap, nn
from blrax import ivon, noisy_value_and_grad

try:
    import optuna
except ImportError:
    raise SystemExit("optuna is required: rerun with `uv run --with optuna ...`")


# --------------------------------------------------------------------------- data
def _standardize(train, test):
    m, s = train.reshape(-1, 1).mean(0), train.reshape(-1, 1).std(0)
    return (train - m) / s, (test - m) / s


def load_mnist(data_dir="data"):
    def imgs(fn):
        with gzip.open(fn, "rb") as f:
            return np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28 * 28)

    def lbls(fn):
        with gzip.open(fn, "rb") as f:
            return np.frombuffer(f.read(), np.uint8, offset=8)

    tr = imgs(f"{data_dir}/mnist_train_images.gz").astype(np.float32)
    te = imgs(f"{data_dir}/mnist_test_images.gz").astype(np.float32)
    tr, te = _standardize(tr, te)
    ytr = lbls(f"{data_dir}/mnist_train_labels.gz").astype(np.int32)
    yte = lbls(f"{data_dir}/mnist_test_labels.gz").astype(np.int32)
    return (jnp.asarray(tr), jnp.asarray(ytr)), (jnp.asarray(te), jnp.asarray(yte))


def compute_ece(logits, labels, num_bins=20):
    probs = nn.softmax(logits, axis=-1)
    conf = jnp.max(probs, axis=-1)
    correct = (jnp.argmax(probs, axis=-1) == labels)
    edges = jnp.linspace(0, 1, num_bins + 1)

    def bin_stat(lo, hi):
        in_bin = (conf > lo) & (conf <= hi)
        n = jnp.sum(in_bin)
        acc = jnp.where(n > 0, jnp.sum(correct * in_bin) / n, 0.0)
        c = jnp.where(n > 0, jnp.sum(conf * in_bin) / n, 0.0)
        return n * jnp.abs(acc - c)

    return jnp.stack([bin_stat(lo, hi) for lo, hi in zip(edges[:-1], edges[1:])]).sum() / len(labels)


# --------------------------------------------------------------------------- model
def make_model(seed, in_size=28 * 28, out_size=10, num_neurons=50, depth=3):
    return eqx.nn.MLP(in_size, out_size, num_neurons, depth, key=jr.PRNGKey(seed))


# --------------------------------------------------------------------------- training
def build_steps(static, optim, estimator, batch_size):
    def loss_fn(params, x, y, *args):
        model = eqx.combine(params, static)
        return optax.softmax_cross_entropy_with_integer_labels(vmap(model)(x), y).mean()

    est_kw = dict(estimator=estimator)
    if estimator == "sampling":
        est_kw.update(method="parallel", mc_samples=1)

    @eqx.filter_jit
    def run_epoch(params, opt_state, key, Xb, Yb):
        def body(carry, batch):
            params, opt_state, key = carry
            x, y = batch
            key, k = jr.split(key)
            _, grads, opt_state = noisy_value_and_grad(
                loss_fn, opt_state, params, k, x, y, **est_kw)
            updates, opt_state = optim.update(grads, opt_state, params)
            return (params := optax.apply_updates(params, updates), opt_state, key), None

        (params, opt_state, _), _ = lax.scan(body, (params, opt_state, key), (Xb, Yb))
        return params, opt_state

    @eqx.filter_jit
    def evaluate(params, x, y):
        logits = vmap(eqx.combine(params, static))(x)
        nll = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
        acc = jnp.mean(jnp.argmax(logits, 1) == y)
        ece = compute_ece(logits, y)
        return nll, acc, ece

    return run_epoch, evaluate


def train(params0, static, optim, estimator, data, seed, max_epochs, batch_size, report=None):
    (Xfit, Yfit), (Xval, Yval) = data
    run_epoch, evaluate = build_steps(static, optim, estimator, batch_size)

    params = params0
    opt_state = optim.init(params)
    n = Xfit.shape[0]
    nsteps = n // batch_size
    key = jr.PRNGKey(seed)

    best = float("inf")
    for epoch in range(max_epochs):
        key, sk, tk = jr.split(key, 3)
        perm = jr.permutation(sk, n)
        Xb = Xfit[perm][: nsteps * batch_size].reshape(nsteps, batch_size, -1)
        Yb = Yfit[perm][: nsteps * batch_size].reshape(nsteps, batch_size)
        params, opt_state = run_epoch(params, opt_state, tk, Xb, Yb)
        val_nll = float(evaluate(params, Xval, Yval)[0])
        if not np.isfinite(val_nll):
            val_nll = 1e6
        best = min(best, val_nll)
        if report is not None:
            report(epoch, val_nll)
    return best, params, evaluate


# --------------------------------------------------------------------------- search
def suggest(trial):
    lr = trial.suggest_float("lr", 1e-3, 1.0, log=True)
    hess_init = trial.suggest_float("hess_init", 1e-2, 10.0, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
    if trial.suggest_categorical("use_clip", [False, True]):
        clip_radius = trial.suggest_float("clip_radius", 0.1, 100.0, log=True)
    else:
        clip_radius = float("inf")
    return dict(lr=lr, hess_init=hess_init, weight_decay=weight_decay, clip_radius=clip_radius)


def make_objective(estimator, params0, static, data, ess, seed, max_epochs, batch_size):
    def objective(trial):
        hp = suggest(trial)
        optim = ivon(hp["lr"], ess=ess, hess_init=hp["hess_init"],
                     weight_decay=hp["weight_decay"], clip_radius=hp["clip_radius"])

        def report(epoch, val_nll):
            trial.report(val_nll, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        best, _, _ = train(params0, static, optim, estimator, data, seed, max_epochs, batch_size, report)
        return best

    return objective


def run_study(estimator, params0, static, data, ess, args):
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=3, max_resource=args.max_epochs, reduction_factor=3)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(
        make_objective(estimator, params0, static, data, ess, args.seed, args.max_epochs, args.batch_size),
        n_trials=args.n_trials, show_progress_bar=False)
    return study


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=40)
    ap.add_argument("--max-epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--val-size", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--estimators", default="sampling,hutchinson")
    ap.add_argument("--out", default="tune_ivon_results.json")
    args = ap.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print("jax devices:", jax.devices())

    (Xtr, Ytr), (Xte, Yte) = load_mnist()
    v = args.val_size
    Xfit, Yfit, Xval, Yval = Xtr[:-v], Ytr[:-v], Xtr[-v:], Ytr[-v:]
    data = ((Xfit, Yfit), (Xval, Yval))
    ess = float(Xfit.shape[0])
    print(f"train-fit={Xfit.shape[0]}  val={Xval.shape[0]}  test={Xte.shape[0]}  ess={ess:.0f}")

    model = make_model(args.seed)
    params0, static = eqx.partition(model, eqx.is_array)

    results = {}
    for estimator in [e.strip() for e in args.estimators.split(",") if e.strip()]:
        print(f"\n=== searching: {estimator} ({args.n_trials} trials, <= {args.max_epochs} epochs) ===")
        study = run_study(estimator, params0, static, data, ess, args)
        bp = study.best_params
        clip = bp.get("clip_radius", float("inf")) if bp.get("use_clip") else float("inf")
        best_hp = dict(lr=bp["lr"], hess_init=bp["hess_init"],
                       weight_decay=bp["weight_decay"], clip_radius=clip)

        # refit best config, report test metrics
        optim = ivon(best_hp["lr"], ess=ess, hess_init=best_hp["hess_init"],
                     weight_decay=best_hp["weight_decay"], clip_radius=best_hp["clip_radius"])
        _, params, evaluate = train(params0, static, optim, estimator, data, args.seed,
                                    args.max_epochs, args.batch_size)
        te_nll, te_acc, te_ece = (float(x) for x in evaluate(params, Xte, Yte))

        n_pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
        results[estimator] = dict(best_val_nll=study.best_value, best_hp=best_hp,
                                  test_nll=te_nll, test_acc=te_acc, test_ece=te_ece,
                                  n_trials=len(study.trials), n_pruned=n_pruned)
        print(f"  best val NLL = {study.best_value:.4f}  (pruned {n_pruned}/{len(study.trials)})")
        print(f"  best hp      = lr={best_hp['lr']:.3g}  hess_init={best_hp['hess_init']:.3g}  "
              f"wd={best_hp['weight_decay']:.3g}  clip={best_hp['clip_radius']}")
        print(f"  test         = acc={te_acc:.4f}  nll={te_nll:.4f}  ece={te_ece:.4f}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")

    if len(results) > 1:
        print("\n--- comparison ---")
        hdr = f"{'estimator':12s} {'val_nll':>8s} {'test_acc':>9s} {'test_nll':>9s} {'test_ece':>9s}  best_hp"
        print(hdr)
        for est, r in results.items():
            hp = r["best_hp"]
            print(f"{est:12s} {r['best_val_nll']:8.4f} {r['test_acc']:9.4f} "
                  f"{r['test_nll']:9.4f} {r['test_ece']:9.4f}  "
                  f"lr={hp['lr']:.3g} h0={hp['hess_init']:.3g} wd={hp['weight_decay']:.3g} clip={hp['clip_radius']}")


if __name__ == "__main__":
    main()
