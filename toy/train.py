"""Train PICE on the 2-D toy energy (toy/energy.py), target exp(-U / T).

Every iteration's paths and weights, and the network parameters every --param_every iterations, are written to --out
(used by page/build.py to make the training demo).

usage: python toy/train.py [--out runs/toy] [--iters 2000] [--seed 1]
"""
import argparse
import json
import os
import sys
import time

import jax
import jax.flatten_util
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from PI import PICE, ess  # noqa: E402
from toy import energy as toy  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/toy")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--paths", type=int, default=512)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--prior_std", type=float, default=3.5)
    ap.add_argument("--sigma_min", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--param_every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    pice = PICE(toy.energy, dim=2, temperature=a.temperature, steps=a.steps, paths=a.paths, prior_std=a.prior_std,
                sigma_min=a.sigma_min, lr=a.lr)
    key = jax.random.PRNGKey(a.seed)
    key, k0 = jax.random.split(key)
    params, opt_state = pice.init(k0)

    # storage: SDE states x(t_0..t_N) incl. the endpoint, normalized log weights, parameters
    n, N, P = a.iters, a.steps, a.paths
    mm = lambda name, shape: np.lib.format.open_memmap(os.path.join(a.out, name), "w+", np.float32, shape)
    S_paths, S_logw = mm("paths.npy", (n, N + 1, P, 2)), mm("log_w.npy", (n, P))
    flat = lambda p: np.asarray(jax.flatten_util.ravel_pytree(p)[0])
    S_par = mm("params.npy", ((n - 1) // a.param_every + 1, flat(params).size))
    json.dump(dict(vars(a), prior_var=pice.var, sigma_max=float(pice.sigma[0])), open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(f"sigma {float(pice.sigma[0]):.3f} -> {a.sigma_min}, prior std {np.sqrt(pice.var):.3f}, T {a.temperature}", flush=True)

    mins = toy.minima()
    t0 = time.time()
    for it in range(n):
        if it % a.param_every == 0:
            S_par[it // a.param_every] = flat(params)          # the parameters that drove iteration `it`
        key, k = jax.random.split(key)
        params, opt_state, out = pice.step(params, opt_state, k)
        S_paths[it, :N], S_paths[it, N], S_logw[it] = np.asarray(out["xs"]), np.asarray(out["x_T"]), np.asarray(out["log_w"])
        if it % 25 == 0 or it == n - 1:
            b = toy.classify(out["x_T"], a.temperature, mins)
            frac = " ".join(f"{name} {np.mean(b == i):.2f}" for i, name in enumerate(toy.NAMES))
            print(f"it {it:5d} ESS {ess(out['log_w']):6.1f}/{P} | {frac} unresolved {np.mean(b < 0):.2f} | "
                  f"loss {float(out['loss']):.3g} | {time.time() - t0:.0f}s", flush=True)
    for s in (S_paths, S_logw, S_par):
        s.flush()


if __name__ == "__main__":
    main()
