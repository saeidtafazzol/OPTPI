"""Build index.html (the tutorial) from page/template.html and a toy training run (toy/train.py).

Inlines the toy landscape, the run's paths, weights and drift field for Fig. 4, and the numbers quoted in Section 3.2.
KaTeX and Plotly load from jsDelivr (pinned in the template).

usage: python page/build.py [--run runs/toy] [--out index.html]
"""
import argparse
import base64
import json
import os
import sys

import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from PI import MLPDrift  # noqa: E402
from toy import energy as toy  # noqa: E402


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def q16(a):
    """int16 quantization: (base64, scale) with a ~= int16 * scale."""
    a = np.asarray(a, np.float64)
    s = max(float(np.max(np.abs(a))), 1e-12) / 32000.0
    return b64(np.round(a / s).astype(np.int16)), s


def export_run(run, mins, frame_every, path_stride=2, quiver=21):
    """Fig. 4 data: every path at every shown iteration (every path_stride-th SDE step), the per-frame basin stats, and
    the network's drift on a grid at a few flow times. Also returns the per-iteration ESS and unresolved fraction."""
    meta = json.load(open(os.path.join(run, "meta.json")))
    paths = np.load(os.path.join(run, "paths.npy"), mmap_mode="r")       # (it, N+1, P, 2)
    log_w = np.load(os.path.join(run, "log_w.npy"), mmap_mode="r")       # (it, P)
    params = np.load(os.path.join(run, "params.npy"), mmap_mode="r")     # (n, D), params[i] drove iteration i * param_every
    it_n, N1, P, _ = paths.shape
    assert frame_every % meta["param_every"] == 0
    frames = list(range(0, it_n, frame_every))
    T = meta["temperature"]

    ess, unres_all = np.zeros(it_n), np.zeros(it_n)
    for i in range(it_n):
        w = np.exp(np.asarray(log_w[i], np.float64))
        ess[i] = 1.0 / np.sum(w ** 2)
        unres_all[i] = np.mean(toy.classify(paths[i][-1], T, mins) < 0)

    stats = []
    for f in frames:
        b, w = toy.classify(paths[f][-1], T, mins), np.exp(np.asarray(log_w[f], np.float64))
        ok = b >= 0
        stats.append(dict(it=f, ess=round(float(ess[f]), 1),
                          frac=(np.bincount(b[ok], minlength=len(mins)) / P).round(4).tolist(),
                          wfrac=np.bincount(b[ok], weights=w[ok], minlength=len(mins)).round(4).tolist(),
                          unres=round(float(np.mean(~ok)), 4), wunres=round(float(w[~ok].sum()), 4)))

    tidx = np.arange(0, N1, path_stride)
    pth = np.stack([paths[f][tidx] for f in frames])                       # (F, T', P, 2)

    net, scale = MLPDrift(2), jnp.float32(np.sqrt(meta["prior_var"]))
    _, unravel = jax.flatten_util.ravel_pytree(net.init(jax.random.PRNGKey(0), jnp.zeros(()), jnp.zeros((1, 2)), scale))
    qx = np.linspace(-(toy.DOMAIN - 0.5), toy.DOMAIN - 0.5, quiver).astype(np.float32)
    QX, QY = np.meshgrid(qx, qx)
    qpts = jnp.asarray(np.stack([QX.ravel(), QY.ravel()], -1))
    dts = np.array([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, .99], np.float32)

    @jax.jit
    def field(flat):
        p = unravel(flat)
        return jax.vmap(lambda t: net.apply(p, jnp.full((qpts.shape[0],), t), qpts, scale))(jnp.asarray(dts))

    drift = np.stack([np.asarray(field(jnp.asarray(params[f // meta["param_every"]]))) for f in frames])  # (F, nt, Q, 2)
    mag = np.linalg.norm(drift, axis=-1)
    u95 = [float(np.percentile(mag[1:, k], 95)) or 1.0 for k in range(len(dts))]   # arrow scale per flow time

    pb, ps = q16(pth)
    db, ds = q16(drift)
    r_conv = (3 * np.sqrt(T) * toy.WIDTHS).round(4).tolist()
    run_data = dict(frames=frames, n=P, Tn=len(tidx), tpath=(tidx / (N1 - 1)).tolist(), paths=pb, paths_s=ps,
                    stats=stats, qx=qx.tolist(), dts=dts.tolist(), drift=db, drift_s=ds, r_conv=r_conv)
    return run_data, u95, meta, ess, unres_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=os.path.join(HERE, "..", "runs", "toy"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "index.html"))
    ap.add_argument("--frame_every", type=int, default=20)
    a = ap.parse_args()

    mins = toy.minima()
    run, u95, meta, ess, unres = export_run(a.run, mins, a.frame_every)

    G, L = 201, toy.DOMAIN
    gx = np.linspace(-L, L, G).astype(np.float32)
    GX, GY = np.meshgrid(gx, gx)
    E = np.asarray(jax.vmap(toy.energy)(jnp.asarray(np.stack([GX.ravel(), GY.ravel()], -1)))).reshape(G, G)
    data = dict(L=L, G=G, E=b64(E.astype(np.float32)), basins=mins, T_model=meta["temperature"], n_paths=meta["paths"],
                u95=u95, run=run)

    # numbers quoted in Section 3.2, all from this run
    window = 500
    last = run["stats"][-1]
    above = np.where(unres >= 0.1)[0]
    pct = lambda v: f"{100 * v:.0f}%"
    nums = {
        "__SIGMA_MAX__": f"{meta['sigma_max']:.2f}",
        "__ESS_WINDOW__": str(window),
        "__ESS_LOW__": str(int(np.sum(ess[:window] <= 10))),
        "__RESOLVED_IT__": str(int(round((above[-1] + 1) / 10) * 10)) if len(above) else "0",   # unresolved < 10% from here on
        "__LAST_IT__": str(last["it"]),
        "__FRAC_A__": pct(last["frac"][0]), "__FRAC_B__": pct(last["frac"][1]), "__UNRES__": pct(last["unres"]),
        "__FRAME_EVERY__": str(a.frame_every),
    }

    html = open(os.path.join(HERE, "template.html")).read()
    img = "data:image/jpeg;base64," + base64.b64encode(open(os.path.join(HERE, "rake.jpg"), "rb").read()).decode()
    for ph, val in [("__RAKE_IMG__", img), ("/*DATA*/null", json.dumps(data).replace("</", "<\\/")), *nums.items()]:
        assert ph in html, ph
        html = html.replace(ph, val)
    open(a.out, "w").write(html)
    print(f"{a.out}: {len(html.encode()) / 1e6:.1f} MB", nums)


if __name__ == "__main__":
    main()
