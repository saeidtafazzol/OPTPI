"""Build the tutorial from page/template.html and a toy training run (toy/train.py).

Writes index.html (the template with the numbers quoted in Section 3.2 filled in) and, next to it, the data the page
fetches when it opens:
    data/toy.json     landscape and basin metadata, Fig. 4 frames and per-frame stats, and the list of binary arrays
    data/energy.f32   toy energy on a G x G grid (float32)
    data/paths.i16    every path at every shown iteration (int16, times the scale in toy.json)
    data/drift.i16    the network's drift on a grid at a few flow times (int16, times its scale)
All binaries are little-endian, row-major, with their shapes in toy.json. KaTeX and Plotly load from jsDelivr.

usage: python page/build.py [--run runs/toy] [--out .]
"""
import argparse
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


def q16(a):
    """int16 quantization: (array, scale) with a ~= int16 * scale."""
    a = np.asarray(a, np.float64)
    s = max(float(np.max(np.abs(a))), 1e-12) / 32000.0
    return np.round(a / s).astype("<i2"), s


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

    r_conv = (3 * np.sqrt(T) * toy.WIDTHS).round(4).tolist()
    run_data = dict(frames=frames, n=P, Tn=len(tidx), tpath=(tidx / (N1 - 1)).tolist(), stats=stats, qx=qx.tolist(),
                    dts=dts.tolist(), r_conv=r_conv)
    return run_data, dict(paths=q16(pth), drift=q16(drift)), u95, meta, ess, unres_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=os.path.join(HERE, "..", "runs", "toy"))
    ap.add_argument("--out", default=os.path.join(HERE, ".."), help="directory for index.html and data/")
    ap.add_argument("--frame_every", type=int, default=20)
    a = ap.parse_args()

    mins = toy.minima()
    run, q, u95, meta, ess, unres = export_run(a.run, mins, a.frame_every)

    G, L = 201, toy.DOMAIN
    gx = np.linspace(-L, L, G).astype(np.float32)
    GX, GY = np.meshgrid(gx, gx)
    E = np.asarray(jax.vmap(toy.energy)(jnp.asarray(np.stack([GX.ravel(), GY.ravel()], -1)))).reshape(G, G)
    arrays = {"energy": (E.astype("<f4"), None), **q}
    os.makedirs(os.path.join(a.out, "data"), exist_ok=True)
    index = {}
    for name, (arr, scale) in arrays.items():
        f = name + (".f32" if arr.dtype.kind == "f" else ".i16")
        arr.tofile(os.path.join(a.out, "data", f))
        index[name] = dict(file=f, dtype="float32" if arr.dtype.kind == "f" else "int16", shape=list(arr.shape),
                           **({} if scale is None else {"scale": scale}))
    data = dict(L=L, G=G, basins=mins, T_model=meta["temperature"], n_paths=meta["paths"], u95=u95, run=run, arrays=index)
    json.dump(data, open(os.path.join(a.out, "data", "toy.json"), "w"), separators=(",", ":"))

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
    for ph, val in nums.items():
        assert ph in html, ph
        html = html.replace(ph, val)
    open(os.path.join(a.out, "index.html"), "w").write(html)
    sizes = {f: os.path.getsize(os.path.join(a.out, "data", f)) / 1e6 for f in os.listdir(os.path.join(a.out, "data"))}
    print(f"index.html {len(html.encode()) / 1e3:.0f} kB, data/ " + ", ".join(f"{f} {v:.1f} MB" for f, v in sorted(sizes.items())), nums)


if __name__ == "__main__":
    main()
