"""The 2-D toy energy: a soft-min of six quadratic bowls and a plateau,

    U(x) = -tau log [ sum_i exp(-(d_i + |x - c_i|^2 / (2 s_i^2)) / tau) + exp(-(H + a |x|^2) / tau) ],

shifted so the deepest point is 0. Two equally deep, narrow basins (A, B) far apart and four shallower, wider ones
(C-F) at distinct depths, separated by a plateau at U ~ H.
"""
import jax
import jax.numpy as jnp
import numpy as np

BASINS = [  # name, centre, depth (energy at the bottom above the global minimum), bowl width s
    ("A", (-4.5, 3.5), 0.0, 0.45),
    ("B", (4.5, -3.5), 0.0, 0.45),
    ("C", (4.0, 4.0), 0.3, 0.9),
    ("D", (-4.5, -3.5), 0.6, 0.9),
    ("E", (0.5, -0.5), 0.9, 0.9),
    ("F", (-0.5, 5.0), 1.2, 0.9),
]
NAMES = [b[0] for b in BASINS]
CENTRES = np.array([b[1] for b in BASINS], np.float32)
DEPTHS = np.array([b[2] for b in BASINS], np.float32)
WIDTHS = np.array([b[3] for b in BASINS], np.float32)
TAU = 0.1        # soft-min smoothness
PLATEAU = 2.5    # energy between the basins
OUTER = 0.015    # plateau rise: PLATEAU + OUTER |x|^2
DOMAIN = 8.0     # plots cover [-DOMAIN, DOMAIN]^2


def _raw(x):
    q = DEPTHS + jnp.sum((x - CENTRES) ** 2, -1) / (2 * WIDTHS ** 2)
    q = jnp.concatenate([q, (PLATEAU + OUTER * jnp.sum(x ** 2))[None]])
    return -TAU * jax.nn.logsumexp(-q / TAU)


E_SHIFT = float(min(_raw(jnp.asarray(c)) for c in CENTRES))


def energy(x):
    """U(x) for one point x (2,); the deepest point is 0."""
    return _raw(x) - E_SHIFT


def minima():
    """Each basin's minimum (Newton-polished from its centre), its energy and the Hessian determinant there."""
    grad, hess = jax.grad(energy), jax.hessian(energy)
    out = []
    for name, c, d, s in BASINS:
        x = jnp.asarray(c, jnp.float32)
        for _ in range(20):
            x = x - jnp.linalg.solve(hess(x), grad(x))
        out.append(dict(name=name, x=float(x[0]), y=float(x[1]), E=float(energy(x)), depth=d, width=s,
                        detH=float(np.linalg.det(np.asarray(hess(x), np.float64)))))
    return out


def classify(x, temperature, mins=None, sigmas=3.0):
    """Basin index of each point x (n, 2), or -1 (unresolved) if it is farther than `sigmas` standard deviations of the
    target's Gaussian around the nearest minimum (sqrt(T) times the bowl width)."""
    mins = minima() if mins is None else mins
    m = np.array([[b["x"], b["y"]] for b in mins])
    d2 = np.sum((np.asarray(x)[:, None] - m) ** 2, -1)
    b = np.argmin(d2, -1)
    r = sigmas * np.sqrt(temperature) * WIDTHS
    return np.where(np.min(d2, -1) < r[b] ** 2, b, -1)
