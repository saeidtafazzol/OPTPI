"""Path Integral Cross-Entropy (PICE) sampler.

Learns a drift u_theta(t, x) so that the SDE

    dX_t = u_theta(t, X_t) dt + sigma(t) dW_t,   X_0 = x0,   t in [0, 1]

ends at samples of the Boltzmann distribution mu(x) ∝ exp(-U(x) / T) of any energy U (Schrödinger–Föllmer:
terminal cost Psi = log(dmu0 / dmu), mu0 = N(x0, sigma_bar^2 I) the endpoint law of the uncontrolled SDE).

Each training step:
  1. roll out N paths under the current drift (Euler–Maruyama on a uniform grid),
  2. weight them by w_i ∝ exp(-S_i),  S_i = Girsanov term + Psi(X_T^(i)),  self-normalized,
  3. take one Adam step on the weighted regression of the drift onto the path increments
         0.5 sum_i w_i sum_k (dt / sigma_k^2) |dX_k^(i) / dt - u_theta(t_k, X_k^(i))|^2,
     whose negative gradient is the importance-sampled forward-KL gradient sum_i w_i F_i.

Usage, for any energy U: R^d -> R written in JAX:

    pice = PICE(U, dim=d, temperature=T)
    params, opt_state = pice.init(jax.random.PRNGKey(0))
    for it in range(iters):
        key, k = jax.random.split(key)
        params, opt_state, out = pice.step(params, opt_state, k)
    x = pice.sample(params, key)          # (paths, d) samples of exp(-U / T)
"""
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax


class MLPDrift(nn.Module):
    """u_theta(t, x): an MLP on the position (scaled by the prior std) and a few Fourier features of t.
    The last layer starts at zero, so training starts from the uncontrolled SDE (u = 0)."""
    dim: int
    hidden: int = 128
    layers: int = 3

    @nn.compact
    def __call__(self, t, x, scale):
        tf = jnp.stack([t, jnp.sin(np.pi * t), jnp.cos(np.pi * t)], -1)
        h = jnp.concatenate([x / scale, jnp.broadcast_to(tf, x.shape[:-1] + (3,))], -1)
        for _ in range(self.layers):
            h = nn.silu(nn.Dense(self.hidden)(h))
        return nn.Dense(self.dim, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros)(h)


def linear_schedule(steps, prior_std, sigma_min):
    """sigma(t) linear from sigma_max to sigma_min at t = 1 on a uniform grid of `steps` steps, with sigma_max set so the
    uncontrolled endpoint has standard deviation prior_std per dimension (sum_k sigma_k^2 dt = prior_std^2).
    Returns sigma (steps,) and dt."""
    dt = 1.0 / steps
    t = np.arange(steps) * dt
    lo, hi = sigma_min, 100.0
    for _ in range(100):   # bisection on sigma_max
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if np.sum((mid + (sigma_min - mid) * t) ** 2) * dt < prior_std ** 2 else (lo, mid)
    return (hi + (sigma_min - hi) * t).astype(np.float32), dt


class PICE:
    """PICE sampler for exp(-energy(x) / temperature), x in R^dim.

    energy:       U(x) for a single point x (dim,); vectorized internally with jax.vmap.
    temperature:  T of the target exp(-U / T).
    x0:           start of every path (default: the origin).
    steps, paths: Euler–Maruyama steps per path and paths per iteration.
    prior_std:    standard deviation of the uncontrolled endpoint around x0 (sigma_bar).
    sigma_min:    sigma(1); sigma(t) is linear (see linear_schedule).
    net:          drift network called as net.apply(params, t, x, scale); default MLPDrift(dim).
    """

    def __init__(self, energy, dim, temperature, *, x0=None, steps=100, paths=512, prior_std=3.5, sigma_min=0.05,
                 lr=1e-3, net=None):
        self.energy = jax.vmap(energy)
        self.dim, self.T, self.steps, self.paths = dim, temperature, steps, paths
        self.x0 = jnp.zeros(dim) if x0 is None else jnp.asarray(x0, jnp.float32)
        sigma, self.dt = linear_schedule(steps, prior_std, sigma_min)
        self.sigma = jnp.asarray(sigma)
        self.t = jnp.arange(steps) * self.dt
        self.var = float(np.sum(sigma ** 2) * self.dt)      # sigma_bar^2, variance of the uncontrolled endpoint
        self.scale = jnp.float32(np.sqrt(self.var))
        self.net = MLPDrift(dim) if net is None else net
        self.opt = optax.adam(lr)
        self.rollout = jax.jit(self._rollout)
        self.update = jax.jit(self._update)

    # ------------------------------------------------------------------------------------------------ model
    def init(self, key):
        params = self.net.init(key, jnp.zeros(()), jnp.zeros((1, self.dim)), self.scale)
        return params, self.opt.init(params)

    def drift(self, params, t, x):
        """u_theta(t, x) for a batch x (..., dim) at one time t."""
        return self.net.apply(params, jnp.full(x.shape[:-1], t), x, self.scale)

    # ------------------------------------------------------------------------------------------------ paths
    def _rollout(self, params, key):
        """Euler–Maruyama paths. Returns the states before each step xs (steps, paths, dim), the increments dX, the
        endpoints X_T (paths, dim), and the Girsanov term  int u.dW / sigma + 0.5 int |u|^2 / sigma^2 dt  (paths,)."""
        dt = self.dt

        def step(carry, inp):
            x, gir = carry
            t, s, k = inp
            u = self.drift(params, t, x)
            dw = jnp.sqrt(dt) * jax.random.normal(k, x.shape)
            dx = u * dt + s * dw
            gir = gir + jnp.sum(u * dw, -1) / s + 0.5 * jnp.sum(u ** 2, -1) / s ** 2 * dt
            return (x + dx, gir), (x, dx)

        x_init = jnp.broadcast_to(self.x0, (self.paths, self.dim))
        keys = jax.random.split(key, self.steps)
        (x_T, gir), (xs, dxs) = jax.lax.scan(step, (x_init, jnp.zeros(self.paths)), (self.t, self.sigma, keys))
        return xs, dxs, x_T, gir

    def psi(self, x):
        """Terminal cost Psi = log(dmu0 / dmu) = U / T - |x - x0|^2 / (2 sigma_bar^2), up to a constant."""
        return self.energy(x) / self.T - 0.5 * jnp.sum((x - self.x0) ** 2, -1) / self.var

    def log_weights(self, x_T, gir):
        """Self-normalized log importance weights log w_i, w_i ∝ exp(-S_i), S_i = Girsanov term + Psi(X_T^(i))."""
        log_w = -(gir + self.psi(x_T))
        log_w = jnp.where(jnp.isfinite(log_w), log_w, -jnp.inf)
        return log_w - jax.nn.logsumexp(log_w)

    # ------------------------------------------------------------------------------------------------ training
    def loss(self, params, xs, dxs, w):
        """Weighted regression of the drift onto the increments: 0.5 sum_i w_i sum_k dt/sigma_k^2 |dX/dt - u|^2."""
        u = jax.vmap(lambda t, x: self.drift(params, t, x))(self.t, xs)                  # (steps, paths, dim)
        r = jnp.sum((dxs / self.dt - u) ** 2, -1) * self.dt / self.sigma[:, None] ** 2   # (steps, paths)
        return 0.5 * jnp.sum(w * jnp.sum(r, 0))

    def _update(self, params, opt_state, xs, dxs, w):
        loss, grads = jax.value_and_grad(self.loss)(params, xs, dxs, w)
        upd, opt_state = self.opt.update(grads, opt_state, params)
        return optax.apply_updates(params, upd), opt_state, loss

    def step(self, params, opt_state, key):
        """One PICE iteration: rollout, weights, one Adam step. `out` holds this iteration's paths and weights."""
        xs, dxs, x_T, gir = self.rollout(params, key)
        log_w = self.log_weights(x_T, gir)
        params, opt_state, loss = self.update(params, opt_state, xs, dxs, jnp.exp(log_w))
        return params, opt_state, dict(xs=xs, dxs=dxs, x_T=x_T, log_w=log_w, loss=loss)

    def sample(self, params, key):
        """Endpoints X_T of one batch of paths under the learned drift."""
        return self.rollout(params, key)[2]


def ess(log_w):
    """Effective sample size 1 / sum w_i^2 of normalized log weights."""
    return float(1.0 / jnp.sum(jnp.exp(2.0 * log_w)))
