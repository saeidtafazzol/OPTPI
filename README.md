# OPTPI: optimization as sampling with path integrals

Presentation: <https://saeidtafazzol.github.io/OPTPI/> (`index.html`).

A path integral sampler trained with the forward KL (Path Integral Cross-Entropy, PICE). It learns a drift
u_θ(t, x) so that the SDE dX = u_θ dt + σ(t) dW, started at a point, ends at samples of exp(−U(x)/T) for an energy U.

## Layout

```
PI.py            the PICE sampler, for any energy U written in JAX
toy/energy.py    the 2-D toy energy (six basins and a plateau)
toy/train.py     trains PICE on the toy and records every iteration for the page
page/            template.html and build.py, which builds index.html from a toy run
```

## Use on your own problem

```python
import jax, jax.numpy as jnp
from PI import PICE

def U(x):                                   # one point x of shape (dim,) -> scalar
    return (jnp.sum(x ** 2) - 4.0) ** 2 / 4.0

pice = PICE(U, dim=3, temperature=0.1, paths=256, prior_std=2.0)
key = jax.random.PRNGKey(0)
params, opt_state = pice.init(key)
for it in range(1000):
    key, k = jax.random.split(key)
    params, opt_state, out = pice.step(params, opt_state, k)   # out: paths, endpoints, log weights, loss
samples = pice.sample(params, key)                              # (paths, dim)
```

`PICE` also takes the start point `x0`, the number of Euler–Maruyama `steps`, `sigma_min` (σ at t = 1; σ(t) is linear and
scaled so the uncontrolled endpoint has standard deviation `prior_std`), the Adam learning rate `lr`, and a custom drift
network `net` (any Flax module called as `net(t, x, scale)`).

## Reproduce the toy and the page

```
pip install -r requirements.txt
python toy/train.py --out runs/toy          # 2000 iterations, 512 paths, about 1 min on a GPU, ~850 MB of records
python page/build.py --run runs/toy         # writes index.html
```
