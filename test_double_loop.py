"""
Test learned PVTOL controllers on tracking a double loop-the-loop trajectory.

Author: Spencer M. Richards
        Autonomous Systems Lab (ASL), Stanford
        (GitHub: spenrich)
"""

import argparse
import os
from functools import partial

import dill

import equinox as eqx

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

from tqdm.auto import tqdm

# Configure JAX before importing anything that uses JAX
jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_enable_x64', True)

from utils.dynamics_config import get_config    # noqa: E402
from utils.misc import simulate                 # noqa: E402

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('seed', help='PRNG seed', type=int)
parser.add_argument('N', help='training set size', type=int)
parser.add_argument('--clip_ctrl', help='clip control signal',
                    action='store_true')
parser.add_argument('--zoh', help='zero-order hold', action='store_true')
parser.add_argument('--freq', nargs='?', help='control frequency',
                    type=int, default=150)
args = parser.parse_args()

prefix = 'PlanarBirotor_seed={}_N={}'.format(args.seed, args.N)

# Load true system
system, config = get_config('PlanarBirotor', backend='jax')
x_min, x_max = config['x_min'], config['x_max']
u_lb, u_ub = config['u_lb'], config['u_ub']
e_min, e_max = config['e_min'], config['e_max']

# Simulation options
Q, R, T = 1., 1., 10.
freq = args.freq
zoh = args.zoh
clip_ctrl = args.clip_ctrl


# Construct reference trajectory (double loop)
def double_loop(t):
    """Evaluate the position along the double loop trajectory at time `t`."""
    T = 5.      # loop period
    scale = 1.  #
    d = 10.     # displacement along `x` from `t=0` to `t=T`
    w = 3.      # loop width (upper bound)
    h = 5.      # loop height
    d, w, h = scale*d, scale*w, scale*h
    x = w*jnp.sin(2*jnp.pi * t/T) + (d/2)*(t/T) - d/2
    y = (h/2)*(1 - jnp.cos(2*jnp.pi * t/T))
    r = jnp.squeeze(jnp.column_stack((x, y)))
    return r


def flat_to_rot(z, dz, ddz, d3z, d4z, g=9.81):
    """Convert the differentially flat state to the rotational state."""
    wx, dwx, ddwx = -ddz[0], -d3z[0], -d4z[0]
    wy, dwy, ddwy = ddz[1] + g, d3z[1], d4z[1]
    u0 = jnp.sqrt(wx**2 + wy**2)
    ϕ = jnp.arctan(wx / wy)
    dϕ = (dwx*wy - wx*dwy) / (u0**2)
    ddϕ = (ddwx*wy - wx*ddwy - 2*dϕ*(wx*dwx + wy*dwy)) / (u0**2)
    return u0, ϕ, dϕ, ddϕ


def flat_to_state(z, dz, ddz, d3z, d4z, g=9.81):
    """Convert the differentially flat state to the PVTOL state and input."""
    u0, ϕ, dϕ, ddϕ = flat_to_rot(z, dz, ddz, d3z, d4z, g)
    s = jnp.array([z[0], z[1], ϕ, dz[0], dz[1], dϕ])
    # ds = jnp.array([dz[0], dz[1], dϕ, ddz[0], ddz[1], ddϕ])
    u = jnp.array([u0, ddϕ])
    return s, u


def reference_func(t, flat_pos, system=system):
    """Compute a reference pair `(s, u)` at time `t`."""
    g, m, L, J = system.gravity, system.mass, system.length, system.inertia
    vel = jax.jacfwd(flat_pos)
    acc = jax.jacfwd(vel)
    jerk = jax.jacfwd(acc)
    snap = jax.jacfwd(jerk)
    z, dz, ddz, d3z, d4z = flat_pos(t), vel(t), acc(t), jerk(t), snap(t)
    s, u = flat_to_state(z, dz, ddz, d3z, d4z, g)

    # Map inertial velocities to body-fixed velocities
    ϕ = s[2]
    dp = s[3:5]  # (dx, dy)
    R = jnp.array([[jnp.cos(ϕ), -jnp.sin(ϕ)],
                   [jnp.sin(ϕ), jnp.cos(ϕ)]])
    s = s.at[3:5].set(R.T @ dp)

    # Map (thrust, moment) to individual rotor thrusts
    H = jnp.array([[m, J/L],
                   [m, -J/L]]) / 2
    u = H @ u

    return s, u


# Compute reference trajectory nodes
N_ref = 1000
t_ref = jnp.linspace(0., T, N_ref + 1)
x_ref, u_ref = jax.vmap(reference_func, in_axes=(0, None))(t_ref, double_loop)

# Load models and initialize controllers
controllers = {}
for name in ('LQR', 'CCM', 'SDC'):
    prefix = 'seed={}_N={}'.format(args.seed, args.N)
    path = os.path.join('trained_models', 'PlanarBirotor', name,
                        prefix + '.dill')
    with open(path, 'rb') as file:
        results = dill.load(file)
    key = results['key']
    controllers[name] = results['model']
if hasattr(system, 'sdc'):
    controllers['LQR (actual)'] = partial(system.tracking_controller,
                                          Q=Q, R=R, linearize=True)
    # controllers['SDC (actual)'] = partial(system.tracking_controller,
    #                                       Q=Q, R=R, linearize=False)
else:
    controllers['LQR (actual)'] = partial(system.tracking_controller, Q=Q, R=R)
for name, ctrl in controllers.items():
    controllers[name] = eqx.filter_jit(ctrl)
    _ = controllers[name](x_ref[0], x_ref[0], u_ref[0])

# Set start state at beginning of reference with zero roll and velocity
x0 = x_ref[0].at[2:].set(0.)

# Simulate
t = jnp.linspace(0., T, int(T * freq) + 1)
sims = {}
for name, ctrl in (pbar := tqdm(controllers.items())):
    pbar.set_postfix({'controller': name})
    x, u, xr, ur, J = simulate(system, ctrl, x0, t, t_ref, x_ref, u_ref,
                               u_lb, u_ub, Q, R, clip_ctrl, zoh)
    sims[name] = {'t': t, 'x': x, 'u': u, 'x_ref': xr, 'u_ref': ur, 'J': J}

# Save results
for name in controllers:
    directory = os.path.join('test_results', 'PlanarBirotor_DoubleLoop', name)
    os.makedirs(directory, exist_ok=True)
    prefix = 'seed={}_N={}'.format(args.seed, args.N)
    path = os.path.join(directory, prefix + '.dill')
    with open(path, 'wb') as file:
        dill.dump(sims[name], file)

# Plotting
fig, ax = plt.subplots(1, 2, figsize=(16, 8), dpi=100)
colors = plt.get_cmap('tab10').colors[:len(controllers)]
alpha = 0.2

for a in ax:
    a.set_ylabel(r'$\dfrac{||x(t) - \bar{x}(t)||_2}{||x(0) - \bar{x}(0)||_2}$')
ax[1].set_xlabel(r'$t$')
for name, color in zip(controllers, colors):
    x, xr = sims[name]['x'], sims[name]['x_ref']
    ax[0].plot(x[:, 0], x[:, 1], label=name, color=color)

    error = jnp.linalg.norm(x - xr, axis=-1)
    error /= error[0]
    ax[1].semilogy(t, error, label=name, color=color)
ax[0].plot(xr[:, 0], xr[:, 1], '--k')
ax[0].set_xlim(-5.1, 5.1)
ax[0].set_ylim(-0.1, 5.1)
ax[-1].legend()

fig.tight_layout()

path = os.path.join('figures', 'testing', 'PlanarBirotor_DoubleLoop',
                    prefix + '.png')
fig.savefig(path, bbox_inches='tight')
