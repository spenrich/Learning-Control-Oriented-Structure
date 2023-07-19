"""
Test learned controllers on trajectory tracking tasks.

Author: Spencer M. Richards
        Autonomous Systems Lab (ASL), Stanford
        (GitHub: spenrich)
"""

import argparse
import os
from functools import partial
from itertools import cycle

import dill

import equinox as eqx

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

from tqdm.auto import tqdm

# Configure JAX before importing anything that uses JAX
jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_enable_x64', True)

from utils.dynamics_config import get_config  # noqa: E402
from utils.dynamics_jax import PlanarBirotor, PlanarSpacecraft  # noqa: E402
from utils.misc import simulate  # noqa: E402


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('system', help='system name', type=str)
parser.add_argument('seed', help='PRNG seed', type=int)
parser.add_argument('N', help='training set size', type=int)
parser.add_argument('--start_locally', help='start close to reference',
                    action='store_true')
parser.add_argument('--clip_ctrl', help='clip control signal',
                    action='store_true')
parser.add_argument('--zoh', help='zero-order hold', action='store_true')
parser.add_argument('--freq', nargs='?', help='control frequency',
                    type=int, default=150)
parser.add_argument('--traj', action='store_true',
                    help='test models trained on trajectory data')
parser.add_argument('--roa', action='store_true',
                    help='test RoA around equilibrium point')
parser.add_argument('--grid', nargs='?', help='grid length',
                    type=int, default=10)
args = parser.parse_args()

# Load true system
system, config = get_config(args.system, backend='jax')
x_min, x_max = config['x_min'], config['x_max']
u_lb, u_ub = config['u_lb'], config['u_ub']
e_min, e_max = config['e_min'], config['e_max']

# Simulation options
Q, R, T = 1., 1., 10.
freq = args.freq
zoh = args.zoh
clip_ctrl = args.clip_ctrl
start_locally = args.start_locally

# Load reference trajectory nodes
if args.roa:
    grid_length = args.grid
    num_traj = grid_length**2
    x_bar, u_bar = system.equilibrium
    t_bar = jnp.array([0.0, 1.0])
    t_ref = jnp.broadcast_to(t_bar, (num_traj, t_bar.size))
    x_ref = jnp.broadcast_to(x_bar, (num_traj, t_bar.size, system.state_dim))
    u_ref = jnp.broadcast_to(u_bar, (num_traj, t_bar.size, system.control_dim))
else:
    path = os.path.join('test_trajectories', args.system + '.dill')
    with open(path, 'rb') as file:
        traj_data = dill.load(file)
    t_ref = jnp.array(traj_data['t'])[:, :-1]
    x_ref = jnp.array(traj_data['x'])[:, :-1]
    u_ref = jnp.array(traj_data['u'])
    num_traj = t_ref.shape[0]

# Load models and initialize controllers
controllers = {}
for name in ('LQR', 'CCM', 'SDC'):
    prefix = 'seed={}_N={}'.format(args.seed, args.N)
    if args.traj:
        prefix += '_traj'
    path = os.path.join('trained_models', args.system, name, prefix + '.dill')
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
    _ = controllers[name](x_ref[0, 0], x_ref[0, 0], u_ref[0, 0])

# TODO.
if not args.roa:
    controllers['LQR (SDC dynamics)'] = eqx.filter_jit(
        partial(controllers['SDC'], linearize=True)
    )
    _ = controllers['LQR (SDC dynamics)'](
        x_ref[0, 0], x_ref[0, 0], u_ref[0, 0]
    )

# Sample start states
if args.roa:
    # Grid a subset of the state space with initial conditions
    if isinstance(system, (PlanarBirotor, PlanarSpacecraft)):
        x0_grid = jnp.linspace(x_min[0], x_max[0], grid_length)
        y0_grid = jnp.linspace(x_min[1], x_max[1], grid_length)
        x0, y0 = jnp.meshgrid(x0_grid, y0_grid)
        x0 = jnp.reshape(x0, (-1, 1))
        y0 = jnp.reshape(y0, (-1, 1))
        x0 = jnp.hstack([x0, y0, jnp.zeros((num_traj, system.state_dim - 2))])
    else:
        raise NotImplementedError()
else:
    key, key0 = jax.random.split(key)
    if start_locally:
        e0 = jax.random.uniform(
            key0, (num_traj, system.state_dim), minval=e_min, maxval=e_max
        )
        x0 = x_ref[:num_traj, 0, :] + e0
    else:
        x0 = jax.random.uniform(
            key0, (num_traj, system.state_dim), minval=x_min, maxval=x_max
        )

# Simulate
t = jnp.linspace(0., T, int(T * freq) + 1)
sims = {}
progress_bar = tqdm(controllers.items())
progress_bar.set_description('Testing')
for name, ctrl in progress_bar:
    progress_bar.set_postfix({'controller': name})
    simulate_ctrl = partial(simulate, system, ctrl,
                            clip_ctrl=clip_ctrl, zoh=zoh)
    in_axes = (0, None, 0, 0, 0, None, None, None, None)
    x, u, xr, ur, J = jax.vmap(simulate_ctrl, in_axes)(
        x0, t, t_ref, x_ref, u_ref, u_lb, u_ub, Q, R
    )
    sims[name] = {'t': t, 'x': x, 'u': u, 'x_ref': xr, 'u_ref': ur, 'J': J}

# Save results
prefix = 'seed={}_N={}'.format(args.seed, args.N)
if args.traj:
    prefix += '_traj'
if start_locally:
    prefix += '_local'
if args.roa:
    prefix += '_roa'

for name in controllers:
    directory = os.path.join('test_results', args.system, name)
    os.makedirs(directory, exist_ok=True)

    path = os.path.join(directory, prefix + '.dill')
    with open(path, 'wb') as file:
        dill.dump(sims[name], file)

# Plotting
fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=100)
color_cycle = cycle(plt.get_cmap('tab10').colors)
alpha = 0.2
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$\dfrac{||x(t) - \bar{x}(t)||_2}{||x(0) - \bar{x}(0)||_2}$')
for name, color in zip(controllers, color_cycle):
    x, xr = sims[name]['x'], sims[name]['x_ref']
    errors = jnp.linalg.norm(x - xr, axis=-1)
    errors /= errors[:, 0].reshape((-1, 1))

    # Compute statistics
    mean, std = jnp.mean(errors, axis=0), jnp.std(errors, axis=0)
    q1, q2, q3 = jnp.quantile(errors, jnp.array([0.25, 0.50, 0.75]), axis=0)
    iqr = q3 - q1
    whisker_lower, whisker_upper = q1 - 1.5*iqr, q3 + 1.5*iqr

    # Plot
    ax.semilogy(t, q2, label=name, color=color)
    ax.fill_between(t, q1, q3, color=color, alpha=alpha)
ax.legend()
fig.tight_layout()

# Save figure
directory = os.path.join('figures', 'testing', args.system)
os.makedirs(directory, exist_ok=True)
fig.savefig(os.path.join(directory, prefix + '.png'), bbox_inches='tight')
