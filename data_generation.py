"""
Generate test trajectories for various systems using MPC.

Author: Spencer M. Richards
        Autonomous Systems Lab (ASL), Stanford
        (GitHub: spenrich)
"""

import argparse
import os

import dill

import jax

import matplotlib.pyplot as plt

import numpy as np

from scipy.interpolate import interp1d

from tqdm import tqdm

# Configure JAX before importing anything that uses JAX
jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_enable_x64', True)

from utils.dynamics_config import get_config                # noqa: E402
from utils.dynamics_numpy import (                          # noqa: E402
    PlanarBirotor, PlanarSpacecraft, ThreeLinkManipulator
)
from utils.mpc import MPCPlanner                            # noqa: E402

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('system', help='system name', type=str)
args = parser.parse_args()

# True dynamics
system, config = get_config(args.system)
x_lb, x_ub = config['x_lb'], config['x_ub']
u_lb, u_ub = config['u_lb'], config['u_ub']
x0_min, x0_max = config['x_min'], config['x_max']
Q, R, T, N = config['Q'], config['R'], config['T'], config['N']

# Setup MPC planner
integrator_steps = 1
num_traj = 100
planner = MPCPlanner(system, N, integrator_steps, Q, R, T,
                     x_lb, x_ub, u_lb, u_ub, fixed_endpoint=True)

# Seed RNG for reproducibility, and sample initial states
n, m = system.dims
xc, uc = system.equilibrium
seed = 42
rng = np.random.default_rng(seed)
x0 = rng.uniform(x0_min, x0_max, (num_traj, n))

# Generate MPC trajectories from `x0` to equilibrium point `xc`
t_mpc_all = np.zeros((num_traj, N + 1))
x_mpc_all = np.zeros((num_traj, N + 1, n))
u_mpc_all = np.zeros((num_traj, N, m))
success = np.full(num_traj, False)
for i in tqdm(range(num_traj)):
    try:
        t_mpc_all[i], x_mpc_all[i], u_mpc_all[i] = planner.solve(x0[i], xc)
        success[i] = True
    except:  # noqa: E722
        print('failure')
        continue

# Keep only successes
t_mpc = t_mpc_all[success]
x_mpc = x_mpc_all[success]
u_mpc = u_mpc_all[success]
num_traj = t_mpc.shape[0]

# Save data
filename = os.path.join('test_trajectories',
                        system.__class__.__name__ + '.dill')
data = {'system': system, 't': t_mpc, 'x': x_mpc, 'u': u_mpc}
data['num_traj'] = num_traj
print(data['num_traj'])
with open(filename, 'wb') as file:
    dill.dump(data, file)

# Plotting
plt.close('all')
T = t_mpc.max()
dt = 0.01
t = np.arange(0., T + dt, dt)

if isinstance(system, PlanarSpacecraft):
    fig, ax = plt.subplots(3, 3, figsize=(16, 8), sharex=True)
    labels = [
        r'$x$', r'$y$', r'$\theta$',
        r'$\dot{x}$', r'$\dot{y}$', r'$\dot{\theta}$',
        r'$F_x$', r'$F_y$', r'$M$',
    ]
    for a in ax[-1]:
        a.set_xlabel(r'$t$')
    for a, label in zip(ax.ravel(), labels):
        a.set_ylabel(label)
    for i in range(num_traj):
        # Linear interpolation, using `(xc, uc)` for extrapolation
        x_bar = interp1d(t_mpc[i], x_mpc[i], axis=0, bounds_error=False,
                         fill_value=(x_mpc[i, 0], xc))
        u_bar = interp1d(t_mpc[i, :-1], u_mpc[i], axis=0, bounds_error=False,
                         fill_value=(u_mpc[i, 0], uc))
        x, u = x_bar(t), u_bar(t)
        ax[0, 0].plot(t, x[:, 0])
        ax[0, 1].plot(t, x[:, 1])
        ax[0, 2].plot(t, x[:, 2])
        ax[1, 0].plot(t, x[:, 3])
        ax[1, 1].plot(t, x[:, 4])
        ax[1, 2].plot(t, x[:, 5])
        ax[2, 0].plot(t, u[:, 0])
        ax[2, 1].plot(t, u[:, 1])
        ax[2, 2].plot(t, u[:, 2])

elif isinstance(system, PlanarBirotor):
    fig, ax = plt.subplots(2, 3, figsize=(15, 5))
    ax[0, 0].set_xlabel(r'$x$ [m]')
    ax[0, 0].set_ylabel(r'$y$ [m]')
    ax[1, 0].set_xlabel(r'$v$ [m/s]')
    ax[1, 0].set_ylabel(r'$w$ [m/s]')
    ax[0, 1].set_xlabel(r'$t$')
    ax[0, 1].set_ylabel(r'$\phi$ [deg]')
    ax[1, 1].set_xlabel(r'$t$')
    ax[1, 1].set_ylabel(r'$\dot\phi$ [deg/s]')
    ax[0, 2].set_xlabel(r'$t$')
    ax[0, 2].set_ylabel(r'$u_1/(mg)$')
    ax[1, 2].set_xlabel(r'$t$')
    ax[1, 2].set_ylabel(r'$u_2/(mg)$')

    m, g = system.mass, system.gravity
    for i in range(num_traj):
        # Linear interpolation, using `(xc, uc)` for extrapolation
        x_bar = interp1d(t_mpc[i], x_mpc[i], axis=0, bounds_error=False,
                         fill_value=(x_mpc[i, 0], xc))
        u_bar = interp1d(t_mpc[i, :-1], u_mpc[i], axis=0, bounds_error=False,
                         fill_value=(u_mpc[i, 0], uc))
        x, u = x_bar(t), u_bar(t)

        ax[0, 0].plot(x[:, 0], x[:, 1])
        ax[1, 0].plot(x[:, 3], x[:, 4])
        ax[0, 1].plot(t, x[:, 2]*180/np.pi)
        ax[1, 1].plot(t, x[:, 5]*180/np.pi)
        ax[0, 2].plot(t, u[:, 0]/(m*g))
        ax[1, 2].plot(t, u[:, 1]/(m*g))
    ax[0, 0].plot(xc[0], xc[1], 'ko')
    ax[1, 0].plot(xc[3], xc[4], 'ko')
    ax[0, 1].axhline(xc[2]*180/np.pi, c='k', ls='--')
    ax[1, 1].axhline(xc[5]*180/np.pi, c='k', ls='--')
    ax[0, 2].axhline(uc[0]/(m*g), c='k', ls='--')
    ax[1, 2].axhline(uc[1]/(m*g), c='k', ls='--')

elif isinstance(system, ThreeLinkManipulator):
    n_dof = 3
    qc, dqc = xc[:n_dof], xc[n_dof:]
    fig, ax = plt.subplots(3, n_dof, figsize=(16, 8),
                           sharex=True, sharey='row')
    for i in range(num_traj):
        # Linear interpolation, using `(xc, uc)` for extrapolation
        x_bar = interp1d(t_mpc[i], x_mpc[i], axis=0, bounds_error=False,
                         fill_value=(x_mpc[i, 0], xc))
        u_bar = interp1d(t_mpc[i, :-1], u_mpc[i], axis=0, bounds_error=False,
                         fill_value=(u_mpc[i, 0], uc))
        x, u = x_bar(t), u_bar(t)
        q, dq = x[:, :n_dof], x[:, n_dof:]
        for j in range(n_dof):
            ax[0, j].plot(t, q[:, j]*180/np.pi)
            ax[1, j].plot(t, dq[:, j]*180/np.pi)
            ax[2, j].plot(t, u[:, j])

    for a in ax.ravel():
        a.set_xlabel(r'$t$ [s]')
    for j in range(n_dof):
        ax[0, j].set_ylabel(r'$\theta_{}$ [deg]'.format(j + 1))
        ax[1, j].set_ylabel(r'$\dot\theta_{}$ [deg/s]'.format(j + 1))
        ax[2, j].set_ylabel(r'$\tau_{}$ [N$\cdot$m]'.format(j + 1))
        ax[0, j].axhline(qc[j]*180/np.pi, c='k', ls='--')
        ax[1, j].axhline(dqc[j]*180/np.pi, c='k', ls='--')
        ax[2, j].axhline(uc[j], c='k', ls='--')

else:
    raise NotImplementedError()

fig.tight_layout()
path = os.path.join('figures', 'generated_trajectories',
                    system.__class__.__name__ + '.png')
fig.savefig(path, bbox_inches='tight')
