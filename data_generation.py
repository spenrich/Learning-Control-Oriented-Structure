"""
Generate trajectories for various systems using MPC.

Author: Spencer M. Richards
        Autonomous Systems Lab (ASL), Stanford
        (GitHub: spenrich)
"""

import argparse
import os
from functools import partial

import dill

import jax

import matplotlib.pyplot as plt

import numpy as np

from scipy.integrate import cumulative_trapezoid, odeint
from scipy.interpolate import interp1d

from tqdm import tqdm

# Configure JAX before importing anything that uses JAX
jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_enable_x64', True)

from utils.dynamics_config import get_config                # noqa: E402
from utils.dynamics_numpy import (                          # noqa: E402
    PlanarBirotor, PlanarSpacecraft, ThreeLinkManipulator,
    broadcast_to_square_matrix
)
from utils.mpc import MPCPlanner                            # noqa: E402

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('system', type=str,
                    help='system name')
parser.add_argument('--seed', nargs='?', type=int, default=42,
                    help='PRNG seed')
parser.add_argument('--num_traj', nargs='?', type=int, default=100,
                    help='number of trajectories to generate')
parser.add_argument('--clip_ctrl', action='store_true',
                    help='clip control signal')
parser.add_argument('--zoh', action='store_true',
                    help='zero-order hold')
parser.add_argument('--freq', nargs='?', type=int, default=150,
                    help='control frequency')
parser.add_argument('--sim', action='store_true',
                    help='simulate tracking')
args = parser.parse_args()
seed = args.seed
num_traj = args.num_traj
clip_ctrl = args.clip_ctrl
zoh = args.zoh
freq = args.freq
do_sim = args.sim

# True dynamics
system, config = get_config(args.system, backend='numpy')
x_lb, x_ub = config['x_lb'], config['x_ub']
u_lb, u_ub = config['u_lb'], config['u_ub']
x0_min, x0_max = config['x_min'], config['x_max']
Q, R, T, N = config['Q'], config['R'], config['T'], config['N']

# Setup MPC planner
integrator_steps = 1
planner = MPCPlanner(system, N, integrator_steps, Q, R, T,
                     x_lb, x_ub, u_lb, u_ub, fixed_endpoint=True)

# Seed RNG for reproducibility, and sample initial states
n, m = system.dims
xc, uc = system.equilibrium
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
        print('failure')  # TODO: use a proper Python warning
        continue

# Keep only successes
t_mpc = t_mpc_all[success]
x_mpc = x_mpc_all[success]
u_mpc = u_mpc_all[success]
num_traj = t_mpc.shape[0]

if not do_sim:
    # Save data as test trajectories
    filename = os.path.join('test_trajectories',
                            system.__class__.__name__ + '.dill')
    data = {'system': system, 't': t_mpc, 'x': x_mpc, 'u': u_mpc}
    data['num_traj'] = num_traj
    print(data['num_traj'])
    with open(filename, 'wb') as file:
        dill.dump(data, file)
else:
    # Simulate with linearization-based LQR tracking
    def simulate(system, controller, x0, t, t_ref, x_ref, u_ref,
                 u_lb=-np.inf, u_ub=np.inf, Q=1., R=1.,
                 clip_ctrl=False, zoh=False):
        """Simulate the closed-loop system."""
        xc, uc = system.equilibrium
        if not clip_ctrl:
            u_lb, u_ub = -np.inf, np.inf
        u_lb = np.broadcast_to(u_lb, (system.control_dim,))
        u_ub = np.broadcast_to(u_ub, (system.control_dim,))
        xr_func = interp1d(t_ref, x_ref, axis=0, bounds_error=False,
                           fill_value=(x_ref[0], xc))
        ur_func = interp1d(t_ref, u_ref, axis=0, bounds_error=False,
                           fill_value=(u_ref[0], uc))
        x_ref_t = xr_func(t)
        u_ref_t = ur_func(t)

        # Simulate, possibly with a zero-order hold
        if zoh:
            x = np.zeros((t.size, system.state_dim))
            u = np.zeros((t.size, system.control_dim))
            x[0] = x0
            for k in range(t.size - 1):
                u[k] = np.clip(controller(x[k], x_ref_t[k], u_ref_t[k]),
                               u_lb, u_ub)
                x[k + 1] = system(x[k], u[k], zoh=True, dt=t[k+1]-t[k])
            u[-1] = np.clip(controller(x[-1], x_ref_t[-1], u_ref_t[-1]),
                            u_lb, u_ub)
        else:
            def ode_cl(x, t):
                xr = xr_func(t).ravel()
                ur = ur_func(t).ravel()
                u = np.clip(controller(x, xr, ur), u_lb, u_ub)
                dx = system(x, u)
                return dx

            x = odeint(ode_cl, x0, t)
            u = np.zeros((t.size, system.control_dim))
            for k in range(t.size):
                u[k] = np.clip(controller(x[k], x_ref_t[k], u_ref_t[k]),
                               u_lb, u_ub)

        # Compute state derivatives
        dx = np.zeros((t.size, system.state_dim))
        for k in range(t.size):
            dx[k] = system(x[k], u[k])

        # Integrate tracking cost with the trapezoid rule
        Q = broadcast_to_square_matrix(Q, system.state_dim)
        R = broadcast_to_square_matrix(R, system.control_dim)
        e = x - x_ref_t
        v = u - u_ref_t
        costs = (np.sum((e@Q)*e, axis=-1) + np.sum((v@R)*v, axis=-1))/2
        J = cumulative_trapezoid(costs, t, initial=0.)
        return x, u, x_ref_t, u_ref_t, J, dx

    Q, R = 1., 2.
    controller = partial(system.tracking_controller, Q=Q, R=R, linearize=True)
    t = np.linspace(0., T, int(T * freq) + 1)
    x = np.zeros((num_traj, t.size, system.state_dim))
    u = np.zeros((num_traj, t.size, system.control_dim))
    x_ref = np.zeros((num_traj, t.size, system.state_dim))
    u_ref = np.zeros((num_traj, t.size, system.control_dim))
    J = np.zeros((num_traj, t.size))
    dx = np.zeros((num_traj, t.size, system.state_dim))
    for i in tqdm(range(num_traj)):
        x[i], u[i], x_ref[i], u_ref[i], J[i], dx[i] = simulate(
            system, controller, x0[i], t,
            t_mpc[i, :-1], x_mpc[i, :-1], u_mpc[i],
            u_lb, u_ub, Q, R, clip_ctrl, zoh
        )

    # Save data as training trajectories
    filename = os.path.join('train_trajectories',
                            system.__class__.__name__ + '.dill')
    data = {
        'system':   system,
        'num_traj': num_traj,
        't_mpc':    t_mpc,
        'x_mpc':    x_mpc,
        'u_mpc':    u_mpc,
        't':        t,
        'x':        x,
        'u':        u,
        'x_ref':    x_ref,
        'u_ref':    u_ref,
        'J':        J,
        'dx':       dx,
    }
    print(data['num_traj'])
    with open(filename, 'wb') as file:
        dill.dump(data, file)

# Plotting
plt.close('all')
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
        for j in range(system.state_dim//2):
            lines = ax[0, j].plot(t_mpc[i], x_mpc[i, :, j], '.--', zorder=1)
            c = lines[0].get_color()
            ax[1, j].plot(t_mpc[i], x_mpc[i, :, system.state_dim//2 + j],
                          '.--', color=c, zorder=1)
            ax[2, j].plot(t_mpc[i, :-1], u_mpc[i, :, j],
                          '.--', color=c, zorder=1)
            if do_sim:
                ax[0, j].plot(t, x[i, :, j],
                              color=c, zorder=0)
                ax[1, j].plot(t, x[i, :, system.state_dim//2 + j],
                              color=c, zorder=0)
                ax[2, j].plot(t, u[i, :, j],
                              color=c, zorder=0)

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
        lines = ax[0, 0].plot(x_mpc[i, :, 0], x_mpc[i, :, 1],
                              '.--', zorder=1)
        c = lines[0].get_color()
        ax[1, 0].plot(x_mpc[i, :, 3], x_mpc[i, :, 4],
                      '.--', color=c, zorder=1)
        ax[0, 1].plot(t_mpc[i], x_mpc[i, :, 2]*180/np.pi,
                      '.--', color=c, zorder=1)
        ax[1, 1].plot(t_mpc[i], x_mpc[i, :, 5]*180/np.pi,
                      '.--', color=c, zorder=1)
        ax[0, 2].plot(t_mpc[i, :-1], u_mpc[i, :, 0]/(m*g),
                      '.--', color=c, zorder=1)
        ax[1, 2].plot(t_mpc[i, :-1], u_mpc[i, :, 1]/(m*g),
                      '.--', color=c, zorder=1)
        if do_sim:
            ax[1, 0].plot(x[i, :, 3], x[i, :, 4], color=c, zorder=0)
            ax[0, 1].plot(t, x[i, :, 2]*180/np.pi, color=c, zorder=0)
            ax[1, 1].plot(t, x[i, :, 5]*180/np.pi, color=c, zorder=0)
            ax[0, 2].plot(t, u[i, :, 0]/(m*g), color=c, zorder=0)
            ax[1, 2].plot(t, u[i, :, 1]/(m*g), color=c, zorder=0)
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
        for j in range(n_dof):
            q_mpc, dq_mpc = x_mpc[i, :, j], x_mpc[i, :, n_dof + j]
            lines = ax[0, j].plot(t_mpc[i], q_mpc*180/np.pi, '.--', zorder=1)
            c = lines[0].get_color()
            ax[1, j].plot(t_mpc[i], dq_mpc*180/np.pi, '.--', color=c, zorder=1)
            ax[2, j].plot(t_mpc[i, :-1], u[i, :, j], '.--', color=c, zorder=1)
            if do_sim:
                q, dq = x[i, :, j], x[i, :, n_dof + j]
                ax[0, j].plot(t, q*180/np.pi, color=c, zorder=0)
                ax[1, j].plot(t, dq*180/np.pi, color=c, zorder=0)
                ax[2, j].plot(t, u[i, :, j], color=c, zorder=0)
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
if do_sim:
    path = os.path.join('figures', 'train_trajectories',
                        system.__class__.__name__ + '.png')
else:
    path = os.path.join('figures', 'test_trajectories',
                        system.__class__.__name__ + '.png')
fig.savefig(path, bbox_inches='tight')
