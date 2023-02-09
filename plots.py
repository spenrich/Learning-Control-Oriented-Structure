"""
Plot results.

Author: Spencer M. Richards
        Autonomous Systems Lab (ASL), Stanford
        (GitHub: spenrich)
"""
import os

import dill

import jax

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import numpy as np

import scipy

# Configure JAX before importing anything that uses JAX
jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_enable_x64', True)

# Create directory for saving figures
directory = 'figures'
os.makedirs(directory, exist_ok=True)

# Plot customization
plt.rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['Times New Roman', 'CMU Serif'],
    'mathtext.fontset':  'cm',
    'font.size':         24,
    'legend.fontsize':   'medium',
    'axes.titlesize':    'large',
    'lines.linewidth':   2,
    'lines.markersize':  10,
})
alpha = 0.2

systems = ['PlanarSpacecraft', 'PlanarBirotor']
titles = ['Spacecraft', 'PVTOL']
controllers = ['LQR', 'CCM', 'SDC', 'LQR (actual)']
seeds = list(range(5))
Ns = [50, 100, 200, 500, 1000]

start_locally = False
fill_whiskers = False
log_scale = True

colors = plt.get_cmap('tab10').colors[:len(controllers)]
legend_kwargs = {
    'handles':          [Patch(label=ctrl, color=color)
                         for ctrl, color in zip(controllers, colors)],
    'loc':              'lower center',
    'ncol':             len(controllers),
    'bbox_to_anchor':   (0.5, 0.),
}


###############################################################################
# Plot error aggregated across test trajectories and random seeds
###############################################################################
fig, ax = plt.subplots(len(systems), 1, figsize=(16, 16), dpi=100,
                       sharex=True, sharey=False)
ax = np.atleast_1d(ax)
alpha = 0.2
for k, system in enumerate(systems):
    ax[k].set_title(titles[k])
    rmse = {ctrl: np.zeros((len(seeds), len(Ns))) for ctrl in controllers}

    for ctrl in controllers:
        for i, seed in enumerate(seeds):
            for j, N in enumerate(Ns):
                # Load test results
                prefix = 'seed={}_N={}'.format(seed, N)
                if start_locally:
                    prefix += '_local'
                path = os.path.join('test_results', system, ctrl,
                                    prefix + '.dill')
                with open(path, 'rb') as file:
                    sims = dill.load(file)

                # Compute normalized trajectory error over time
                t, x, x_ref = sims['t'], sims['x'], sims['x_ref']
                e_sq = np.sum((x - x_ref)**2, axis=-1)
                e_sq /= e_sq[:, 0].reshape((-1, 1))

                # Compute RMSE (averaged across all test trajectories)
                rmse_all = np.sqrt(
                    scipy.integrate.trapezoid(e_sq, t, axis=-1) / t[-1]
                )
                rmse[ctrl][i, j] = np.mean(rmse_all)

    for ctrl, color in zip(controllers, colors):
        # Compute statistics
        mean, std = np.mean(rmse[ctrl], axis=0), np.std(rmse[ctrl], axis=0)
        q1, q2, q3 = np.quantile(rmse[ctrl], (0.25, 0.50, 0.75), axis=0)
        iqr = q3 - q1

        # Plot
        # ls = (0, (10, 15)) if name == 'LQR (actual)' else '-'
        ls = (0, (10, 6)) if ctrl == 'LQR (actual)' else '-'
        if log_scale:
            ax[k].loglog(Ns, q2, marker='o', label=ctrl, color=color, ls=ls)
        else:
            ax[k].plot(Ns, q2, marker='o', label=ctrl, color=color, ls=ls)
        if fill_whiskers:
            whisker_lo, whisker_hi = q1 - 1.5*iqr, q3 + 1.5*iqr
            ax[k].fill_between(Ns, whisker_lo, whisker_hi,
                               color=color, alpha=alpha)
        else:
            ax[k].fill_between(Ns, q1, q3, color=color, alpha=alpha)

for a in ax:
    a.set_ylabel(r'$\dfrac{1}{N_\mathrm{test}}\sum_{k=1}^{N_\mathrm{test}}\ '
                 r'\sqrt{\dfrac{1}{T^{(k)}}\int_0^{T^{(k)}}\!\dfrac{'
                 r'||e^{(k)}(t)||_2^2}{||e^{(k)}(0)||_2^2}\,dt}$')
    a.set_ylabel(r'$\mathrm{RMS}(\mathcal{T}_\mathrm{test})$')
ax[1].set_xlabel(r'$N$')

fig.legend(**legend_kwargs)
fig.tight_layout()
fig.subplots_adjust(bottom=0.12)

if start_locally:
    fig.savefig(os.path.join(directory, 'efficiency_local.png'),
                bbox_inches='tight')
else:
    fig.savefig(os.path.join(directory, 'efficiency.png'), bbox_inches='tight')


###############################################################################
# Plot all trajectories for a single seed
###############################################################################
seeds = [4, 4]
pads = [45, 40]

row_size, col_size = 4, 6
figsize = (len(Ns)*col_size, len(systems)*row_size)
fig, ax = plt.subplots(len(systems), len(Ns), figsize=figsize, dpi=100,
                       sharex=True, sharey=False)
ax = np.atleast_1d(ax)

for a in ax[-1, :]:
    a.set_xlabel(r'$t$')
for a in ax[:, 0]:
    a.set_ylabel(r'$||e(t)||_2 / ||e(0)||_2$')
for a, N in zip(ax[0, :], Ns):
    a.set_title(r'$N = {}$'.format(N))

for ctrl, color in zip(controllers, colors):
    for i, (system, title, seed, pad) in enumerate(zip(systems, titles, seeds,
                                                       pads)):
        for j, N in enumerate(Ns):
            # Load test results
            prefix = 'seed={}_N={}'.format(seed, N)
            if start_locally:
                prefix += '_local'
            path = os.path.join('test_results', system, ctrl, prefix + '.dill')
            with open(path, 'rb') as file:
                sims = dill.load(file)

            # Compute normalized trajectory error over time
            t, x, x_ref = sims['t'], sims['x'], sims['x_ref']
            errors = np.linalg.norm(x - x_ref, axis=-1)
            errors /= errors[:, 0].reshape((-1, 1))

            # Compute statistics
            mean, std = np.mean(errors, axis=0), np.std(errors, axis=0)
            q1, q2, q3 = np.quantile(errors, (0.25, 0.50, 0.75), axis=0)
            iqr = q3 - q1
            whisker_lo, whisker_hi = q1 - 1.5*iqr, q3 + 1.5*iqr

            # Plot
            ls = (0, (10, 6)) if ctrl == 'LQR (actual)' else '-'
            if log_scale:
                ax[i, j].semilogy(t, q2, label=ctrl, color=color, ls=ls)
            else:
                ax[i, j].plot(t, q2, label=ctrl, color=color, ls=ls)
            if fill_whiskers:
                ax[i, j].fill_between(t, whisker_lo, whisker_hi,
                                      color=color, alpha=alpha)
            else:
                ax[i, j].fill_between(t, q1, q3, color=color, alpha=alpha)
        ax[i, -1].yaxis.set_label_position('right')
        ax[i, -1].set_ylabel(title, rotation=270, labelpad=25)

fig.legend(**legend_kwargs)
fig.tight_layout()
fig.subplots_adjust(bottom=0.22)

if start_locally:
    fig.savefig(os.path.join(directory, 'trends_local.png'),
                bbox_inches='tight')
else:
    fig.savefig(os.path.join(directory, 'trends.png'), bbox_inches='tight')


###############################################################################
# Plot double-loop trajectory for the PVTOL (PlanarBirotor)
###############################################################################
seed = 0
row_size, col_size = 4, 5
figsize = (len(Ns)*col_size, 2*row_size)
fig, ax = plt.subplots(2, len(Ns), figsize=figsize, dpi=100,
                       sharex='row', sharey='row')
ax = np.atleast_2d(ax)

# Axes titling
pad = 0.8
ax[0, 0].set_ylabel(r'$p_y$')
for a in ax[0, :]:
    a.set_xlabel(r'$p_x$')
ax[1, 0].set_ylabel(r'$||e(t)||_2 / ||e(0)||_2$')
for a in ax[1, :]:
    a.set_xlabel(r'$t$')
for a, N in zip(ax[0, :], Ns):
    a.set_title(r'$N = {}$'.format(N))
for a in ax[0, :]:
    a.set_xlim(-5. - pad, 5. + pad)
    a.set_ylim(-pad, 5. + pad)

# Plot trajectories
for j, N in enumerate(Ns):
    for ctrl, color in zip(controllers, colors):
        # Load test results
        prefix = 'seed={}_N={}'.format(seed, N)
        if start_locally:
            prefix += '_local'
        path = os.path.join('test_results', 'PlanarBirotor_DoubleLoop', ctrl,
                            prefix + '.dill')
        with open(path, 'rb') as file:
            sims = dill.load(file)

        # Compute normalized trajectory error over time
        t, x, x_ref = sims['t'], sims['x'], sims['x_ref']
        error = np.linalg.norm(x - x_ref, axis=-1)
        error /= error[0]

        # Plot
        # ls = (0, (10, 6)) if name == 'LQR (actual)' else '-'
        ax[0, j].plot(x[:, 0], x[:, 1], color=color)
        if log_scale:
            ax[1, j].semilogy(t, error, color=color)
        else:
            ax[1, j].plot(t, error, color=color)

# Plot double-loop reference trajectory
for a in ax[0, :]:
    a.plot(x_ref[:, 0], x_ref[:, 1], '--k')

# Legend
fig.legend(**legend_kwargs)
fig.tight_layout()
fig.subplots_adjust(bottom=0.2)

# Save figure
fig.savefig(os.path.join(directory, 'double_loop.png'), bbox_inches='tight')
