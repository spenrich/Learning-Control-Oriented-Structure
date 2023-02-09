"""
Train dynamics, controllers and certificates to fit data.

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
from jax.example_libraries import optimizers

import matplotlib.pyplot as plt

import numpy as np

from tqdm.auto import tqdm

from utils.dynamics_config import get_config
from utils.misc import pytree_permute, pytree_sos
from utils.preprocessing import IdentityScaler
from utils.tracking import (NeuralCCMController, NeuralLQRController,
                            NeuralSDCController)

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('system', help='system name', type=str)
parser.add_argument('seed', help='PRNG seed', type=int)
parser.add_argument('N', help='training set size', type=int)
parser.add_argument('--Nc', nargs='?', help='constraint set size',
                    type=int, default=10000)
parser.add_argument('--epochs', nargs='?', help='number of epochs',
                    type=int, default=50000)
parser.add_argument('--lr', nargs='?', help='learning rate',
                    type=float, default=1e-3)
parser.add_argument('--holdout_frac', nargs='?', help='holdout fraction',
                    type=float, default=0.1)
parser.add_argument('--batch_frac', nargs='?', help='batch fraction',
                    type=float, default=1.)
parser.add_argument('--reg_coef', nargs='?', help='regularization coefficient',
                    type=float, default=1e-6)
parser.add_argument('--hidden_width', nargs='?', help='hidden width',
                    type=int, default=128)
parser.add_argument('--hidden_depth', nargs='?', help='hidden depth',
                    type=int, default=2)
parser.add_argument('--sample_locally',
                    help='sample constraint points locally',
                    action='store_true')
args = parser.parse_args()
prefix = '{}_seed={}_N={}'.format(args.system, args.seed, args.N)

# Configure JAX and set random seed
# jax.config.update('jax_platform_name', 'cpu')
# jax.config.update("jax_enable_x64", True)
key = jax.random.PRNGKey(args.seed)

# Get true dynamics
system, config = get_config(args.system, backend='jax')
u_lb, u_ub = config['u_lb'], config['u_ub']
x_min, x_max = config['x_min'], config['x_max']
u_min, u_max = config['u_min'], config['u_max']
e_min, e_max = config['e_min'], config['e_max']

# Generate data
# TODO: avoid loading all this data into GPU memory
n, m = system.dims
N, Nc = args.N, args.Nc
sample_locally = args.sample_locally

# Regression samples (labelled)
key, key_x, key_u = jax.random.split(key, 3)
X = jax.random.uniform(key_x, (N, n), minval=x_min, maxval=x_max)
U = jax.random.uniform(key_u, (N, m), minval=u_min, maxval=u_max)
Y = jax.vmap(system)(X, U)

# Constraint samples (unlabelled)
key, key_x, key_u, key_e, key_v = jax.random.split(key, 5)
Xc = jnp.vstack([
    X, jax.random.uniform(key_x, (Nc, n), minval=x_min, maxval=x_max)
])
Uc = jnp.vstack([
    U, jax.random.uniform(key_u, (Nc, m), minval=u_min, maxval=u_max)
])
if sample_locally:
    E = jax.random.uniform(key_e, (N + Nc, n), minval=e_min, maxval=e_max)
    Xc_ref = Xc - E
else:
    Xc_ref = jax.random.uniform(key_e, (N + Nc, n), minval=x_min, maxval=x_max)
Uc_ref = jax.random.uniform(key_v, (N + Nc, m), minval=u_min, maxval=u_max)

# Training/validation split and preconditioner
holdout_frac = args.holdout_frac
data = {'train': {}, 'valid': {}}
for name, D in zip(['X', 'U', 'Y', 'Xc', 'Uc', 'Xc_ref', 'Uc_ref'],
                   [X, U, Y, Xc, Uc, Xc_ref, Uc_ref]):
    split = int((1 - holdout_frac) * D.shape[0])
    data['train'][name], data['valid'][name] = jnp.split(D, [split, ])
preconditioner = IdentityScaler()

# Initialize controller models
key, key_model = jax.random.split(key, 2)
kwargs = {
    'model':                system,
    'hidden_width':         args.hidden_width,
    'hidden_depth':         args.hidden_depth,
    'hidden_activation':    jnp.tanh,
    'preconditioner':       preconditioner,
    'key':                  key_model,
}
kwargs_con = {
    'contraction_rate':     0.5,
    'margin':               1e-2,
    'overshoot':            10.,
}
models = {
    'LQR':  NeuralLQRController(**kwargs),
    'CCM':  NeuralCCMController(**kwargs, **kwargs_con, learn_caf=True),
    'SDC':  NeuralSDCController(**kwargs, learn_caf=True),
}
params_learned, params_static = {}, {}
for m in models:
    params_learned[m], params_static[m] = eqx.partition(models[m],
                                                        eqx.is_array)

# Define loss function and gradient-based optimizer
init_opt, update_opt, get_params = optimizers.adam(args.lr)
opt_states = {name: init_opt(params_learned[name]) for name in models}


@eqx.filter_jit
def compute_losses(model, data, reg_coef=0.):
    """TODO."""
    # print('Compiling function COMPUTE_LOSSES')
    X, U, Y, Xc, Uc, Xc_ref, Uc_ref = [
        data[k] for k in ('X', 'U', 'Y', 'Xc', 'Uc', 'Xc_ref', 'Uc_ref')
    ]
    loss_terms = jnp.array([
        model.loss_regression(X, U, Y),
        jnp.mean(jax.vmap(model.loss_auxiliary)(Xc, Uc, Xc_ref, Uc_ref)),
        reg_coef*pytree_sos(model),
    ])
    return loss_terms


def opt_step(data, idx, opt_state, reg_coef, params_static):
    """TODO."""
    # print('Compiling function OPT_STEP')
    params_learned = get_params(opt_state)
    model = eqx.combine(params_learned, params_static)
    loss_func = lambda *args: jnp.sum(compute_losses(*args))  # noqa: E731
    grads = eqx.filter_grad(loss_func)(model, data, reg_coef)
    opt_state = update_opt(idx, grads, opt_state)
    return opt_state


# Compute initial losses
step_funcs, loss_train, loss_valid, loss_best, epoch_best = {}, {}, {}, {}, {}
for name in models:
    step_funcs[name] = jax.jit(partial(opt_step,
                                       params_static=params_static[name]))
    loss_train[name] = np.reshape(compute_losses(models[name], data['train']),
                                  (1, -1))
    loss_valid[name] = np.reshape(compute_losses(models[name], data['valid']),
                                  (1, -1))
    loss_best[name] = np.array(compute_losses(models[name], data['valid']))
    epoch_best[name] = 0
epoch_offset = 1

# Do batch stochastic gradient descent
str_fmt = '{:.4f}'
reg_coef = args.reg_coef
batch_frac = args.batch_frac
num_epochs = args.epochs


def load_batch(data, batch_frac, offset_frac):
    """Load a batch of data from a larger dataset."""

    def slicer(x):
        N = x.shape[0]
        batch_size, offset = int(batch_frac*N), int(offset_frac*N)
        x_batch = x[offset:offset+batch_size]
        return x_batch

    batch = jax.tree_util.tree_map(slicer, data)
    return batch


# Pre-compile functions
for name in models:
    num_loss_terms = loss_train[name].shape[-1]
    loss_train[name] = np.vstack([loss_train[name],
                                  np.zeros((num_epochs, num_loss_terms))])
    loss_valid[name] = np.vstack([loss_valid[name],
                                  np.zeros((num_epochs, num_loss_terms))])
print('Pre-compiling ... ', end='')
for m in models:
    batch = load_batch(data['train'], batch_frac, 0.)
    _ = step_funcs[m](batch, 0, opt_states[m], reg_coef)
print('done!')

# Do batch stochastic gradient descent
print('Models:        ', list(models))
print('Initial losses:', [str_fmt.format(loss_best[m].sum()) for m in models])
for m in models:
    # Reset PRNG key to ensure data is shuffled the same way for each model
    data_shuffled = data['train']
    key_train = key

    progress_bar = tqdm(range(epoch_offset, epoch_offset + num_epochs))
    progress_bar.set_description(m)
    for epoch in progress_bar:
        # Shuffle training data
        key_train, key_shuffle = jax.random.split(key_train, 2)
        data_shuffled = pytree_permute(key_shuffle, data_shuffled)

        # Do a gradient step
        # TODO: guard against ragged batches to preserve JIT-speed
        for k in np.arange(0., 1., batch_frac):
            batch = load_batch(data_shuffled, batch_frac, k)
            opt_states[m] = step_funcs[m](batch, epoch - 1, opt_states[m],
                                          reg_coef)

        # Extract the model candidate and compute the new training loss
        model_candidate = eqx.combine(get_params(opt_states[m]),
                                      params_static[m])
        loss_train[m][epoch] = compute_losses(model_candidate, data['train'])

        # Compute the new validation loss, and update the model if the
        # validation loss has improved
        loss_valid[m][epoch] = compute_losses(model_candidate, data['valid'])
        if loss_valid[m][epoch].sum() <= loss_best[m].sum():
            epoch_best[m] = epoch
            models[m] = model_candidate
            loss_best[m] = loss_valid[m][epoch]

        # Set progress bar
        progress_bar.set_postfix({
            'train loss':   str_fmt.format(loss_train[m][epoch].sum()),
            'valid loss':   str_fmt.format(loss_valid[m][epoch].sum()),
            'best loss':    str_fmt.format(loss_best[m].sum()),
        })
epoch_offset += num_epochs

# Save hyperparameters and models
key = key_train  # save PRNG key state from after all epochs
prefix = 'seed={}_N={}'.format(args.seed, args.N)
for m in models:
    results = {
        'seed':             args.seed,
        'key':              key,
        'N':                N,
        'Nc':               Nc,
        'sample_locally':   sample_locally,
        'holdout_frac':     holdout_frac,
        'batch_frac':       batch_frac,
        'lr':               args.lr,
        'reg_coef':         reg_coef,
        'system':           system,
        'model':            models[m],
        'loss_train':       loss_train[m],
        'loss_valid':       loss_valid[m],
    }
    directory = os.path.join('trained_models', args.system, m)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, prefix + '.dill')
    with open(path, 'wb') as file:
        dill.dump(results, file)

# Plotting
fig, ax = plt.subplots(num_loss_terms, 2, figsize=(12, 8),
                       sharex=True, sharey=False)
color_cycle = cycle(plt.get_cmap('tab10').colors)

for m, c in zip(models, color_cycle):
    loss_total = loss_train[m].sum(axis=-1)
    ax[0, 0].semilogy(loss_total, label=m, color=c)
    idx_min = np.argmin(loss_total)
    ax[0, 0].semilogy(idx_min, loss_total[idx_min], 'o', c=c, mec='k', ms=7)
    ax[0, 0].semilogy(epoch_best[m], loss_total[epoch_best[m]], '*', c=c,
                      mec='k', ms=14)

    ax[0, 1].semilogy(loss_valid[m].sum(axis=-1), label=m, color=c)
    ax[0, 1].semilogy(epoch_best[m], loss_best[m].sum(), '*', c=c,
                      mec='k', ms=14)
    for i in range(num_loss_terms - 1):
        ax[i + 1, 0].semilogy(loss_train[m][:, i], label=m, color=c)
        ax[i + 1, 1].semilogy(loss_valid[m][:, i], label=m, color=c)

ax[0, 0].set_title('training')
ax[0, 1].set_title('validation')
ax[-1, 0].set_xlabel('epoch')
ax[-1, 1].set_xlabel('epoch')
ax[0, 0].set_ylabel('total loss')
ax[1, 0].set_ylabel('regression loss')
ax[2, 0].set_ylabel('auxiliary loss')
# ax[3, 0].set_ylabel('regularization loss')
ax[0, 1].legend()
fig.tight_layout()

# Save figure
directory = os.path.join('figures', 'training', args.system)
os.makedirs(directory, exist_ok=True)
fig.savefig(os.path.join(directory, prefix + '.png'), bbox_inches='tight')
