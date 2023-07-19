"""
Configuration parameters for various dynamical systems.

Author: Spencer M. Richards
        Autonomous Systems Lab (ASL), Stanford
        (GitHub: spenrich)
"""
from importlib import import_module

import numpy as np


def get_config(system_name: str, backend: str = 'numpy'):
    """TODO."""
    backend = backend.lower()
    if backend not in ('numpy', 'casadi', 'jax'):
        raise ValueError('Argument `backend` must be one of: '
                         '`numpy`, `casadi`, `jax`.')

    module = 'utils.dynamics_' + backend
    cls = getattr(import_module(module), system_name)

    if system_name == 'PlanarBirotor':
        g, m, L, J = 9.81, 0.5, 0.25, 0.005
        params_dict = {'gravity': g, 'mass': m, 'length': L, 'inertia': J}
        labels_x = (r'$p_x$', r'$p_y$', r'$\phi$',
                    r'$v_x$', r'$v_y$', r'$\dot{\phi}$')
        labels_u = (r'$F_R$', r'$F_L$')

        skip_dim = 2
        sparse_actuator = True
        # x_ub = np.inf
        x_ub = np.array([np.inf, np.inf, np.pi, np.inf, np.inf, np.inf])
        x_lb = -x_ub
        u_lb, u_ub = np.array([0.1, 2.])*m*g
        x_max = np.array([10., 10., np.pi/3, 2., 1., np.pi/3])
        x_min = -x_max
        u_min, u_max = m*g/2 - 1., m*g/2 + 1.
        e_max = 0.5

    elif system_name == 'PlanarSpacecraft':
        m, J, d = 0.5, 0.005, 0.1
        params_dict = {'mass': m, 'inertia': J, 'offset': d}
        labels_x = (r'$p_x$', r'$p_y$', r'$\theta$',
                    r'$\dot{p}_x$', r'$\dot{p}_y$', r'$\dot{\theta}$')
        labels_u = (r'$F_x$', r'$F_y$', r'$M$')

        skip_dim = 3
        sparse_actuator = True
        x_ub = np.inf
        u_ub = np.array([1., 1., 0.1])
        x_max = np.array([1., 1., np.pi, 0.2, 0.2, 0.25])
        u_max = u_ub
        x_lb, u_lb, x_min, u_min = -x_ub, -u_ub, -x_max, -u_max
        e_max = 0.2

    elif system_name == 'ThreeLinkManipulator':
        n_dof = 3
        params_dict = {
            'lengths':  1.,
            'masses':   1.,
            'inertias': 1e-2,
        }
        labels_x = (
            tuple(r'$\theta_{}$'.format(i + 1) for i in range(n_dof))
            + tuple(r'$\dot\theta_{}$'.format(i + 1) for i in range(n_dof))
        )
        labels_u = tuple(r'$\tau_{}$'.format(i + 1) for i in range(n_dof))

        skip_dim = n_dof
        sparse_actuator = True
        x_ub = np.array([np.pi, np.pi/2, np.pi/2, np.inf, np.inf, np.inf])
        u_ub = 20. * np.ones(n_dof)
        x_max = np.pi / np.array([1., 2., 2., 8., 8., 8.])
        u_max = u_ub
        x_lb, u_lb, x_min, u_min = -x_ub, -u_ub, -x_max, -u_max
        e_max = 0.5

    else:
        raise NotImplementedError()

    e_min = -e_max
    system = cls(**params_dict)
    config = {
        'skip_dim':         skip_dim,         # dimensions skipped in C3M
        'sparse_actuator':  sparse_actuator,  # flags `B(x)` as sparse in C3M
        'x_lb':             x_lb,             # state lower bound for MPC
        'x_ub':             x_ub,             # state upper bound for MPC
        'u_lb':             u_lb,             # input lower bound for MPC
        'u_ub':             u_ub,             # input upper bound for MPC
        'x_min':            x_min,            # state lower bound for sampling
        'x_max':            x_max,            # state upper bound for sampling
        'u_min':            u_min,            # input lower bound for sampling
        'u_max':            u_max,            # input upper bound for sampling
        'e_min':            e_min,            # error lower bound for sampling
        'e_max':            e_max,            # error upper bound for sampling
        'Q':                1.,               # state weight for MPC
        'R':                1e-2,             # input weight for MPC
        'T':                5.,               # time horizon for MPC
        'N':                100,              # shooting nodes for MPC
        'labels_x':         labels_x,         # plot labels for states
        'labels_u':         labels_u,         # plot labels for inputs
    }
    return system, config
