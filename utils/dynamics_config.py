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

    if system_name == 'InvertedPendulum':
        g, m, L = 9.81, 1., 1.
        params_dict = {'gravity': g, 'mass': m, 'length': L}
        labels_x = (r'$\theta$', r'$\dot{\theta}$')
        labels_u = (r'$\tau$',)

        skip_dim = 1
        sparse_actuator = True
        x_ub = np.inf
        u_ub = 2*m*g*L
        x_max = np.array([np.pi/3, np.pi/4])
        u_max = u_ub
        x_lb, u_lb, x_min, u_min = -x_ub, -u_ub, -x_max, -u_max
        e_max = 0.5
        Q, R, T, N = 1., 1e-2, 5., 100

    elif system_name == 'CartPole':
        g, L, mc, mp = 9.81, 1., 0.1, 0.8
        params_dict = {'gravity': g, 'length': L,
                       'mass_cart': mc, 'mass_pendulum': mp}
        labels_x = (r'$p$', r'$\theta$', r'$\dot{p}$', r'$\dot{\theta}$')
        labels_u = (r'$u$',)

        skip_dim = 1
        sparse_actuator = False
        x_ub = np.inf
        # u_ub = np.inf
        u_ub = 20.
        # x_max = np.array([5., np.pi/2, 2., np.pi/2])
        # x_max = np.array([10., np.pi/3, 10., np.pi])
        # x_max = np.array([5., np.pi/2, 2., np.pi/2])
        # x_max = np.array([5., np.pi/3, 1., np.pi/3])
        x_max = np.array([1., np.pi/4, 0.5, np.pi/4])
        # u_max = 100.
        # u_max = 60.
        u_max = 20.
        x_lb, u_lb, x_min, u_min = -x_ub, -u_ub, -x_max, -u_max
        e_max = 0.5
        Q, R, T, N = 1., 1e-2, 5., 100

    elif system_name == 'PlanarBirotor':
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
        Q, R, T, N = 1., 1e-2, 5., 100

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
        Q, R, T, N = 1., 1e-2, 5., 100
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
        'Q':                Q,                # state weight for MPC
        'R':                R,                # input weight for MPC
        'T':                T,                # time horizon for MPC
        'N':                N,                # shooting nodes for MPC
        'labels_x':         labels_x,         #
        'labels_u':         labels_u,         #
    }
    return system, config
