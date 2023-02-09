"""
Trajectory generation via MPC.

Author: Spencer M. Richards
        Autonomous Systems Lab (ASL), Stanford
        (GitHub: spenrich)
"""
from dataclasses import asdict, fields
from importlib import import_module

import casadi as cs

import numpy as np

from .dynamics_casadi import Dynamics as Dynamics_cs
from .dynamics_jax import Dynamics as Dynamics_jax
from .dynamics_numpy import Dynamics as Dynamics_np, broadcast_to_square_matrix


class MPCPlanner:
    """Plan trajectories by solving an optimal control problem."""

    def __init__(self, dynamics: Dynamics_cs | Dynamics_jax | Dynamics_np,
                 num_stages: int,
                 integrator_steps: int = 1,
                 Q: float | np.ndarray = 0.,
                 R: float | np.ndarray = 1.,
                 T: float | None = None,
                 x_lb: float | np.ndarray = -np.inf,
                 x_ub: float | np.ndarray = np.inf,
                 u_lb: float | np.ndarray = -np.inf,
                 u_ub: float | np.ndarray = np.inf,
                 dt_max: float | None = None,
                 fixed_endpoint: bool = False,
                 model_name: str | None = None,
                 backend: str = 'casadi'):
        """Initialize; see `MPCPlanner`."""
        if isinstance(dynamics, (Dynamics_np, Dynamics_jax)):
            path = __name__.rsplit('.', maxsplit=1)[0]
            cls = getattr(import_module('.dynamics_casadi', path),
                          dynamics.__class__.__name__)
            init_fields = {f.name for f in fields(cls) if f.init}
            arg_fields = {k: v for k, v in asdict(dynamics).items()
                          if k in init_fields}
            self.dynamics = cls(**arg_fields)
        else:
            self.dynamics = dynamics

        if isinstance(T, (float, int)):
            T = float(T)
            if T <= 0:
                raise ValueError('Given time horizon `T` must be positive.')
            self.free_time = False
        elif T is None:
            self.free_time = True
        else:
            raise ValueError('Time horizon `T` must be a positive float '
                             'or `None`.')
        self.T = T

        if isinstance(dt_max, (float, int)):
            dt_max = float(dt_max)
            if dt_max <= 0:
                raise ValueError('Given maximum time step `dt_max` must be '
                                 'positive.')
        elif dt_max is None:
            pass
        else:
            raise ValueError('Maximum time step `dt_max` must be a positive '
                             'float or `None`.')
        self.dt_max = dt_max

        n, m = self.dynamics.dims
        self.N = num_stages
        self.integrator_steps = integrator_steps
        self.Q = broadcast_to_square_matrix(Q, n)
        self.R = broadcast_to_square_matrix(R, m)
        self.x_lb = np.broadcast_to(x_lb, n)
        self.x_ub = np.broadcast_to(x_ub, n)
        self.u_lb = np.broadcast_to(u_lb, m)
        self.u_ub = np.broadcast_to(u_ub, m)
        self.fixed_endpoint = fixed_endpoint
        if model_name is None:
            self.model_name = dynamics.__class__.__name__
        else:
            self.model_name = model_name
        self.backend = backend

        if backend == 'casadi':
            self._casadi_objects = self.construct_mpc_casadi()
        else:
            raise ValueError('Argument `backend` must be one of: '
                             '`casadi`.')

    def construct_mpc_casadi(self):
        """Contruct an MPC optimization via CasADi."""
        n, m = self.dynamics.dims
        opti = cs.Opti()
        x0 = opti.parameter(n)              # initial state
        x = opti.variable(n, self.N + 1)    # state trajectory
        u = opti.variable(m, self.N)        # control trajectory
        if self.free_time:
            T = opti.variable()             # time horizon
            opti.subject_to(T >= 0)
            opti.set_initial(T, 10)         # TODO.
            cost = T
        else:
            T = self.T
            cost = 0.

        # Stage constraints and costs
        opti.subject_to(x[:, 0] == x0)
        dt = T / self.N
        for k in range(self.N):
            # Stage cost
            if k > 0:
                cost += x[:, k].T @ (self.Q @ x[:, k]) / 2
            cost += u[:, k].T @ (self.R @ u[:, k]) / 2

            # Dynamics constraints in discrete-time
            x_next = self.dynamics(x[:, k], u[:, k], zoh=True, dt=dt,
                                   integrator_steps=self.integrator_steps)
            opti.subject_to(x[:, k + 1] == x_next)

            # State and control bounds
            if k > 0:
                for i in range(n):
                    if np.isfinite(self.x_lb[i]):
                        opti.subject_to(x[i, k] >= self.x_lb[i])
                    if np.isfinite(self.x_ub[i]):
                        opti.subject_to(x[i, k] <= self.x_ub[i])
            for i in range(m):
                if np.isfinite(self.u_lb[i]):
                    opti.subject_to(u[i, k] >= self.u_lb[i])
                if np.isfinite(self.u_ub[i]):
                    opti.subject_to(u[i, k] <= self.u_ub[i])

        # For fixed-endpoint problems
        if self.fixed_endpoint:
            xT = opti.parameter(n)              # final state
            opti.subject_to(x[:, -1] == xT)     # terminal constraint

        # Set objective, and assemble variables and parameters
        opti.minimize(cost)
        parameters = {'x0': x0}
        if self.fixed_endpoint:
            parameters['xT'] = xT
        variables = {'x': x, 'u': u}
        if self.free_time:
            variables['T'] = T
        return opti, parameters, variables

    def solve(self, x0, xT=None, verbose=False):
        """Solve the MPC problem."""
        if self.fixed_endpoint and xT is None:
            raise ValueError('Terminal state `xT` must be provided for a '
                             'fixed-endpoint problem.')
        if self.backend == 'casadi':
            opti, parameters, variables = self._casadi_objects
            x, u = variables['x'], variables['u']
            opti.set_value(parameters['x0'], x0)
            if self.fixed_endpoint:
                opti.set_value(parameters['xT'], xT)
            if verbose:
                solver_options = {}
            else:
                solver_options = {
                    'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'
                }
            opti.solver('ipopt', solver_options)

            solution = opti.solve()
            n, m = self.dynamics.dims
            x = np.reshape(solution.value(variables['x']).T, (self.N + 1, n))
            u = np.reshape(solution.value(variables['u']).T, (self.N, m))
            if self.free_time:
                T = solution.value(variables['T'])
            else:
                T = self.T
            t = np.linspace(0, T, self.N + 1)
        return t, x, u
