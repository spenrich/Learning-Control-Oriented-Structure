"""
Equations of motion for various dynamical systems in in CasADi.

Author: Spencer M. Richards
        Autonomous Systems Lab (ASL), Stanford
        (GitHub: spenrich)
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import casadi as cs

import numpy as np


class Dynamics(ABC):
    """Base class for autonomous continuous-time dynamical systems."""

    def __init_subclass__(cls):
        """Ensure any subclass of this class is a dataclass."""
        return dataclass(frozen=True)(cls)

    @property
    @abstractmethod
    def dims(self) -> tuple[int, int]:
        """Return the dimensions of the state and input for this system."""
        raise NotImplementedError()

    @property
    def state_dim(self) -> int:
        """Return the dimension of the state `x` for this system."""
        return self.dims[0]

    @property
    def control_dim(self) -> int:
        """Return the dimension of the control input `u` for this system."""
        return self.dims[1]

    @abstractmethod
    def dynamics(self, x, u):
        """Evaluate the dynamics `dx/dt = f(x, u)`."""
        raise NotImplementedError()

    def __call__(self, x, u, zoh=False, dt=None,
                 integrator_steps=1, integrator_order=4):
        """Evaluate this dynamics model at state `x` and control `u`."""
        if zoh:  # `y = x(t + dt)`
            if dt is None:
                raise ValueError('Argument `dt` must be provided for '
                                 'zero-order-hold control.')
            if isinstance(dt, float) and dt <= 0.:
                raise ValueError('Argument `dt` must be a positive float for '
                                 'zero-order-hold control.')
            if not isinstance(integrator_steps, int) or integrator_steps <= 0:
                raise ValueError('Argument `integrator_steps` must be a '
                                 'positive integer.')
            if not isinstance(integrator_order, int) or integrator_order <= 0:
                raise ValueError('Argument `integrator_order` must be a '
                                 'positive integer.')
            dt = dt/integrator_steps
            y = x
            for _ in range(integrator_steps):
                # Explicit Runge-Kutta (ERK) integration step
                # TODO: allow for integration orders other than 4
                k1 = self.dynamics(y, u)
                k2 = self.dynamics(y + (dt/2)*k1, u)
                k3 = self.dynamics(y + (dt/2)*k2, u)
                k4 = self.dynamics(y + dt*k3, u)
                y = y + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        else:  # `y = dx/dt`
            y = self.dynamics(x, u)
        return y


###############################################################################
#                          DYNAMICAL SYSTEMS EXAMPLES                         #
###############################################################################


class CartPole(Dynamics):
    """Dynamics of an inverted pendulum mounted on a cart."""

    gravity:        float = 9.81
    length:         float = 1.
    mass_cart:      float = 1.
    mass_pendulum:  float = 1.

    @property
    def dims(self) -> tuple[int, int]:
        """Return the dimensions of the state and input for this system."""
        return 4, 1

    def dynamics(self, x, u):
        """Evaluate the dynamics `dx/dt = f(x, u)`."""
        g, L, mc, mp = (self.gravity, self.length,
                        self.mass_cart, self.mass_pendulum)
        ??, dx, d?? = x[1], x[2], x[3]
        c??, s?? = cs.cos(??), cs.sin(??)
        h = mc + mp*(s??**2)
        f = cs.vcat([dx,
                     d??,
                     (mp*s??*(g*c?? - L*(d??**2)) + u) / h,
                     ((mc+mp)*g*s??/L - mp*(d??**2)*s??*c?? + u*c??/L) / h])
        return f


class Crane(Dynamics):
    """Dynamics of a simple crane."""

    damping:    float = 0.001
    spring:     float = 0.9

    @property
    def dims(self) -> tuple[int, int]:
        """Return the dimensions of the state and input for this system."""
        return 4, 1

    def dynamics(self, x, u):
        """Evaluate the dynamics `dx/dt = f(x, u)`."""
        ??, k = self.damping, self.spring
        n, m = self.dims
        p1, v1, p2, v2 = [x[i] for i in range(n)]
        f = cs.vcat([v1, u, v2, -??*v2 - k*(p2 - p1)])
        return f


class InvertedPendulum(Dynamics):
    """Dynamics of an inverted pendulum."""

    gravity:    float = 9.81
    mass:       float = 1.
    length:     float = 1.

    @property
    def dims(self) -> tuple[int, int]:
        """Return the dimensions of the state and input for this system."""
        return 2, 1

    def dynamics(self, x, u):
        """Evaluate the dynamics `dx/dt = f(x, u)`."""
        m, g, L = self.mass, self.gravity, self.length
        ??, d?? = x[0], x[1]
        f = cs.vcat([d??, (g/L)*cs.sin(??) + u[0]/(m*L**2)])
        return f


class PlanarBirotor(Dynamics):
    """Dynamics of a planar bi-rotor vehicle."""

    gravity:    float = 9.81
    mass:       float = 0.486
    length:     float = 0.25
    inertia:    float = 0.00383

    @property
    def dims(self) -> tuple[int, int]:
        """Return the dimensions of the state and input for this system."""
        return 6, 2

    def dynamics(self, x, u):
        """Evaluate the dynamics `dx/dt = f(x, u)`."""
        g, m, L, J = self.gravity, self.mass, self.length, self.inertia
        ??, d?? = x[2], x[5]
        c??, s?? = cs.cos(??), cs.sin(??)
        v, w = x[3], x[4]
        thrust = (u[0] + u[1])/m
        moment = (u[0] - u[1])*L/J
        f = cs.vcat([v*c?? - w*s??,
                     v*s?? + w*c??,
                     d??,
                     w*d?? - g*s??,
                     -v*d?? - g*c?? + thrust,
                     moment])
        return f


class PlanarSpacecraft(Dynamics):
    """Dynamics of a planar spacecraft with an offset center-of-mass."""

    mass:       float = 30.
    inertia:    float = 0.5
    offset:     np.ndarray = np.array([0.075, 0.075])

    def __post_init__(self):
        """TODO."""
        object.__setattr__(self, 'offset', np.broadcast_to(self.offset, (2,)))

    @property
    def dims(self) -> tuple[int, int]:
        """Return the dimensions of the state and input for this system."""
        return 6, 3

    def dynamics(self, x, u):
        """Evaluate the dynamics `dx/dt = f(x, u)`."""
        m, J = self.mass, self.inertia
        pox, poy = self.offset
        vx, vy, ?? = x[3], x[4], x[5]
        Fx, Fy, M = u[0], u[1], u[2]
        d?? = (M - pox*Fy + poy*Fx)/J
        f = cs.vcat([vx,
                     vy,
                     ??,
                     (Fx + d??*poy + (??**2)*pox)/m,
                     (Fy - d??*pox + (??**2)*poy)/m,
                     d??])
        return f


class PVTOL(Dynamics):
    """Dynamics of a planar vertical take-off and landing (PVTOL) vehicle."""

    gravity:    float = 9.81
    coupling:   float = 1.
    body_fixed: bool = True

    @property
    def dims(self) -> tuple[int, int]:
        """Return the dimensions of the state and input for this system."""
        return 6, 2

    def dynamics(self, x, u):
        """Evaluate the dynamics `dx/dt = f(x, u)`."""
        g, ?? = self.gravity, self.coupling
        ??, d?? = x[2], x[5]
        c??, s?? = cs.cos(??), cs.sin(??)
        if self.body_fixed:
            v, w = x[3], x[4]
            f = cs.vcat([v*c?? - w*s??,
                         v*s?? + w*c??,
                         d??,
                         w*d?? - g*s?? + ??*u[1],
                         -v*d?? - g*c?? + u[0],
                         u[1]])
        else:
            dx, dy = x[3], x[4]
            f = cs.vcat([dx,
                         dy,
                         d??,
                         -s??*u[0] + ??*c??*u[1],
                         -g + c??*u[0] + ??*s??*u[1],
                         u[1]])
        return f
