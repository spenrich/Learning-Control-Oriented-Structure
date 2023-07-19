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

from .dynamics_numpy import Array


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
        φ, dx, dφ = x[1], x[2], x[3]
        cφ, sφ = cs.cos(φ), cs.sin(φ)
        h = mc + mp*(sφ**2)
        f = cs.vcat([dx,
                     dφ,
                     (mp*sφ*(g*cφ - L*(dφ**2)) + u) / h,
                     ((mc+mp)*g*sφ/L - mp*(dφ**2)*sφ*cφ + u*cφ/L) / h])
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
        β, k = self.damping, self.spring
        n, m = self.dims
        p1, v1, p2, v2 = [x[i] for i in range(n)]
        f = cs.vcat([v1, u, v2, -β*v2 - k*(p2 - p1)])
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
        θ, dθ = x[0], x[1]
        f = cs.vcat([dθ, (g/L)*cs.sin(θ) + u[0]/(m*L**2)])
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
        φ, dφ = x[2], x[5]
        cφ, sφ = cs.cos(φ), cs.sin(φ)
        v, w = x[3], x[4]
        thrust = (u[0] + u[1])/m
        moment = (u[0] - u[1])*L/J
        f = cs.vcat([v*cφ - w*sφ,
                     v*sφ + w*cφ,
                     dφ,
                     w*dφ - g*sφ,
                     -v*dφ - g*cφ + thrust,
                     moment])
        return f


class PlanarSpacecraft(Dynamics):
    """Dynamics of a planar spacecraft with an offset center-of-mass."""

    mass:       float = 30.
    inertia:    float = 0.5
    offset:     Array = np.array([0.075, 0.075])

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
        vx, vy, ω = x[3], x[4], x[5]
        Fx, Fy, M = u[0], u[1], u[2]
        dω = (M - pox*Fy + poy*Fx)/J
        f = cs.vcat([vx,
                     vy,
                     ω,
                     (Fx + dω*poy + (ω**2)*pox)/m,
                     (Fy - dω*pox + (ω**2)*poy)/m,
                     dω])
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
        g, ε = self.gravity, self.coupling
        φ, dφ = x[2], x[5]
        cφ, sφ = cs.cos(φ), cs.sin(φ)
        if self.body_fixed:
            v, w = x[3], x[4]
            f = cs.vcat([v*cφ - w*sφ,
                         v*sφ + w*cφ,
                         dφ,
                         w*dφ - g*sφ + ε*u[1],
                         -v*dφ - g*cφ + u[0],
                         u[1]])
        else:
            dx, dy = x[3], x[4]
            f = cs.vcat([dx,
                         dy,
                         dφ,
                         -sφ*u[0] + ε*cφ*u[1],
                         -g + cφ*u[0] + ε*sφ*u[1],
                         u[1]])
        return f


class ThreeLinkManipulator(Dynamics):
    """Dynamics of a three-link, open-chain manipulator."""

    gravity:    float = 9.81
    lengths:    Array = np.array([1., 1., 1.])
    masses:     Array = np.array([1., 1., 1.])
    inertias:   Array = np.array([1e-2, 1e-2, 1e-2])

    def __post_init__(self):
        """TODO."""
        n_dof = 3
        object.__setattr__(self, 'masses',
                           np.broadcast_to(self.masses, (n_dof,)))
        object.__setattr__(self, 'lengths',
                           np.broadcast_to(self.lengths, (n_dof,)))
        object.__setattr__(self, 'inertias',
                           np.broadcast_to(self.inertias, (n_dof,)))

    @property
    def dims(self) -> tuple[int, int]:
        """Return the dimensions of the state and input for this system."""
        return 6, 3

    def dynamics(self, x, u):
        """Evaluate the dynamics `dx/dt = f(x, u)`."""
        n_dof = 3
        g, L, m, J = self.gravity, self.lengths, self.masses, self.inertias
        r = L/2
        q, dq = x[:n_dof], x[n_dof:]
        c, s = cs.cos(q), cs.sin(q)
        c_12 = cs.cos(q[1] + q[2])
        s_12 = cs.sin(q[1] + q[2])

        # Mass matrix (inverse)
        M_00 = (J[0] + J[1] + J[2] + m[1]*(r[0]**2)*(c[1]**2)
                + m[2]*(L[0]*c[1] + r[1]*c_12)**2)
        M_11 = (J[1] + J[2] + m[1]*r[0]**2
                + m[2]*(L[0]**2 + r[1]**2 + 2*L[0]*r[1]*c[2]))
        M_12 = J[2] + m[2]*r[1]*(r[1] + L[0]*c[2])
        M_22 = J[2] + m[2]*r[1]**2
        det = M_11*M_22 - M_12**2
        M_inv = cs.vcat([
            cs.hcat([1/M_00, 0.,        0.]),
            cs.hcat([0.,     M_22/det,  -M_12/det]),
            cs.hcat([0.,     -M_12/det, M_11/det]),
        ])

        # Non-zero Christoffel symbols and Coriolis vector
        C_001 = -(m[1]*(r[0]**2)*c[1]*s[1]
                  + m[2]*(L[0]*c[1] + r[1]*c_12)*(L[0]*s[1] + r[1]*s_12))/2
        C_002 = -m[2]*r[1]*s_12*(L[0]*c[1] + r[1]*c_12)/2
        C_010 = C_001
        C_020 = C_002

        C_100 = -C_001
        C_112 = -L[0]*m[2]*r[1]*s[2]/2
        C_121 = C_112
        C_122 = C_112

        C_200 = -C_002
        C_211 = -C_112

        Cdq = cs.vcat([
            (C_001 + C_010)*dq[0]*dq[1] + (C_002 + C_020)*dq[0]*dq[2],
            C_100*dq[0]**2 + (C_112 + C_121)*dq[1]*dq[2] + C_122*dq[2]**2,
            C_200*dq[0]**2 + C_211*dq[1]**2,
        ])

        # Potential vector
        dV = -g * cs.vcat([
            cs.MX(1, 1),
            (m[1]*r[0] + m[2]*L[0])*c[1] + m[2]*r[1]*c_12,
            m[2]*r[1]*c_12,
        ])

        f = cs.vcat([
            dq,
            M_inv @ (u - Cdq - dV)
        ])
        return f
