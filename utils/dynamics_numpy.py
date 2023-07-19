"""
Equations of motion for various dynamical systems in NumPy.

Author: Spencer M. Richards
        Autonomous Systems Lab (ASL), Stanford
        (GitHub: spenrich)
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

import scipy
from scipy.integrate import odeint

# Type annotation for NumPy arrays
Array = ArrayLike


def wrap_to_pi(x):
    """Wrap an angle in radians to the interval (-pi, pi]."""
    return np.pi - np.mod(-x + np.pi, 2*np.pi)


def sinc(x):
    """Evaluate the unnormalized sinc function `sin(x)/x`."""
    return np.sinc(x/np.pi)


def cosc(x):
    """Evaluate the function `(cos(x) - 1)/x`.

    The name `cosc` is inspired by the function `sinc(x) := sin(x)/x`.
    """
    return -np.sin(x/2)*sinc(x/2)


def broadcast_to_square_matrix(A: ArrayLike, n: int) -> np.ndarray:
    """Broadcast argument `A` to a conformable square matrix."""
    A = np.asarray(A)
    if A.ndim > 2:
        raise ValueError('Argument `A` has too many dimensions.')
    elif A.ndim <= 1:
        A = np.broadcast_to(A, (n,))
        A = np.diag(A)
    return A


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
    def dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Compute the dynamics `dx/dt = f(x, u)`."""
        raise NotImplementedError()

    def __call__(self, x: np.ndarray, u: np.ndarray,
                 zoh: bool = False, dt: float | None = None) -> np.ndarray:
        """Evaluate this dynamics model at state `x` and control `u`."""
        x, u = np.atleast_1d(x), np.atleast_1d(u)
        if zoh:  # `y = x(t + dt)`
            if dt is None:
                raise ValueError('Argument `dt` must be a positive float for '
                                 'zero-order-hold control.')

            def ode(x, t, u):
                return self.dynamics(x, u)
            y = odeint(ode, x, np.array([0., dt]), (u,))[-1]
        else:  # `y = dx/dt`
            y = self.dynamics(x, u)
        return y


class ControlAffineDynamics(Dynamics):
    """A continuous-time dynamical system in control-affine form."""

    @abstractmethod
    def caf(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate the terms `(f, B)` for `dx/dt = f(x) + B(x)u`."""
        raise NotImplementedError()

    def dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Evaluate the dynamics `dx/dt = f(x, u)`."""
        f, B = self.caf(x)
        dx = f + B@u
        return dx


class SDCDynamics(ControlAffineDynamics):
    """Base class for dynamics in state-dependent coefficient (SDC) form."""

    @property
    @abstractmethod
    def equilibrium(self) -> tuple[np.ndarray, np.ndarray]:
        """Return an equilibrium pair `(x_bar, u_bar)` for this system."""
        raise NotImplementedError()

    @abstractmethod
    def sdc(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute an SDC form of the system dynamics at state `x`.

        An SDC form `(A, B)` is such that:
            `dx/dt = A(x - x_bar)@(x - x_bar) + B(x)@(u - u_bar)`,
        where `(x_bar, u_bar)` is an equilibrium pair for this system.
        """
        raise NotImplementedError()

    def caf(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate the terms `(f, B)` for `dx/dt = f(x) + B(x)u`."""
        x_bar, u_bar = self.equilibrium
        A, B = self.sdc(x)
        f = A@(x - x_bar) - B@u_bar
        return f, B

    def controller(self, x, Q=1., R=1., linearize=False):
        """Compute the control input at state `x`.

        This controller drives `(x, u)` to the equilibrium `(x_bar, u_bar)`.
        """
        n, m = self.dims
        x_bar, u_bar = self.equilibrium
        Q = broadcast_to_square_matrix(Q, n)
        R = broadcast_to_square_matrix(R, m)
        if linearize:
            A, B = self.sdc(x_bar)
        else:
            A, B = self.sdc(x)
        M = scipy.linalg.solve_continuous_are(A, B, Q, R)
        dV = M @ (x - x_bar)
        u = u_bar - scipy.linalg.solve(R, B.T @ dV, assume_a='pos')
        return u

    def tracking_sdc(self, e, x_bar, u_bar) -> tuple[np.ndarray, np.ndarray]:
        """Compute a tracking SDC form of the system dynamics."""
        raise NotImplementedError()

    def tracking_controller(self, x, x_bar, u_bar, Q=1., R=1.,
                            linearize=False):
        """Compute the tracking control input."""
        n, m = self.dims
        e = x - x_bar
        if linearize:
            A, B = self.tracking_sdc(np.zeros(n), x_bar, u_bar)
        else:
            A, B = self.tracking_sdc(e, x_bar, u_bar)
        Q = broadcast_to_square_matrix(Q, n)
        R = broadcast_to_square_matrix(R, m)
        M = scipy.linalg.solve_continuous_are(A, B, Q, R)
        dV = M @ e
        u = u_bar - scipy.linalg.solve(R, B.T @ dV, assume_a='pos')
        return u


###############################################################################
#                          DYNAMICAL SYSTEMS EXAMPLES                         #
###############################################################################


class Car(SDCDynamics):
    """Dynamics of a simple car."""

    @property
    def dims(self) -> tuple[int, int]:
        """Return the dimensions of the state and input for this system."""
        return 4, 2

    @property
    def equilibrium(self):
        """Return an equilibrium pair `(x_bar, u_bar)` for this system."""
        n, m = self.dims
        x_bar, u_bar = np.zeros(n), np.zeros(m)
        return x_bar, u_bar

    @classmethod
    def wrap_state(cls, x):
        """Adjust the state such that the angle lies in (-pi, pi]."""
        x[..., 3] = wrap_to_pi(x[..., 3])
        return x

    def caf(self, x: np.ndarray):
        """Evaluate the terms `(f, B)` for `dx/dt = f(x) + B(x)u`."""
        _, _, θ, v = x
        f = np.array([v*np.cos(θ), v*np.sin(θ), 0., 0.])
        B = np.array([[0., 0.],
                      [0., 0.],
                      [1., 0.],
                      [0., 1.]])
        return f, B

    def sdc(self, x):
        """Compute an SDC form of the system dynamics at state `x`.

        An SDC form `(A, B)` is such that:
            `dx/dt = A(x - x_bar)@(x - x_bar) + B(x)@(u - u_bar)`,
        where `(x_bar, u_bar)` is an equilibrium pair for this system.
        """
        _, _, θ, v = x
        A = np.array([[0., 0., 0.,        np.cos(θ)],
                      [0., 0., v*sinc(θ), 0.],
                      [0., 0., 0.,        0.],
                      [0., 0., 0.,        0.]])
        B = np.array([[0., 0.],
                      [0., 0.],
                      [1., 0.],
                      [0., 1.]])
        return A, B


class CartPole(SDCDynamics):
    """Dynamics of an inverted pendulum mounted on a cart."""

    gravity:        float = 9.81
    length:         float = 1.
    mass_cart:      float = 1.
    mass_pendulum:  float = 1.

    @property
    def dims(self) -> tuple[int, int]:
        """Return the dimensions of the state and input for this system."""
        return 4, 1

    @property
    def equilibrium(self):
        """Return an equilibrium pair `(x_bar, u_bar)` for this system."""
        n, m = self.dims
        x_bar, u_bar = np.zeros(n), np.zeros(m)
        return x_bar, u_bar

    @classmethod
    def wrap_state(cls, x):
        """Adjust the state such that the angle lies in (-pi, pi]."""
        x[..., 1] = wrap_to_pi(x[..., 1])
        return x

    def caf(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the terms `(f, B)` for `dx/dt = f(x) + B(x)u`."""
        g, L, mc, mp = (self.gravity, self.length,
                        self.mass_cart, self.mass_pendulum)
        _, φ, dx, dφ = x
        cφ, sφ = np.cos(φ), np.sin(φ)
        h = mc + mp*(sφ**2)
        f = np.array([dx,
                      dφ,
                      mp*sφ*(g*cφ - L*(dφ**2)) / h,
                      ((mc+mp)*g*sφ/L - mp*(dφ**2)*sφ*cφ) / h])
        B = np.array([[0.],
                      [0.],
                      [1/h],
                      [cφ/(L*h)]])
        return f, B

    def annihilator(self, x: np.ndarray) -> np.ndarray:
        """Evaluate an annihilator actuation matrix for the dynamics."""
        _, φ, _, _ = x
        L, mc, mp = self.length, self.mass_cart, self.mass_pendulum
        h = mc + mp*(np.sin(φ)**2)
        B_perp = np.array([[1., 0., 0.],
                           [0., 1., 0.],
                           [0., 0., np.cos(φ)/(L*h)],
                           [0., 0., -1/h]])
        return B_perp

    def sdc(self, x):
        """Compute an SDC form of the system dynamics at state `x`."""
        _, φ, _, dφ = x
        g, L, mc, mp = (self.gravity, self.length,
                        self.mass_cart, self.mass_pendulum)
        cφ, sφ = np.cos(φ), np.sin(φ)
        h = mc + mp*(sφ**2)
        A = np.array([
            [0., 0.,                      1., 0.],
            [0., 0.,                      0., 1.],
            [0., mp*g*cφ*sinc(φ)/h,       0., -mp*L*dφ*sφ/h],
            [0., (mc + mp)*g*sinc(φ)/L/h, 0., -mp*dφ*sφ*cφ/h],
        ])
        B = np.array([
            [0.],
            [0.],
            [1/h],
            [cφ/(L*h)]
        ])
        return A, B


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
        p1, v1, p2, v2 = x
        f = np.array([v1, u[0], v2, -β*v2 - k*(p2 - p1)])
        return f


class InvertedPendulum(ControlAffineDynamics):
    """Dynamics of an inverted pendulum."""

    gravity:    float = 9.81
    mass:       float = 1.
    length:     float = 1.

    @property
    def dims(self) -> tuple[int, int]:
        """Return the dimensions of the state and input for this system."""
        return 2, 1

    @classmethod
    def wrap_state(cls, x: np.ndarray) -> np.ndarray:
        """Adjust the state such that the angle lies in (-pi, pi]."""
        x = x.at[..., 0].set(wrap_to_pi(x[..., 0]))
        return x

    @property
    def equilibrium(self):
        """Return an equilibrium pair `(x_bar, u_bar)` for this system."""
        x_bar = np.zeros(self.state_dim)
        u_bar = np.zeros(self.control_dim)
        return x_bar, u_bar

    def caf(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the terms `(f, B)` for `dx/dt = f(x) + B(x)u`."""
        m, g, L = self.mass, self.gravity, self.length
        θ, dθ = x
        f = np.array([dθ, (g/L)*np.sin(θ)])
        B = np.array([[0.],
                      [1./(m*L**2)]])
        return f, B


class PlanarBirotor(SDCDynamics):
    """Dynamics of a planar bi-rotor vehicle."""

    gravity:    float = 9.81
    mass:       float = 0.486
    length:     float = 0.25
    inertia:    float = 0.00383

    @property
    def dims(self) -> tuple[int, int]:
        """Return the dimensions of the state and input for this system."""
        return 6, 2

    @property
    def equilibrium(self):
        """Return an equilibrium pair `(x_bar, u_bar)` for this system."""
        m, g = self.mass, self.gravity
        x_bar = np.zeros(self.dims[0])
        u_bar = np.array([m*g/2, m*g/2])
        return x_bar, u_bar

    @classmethod
    def wrap_state(cls, x):
        """Adjust the state such that the angle lies in (-pi, pi]."""
        x[..., 2] = wrap_to_pi(x[..., 2])
        return x

    def caf(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the terms `(f, B)` for `dx/dt = f(x) + B(x)u`."""
        g, m, L, J = self.gravity, self.mass, self.length, self.inertia
        _, _, φ, v, w, dφ = x
        cφ, sφ = np.cos(φ), np.sin(φ)
        f = np.array([v*cφ - w*sφ,
                      v*sφ + w*cφ,
                      dφ,
                      w*dφ - g*sφ,
                      -v*dφ - g*cφ,
                      0.])
        B = np.array([[0., 0.],
                      [0., 0.],
                      [0., 0.],
                      [0., 0.],
                      [1/m, 1/m],
                      [L/J, -L/J]])
        return f, B

    def sdc(self, x: np.ndarray):
        """Compute an SDC form of the system dynamics at state `x`."""
        g, m, L, J = self.gravity, self.mass, self.length, self.inertia
        φ = x[2]
        cφ, sφ = np.cos(φ), np.sin(φ)
        v, w = x[3], x[4]
        A = np.array([[0., 0., 0.,         cφ, -sφ, 0.],
                      [0., 0., 0.,         sφ, cφ,  0.],
                      [0., 0., 0.,         0., 0.,  1.],
                      [0., 0., -g*sinc(φ), 0., 0.,  w],
                      [0., 0., -g*cosc(φ), 0., 0.,  -v],
                      [0., 0., 0.,         0., 0.,  0.]])
        B = np.array([[0., 0.],
                      [0., 0.],
                      [0., 0.],
                      [0., 0.],
                      [1/m, 1/m],
                      [L/J, -L/J]])
        return A, B

    def tracking_sdc(self, e, x, u, slider=0.):
        """Compute a tracking SDC form of the system dynamics."""
        g, m, L, J = self.gravity, self.mass, self.length, self.inertia
        φ, dφ, eφ, edφ = x[2], x[5], e[2], e[5]
        cφ, sφ = np.cos(φ), np.sin(φ)
        v, w = x[3], x[4]
        ev, ew = e[3], e[4]

        A0 = np.array([
            [0., 0., (v*cφ - w*sφ)*cosc(eφ) - (v*sφ + w*cφ)*sinc(eφ), np.cos(φ + eφ), -np.sin(φ + eφ), 0.],         # noqa: E501
            [0., 0., (v*sφ + w*cφ)*cosc(eφ) + (v*cφ - w*sφ)*sinc(eφ), np.sin(φ + eφ), np.cos(φ + eφ),  0.],         # noqa: E501
            [0., 0., 0.,                                              0.,             0.,              1.],         # noqa: E501
            [0., 0., -g*(sφ*cosc(eφ) + cφ*sinc(eφ)),                  0.,             dφ,              w + ew],     # noqa: E501
            [0., 0., -g*(cφ*cosc(eφ) - sφ*sinc(eφ)),                  -dφ,            0.,              -(v + ev)],  # noqa: E501
            [0., 0., 0.,                                              0.,             0.,              0.],         # noqa: E501
        ])
        A1 = np.array([
            [0., 0., (v*cφ - w*sφ)*cosc(eφ) - ((v+ev)*sφ + (w+ew)*cφ)*sinc(eφ), cφ*np.cos(eφ), -sφ*np.cos(eφ), 0.],  # noqa: E501
            [0., 0., (v*sφ + w*cφ)*cosc(eφ) + ((v+ev)*cφ - (w+ew)*sφ)*sinc(eφ), sφ*np.cos(eφ), cφ*np.cos(eφ),  0.],  # noqa: E501
            [0., 0., 0.,                                                        0.,            0.,             1.],  # noqa: E501
            [0., 0., -g*(sφ*cosc(eφ) + cφ*sinc(eφ)),                            0.,            dφ + edφ,       w],   # noqa: E501
            [0., 0., -g*(cφ*cosc(eφ) - sφ*sinc(eφ)),                            -(dφ + edφ),   0.,             -v],  # noqa: E501
            [0., 0., 0.,                                                        0.,            0.,             0.],  # noqa: E501
        ])
        A = A0 + slider*(A1 - A0)

        B = np.array([[0., 0.],
                      [0., 0.],
                      [0., 0.],
                      [0., 0.],
                      [1/m, 1/m],
                      [L/J, -L/J]])
        return A, B

    def tracking_controller(self, x, x_bar, u_bar, Q=1., R=1., linearize=False,
                            slider=0.):
        """Compute the tracking control input."""
        n, m = self.dims
        e = x - x_bar
        if linearize:
            A, B = self.tracking_sdc(np.zeros(n), x_bar, u_bar, 0.)
        else:
            A, B = self.tracking_sdc(e, x_bar, u_bar, slider)
        Q = broadcast_to_square_matrix(Q, n)
        R = broadcast_to_square_matrix(R, m)
        M = scipy.linalg.solve_continuous_are(A, B, Q, R)
        dV = M @ e
        u = u_bar - scipy.linalg.solve(R, B.T @ dV, assume_a='pos')
        return u


class PlanarSpacecraft(SDCDynamics):
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

    @classmethod
    def wrap_state(cls, x: np.ndarray) -> np.ndarray:
        """Adjust the state such that the angle lies in (-pi, pi]."""
        x = x.at[..., 2].set(wrap_to_pi(x[..., 2]))
        return x

    @property
    def equilibrium(self):
        """Return an equilibrium pair `(x_bar, u_bar)` for this system."""
        x_bar = np.zeros(self.state_dim)
        u_bar = np.zeros(self.control_dim)
        return x_bar, u_bar

    def caf(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the terms `(f, B)` for `dx/dt = f(x) + B(x)u`."""
        m, J = self.mass, self.inertia
        dx, dy = self.offset
        _, _, _, vx, vy, ω = x
        f = np.array([vx, vy, ω, (ω**2)*dx/m, (ω**2)*dy/m, 0.])
        B = np.array([[0.,                0.,              0.],
                      [0.,                0.,              0.],
                      [0.,                0.,              0.],
                      [(1 + (dy**2)/J)/m, -dx*dy/(m*J),    dy/(m*J)],
                      [-dx*dy/(m*J),      (1 + dx**2/J)/m, -dx/(m*J)],
                      [dy/J,              -dx/J,           1./J]])
        return f, B

    def sdc(self, x: np.ndarray):
        """Compute an SDC form of the system dynamics at state `x`."""
        m, J = self.mass, self.inertia
        dx, dy = self.offset
        ω = x[5]
        A = np.array([[0., 0., 0., 1., 0., 0.],
                      [0., 0., 0., 0., 1., 0.],
                      [0., 0., 0., 0., 0., 1.],
                      [0., 0., 0., 0., 0., 2*dx*ω],
                      [0., 0., 0., 0., 0., 2*dy*ω],
                      [0., 0., 0., 0., 0., 0.]]) / m
        B = np.array([[0.,                0.,              0.],
                      [0.,                0.,              0.],
                      [0.,                0.,              0.],
                      [(1 + (dy**2)/J)/m, -dx*dy/(m*J),    dy/(m*J)],
                      [-dx*dy/(m*J),      (1 + dx**2/J)/m, -dx/(m*J)],
                      [dy/J,              -dx/J,           1./J]])
        return A, B

    def tracking_sdc(self, e, x, u):
        """Compute a tracking SDC form of the system dynamics."""
        m, J = self.mass, self.inertia
        dx, dy = self.offset
        ω, eω = x[5], e[5]
        A = np.array([[0., 0., 0., 1., 0., 0.],
                      [0., 0., 0., 0., 1., 0.],
                      [0., 0., 0., 0., 0., 1.],
                      [0., 0., 0., 0., 0., dx*(2*ω + eω)],
                      [0., 0., 0., 0., 0., dy*(2*ω + eω)],
                      [0., 0., 0., 0., 0., 0.]]) / m
        B = np.array([[0.,                0.,              0.],
                      [0.,                0.,              0.],
                      [0.,                0.,              0.],
                      [(1 + (dy**2)/J)/m, -dx*dy/(m*J),    dy/(m*J)],
                      [-dx*dy/(m*J),      (1 + dx**2/J)/m, -dx/(m*J)],
                      [dy/J,              -dx/J,           1./J]])
        return A, B


class PVTOL(SDCDynamics):
    """Dynamics of a planar vertical take-off and landing (PVTOL) vehicle."""

    gravity:    float = 9.81
    coupling:   float = 1.
    body_fixed: bool = True

    @property
    def dims(self) -> tuple[int, int]:
        """Return the dimensions of the state and input for this system."""
        return 6, 2

    @property
    def equilibrium(self):
        """Return an equilibrium pair `(x_bar, u_bar)` for this system."""
        x_bar = np.zeros(self.state_dim)
        u_bar = np.array([self.gravity, 0.])
        return x_bar, u_bar

    @classmethod
    def wrap_state(cls, x):
        """Adjust the state such that the angle lies in (-pi, pi]."""
        x[..., 2] = wrap_to_pi(x[..., 2])
        return x

    def sdc(self, x: np.ndarray):
        """Compute an SDC form of the system dynamics at state `x`.

        An SDC form `(A, B)` is such that:
            `dx/dt = A(x - x_bar)@(x - x_bar) + B(x)@(u - u_bar)`,
        where `(x_bar, u_bar)` is an equilibrium pair for this system.
        """
        g, ε = self.gravity, self.coupling
        φ = x[2]
        cφ, sφ = np.cos(φ), np.sin(φ)
        if self.body_fixed:
            v, w = x[3], x[4]
            A = np.array([[0., 0., 0.,         cφ, -sφ, 0.],
                          [0., 0., 0.,         sφ, cφ,  0.],
                          [0., 0., 0.,         0., 0.,  1.],
                          [0., 0., -g*sinc(φ), 0., 0.,  w],
                          [0., 0., -g*cosc(φ), 0., 0.,  -v],
                          [0., 0., 0.,         0., 0.,  0.]])
            B = np.array([[0., 0.],
                          [0., 0.],
                          [0., 0.],
                          [0., ε],
                          [1., 0.],
                          [0., 1.]])
        else:
            A = np.array([[0., 0., 0.,         1., 0., 0.],
                          [0., 0., 0.,         0., 1., 0.],
                          [0., 0., 0.,         0., 0., 1.],
                          [0., 0., -g*sinc(φ), 0., 0., 0.],
                          [0., 0., g*cosc(φ),  0., 0., 0.],
                          [0., 0., 0.,         0., 0., 0.]])
            B = np.array([[0.,  0.],
                          [0.,  0.],
                          [0.,  0.],
                          [-sφ, ε*cφ],
                          [cφ,  ε*sφ],
                          [0.,  1.]])
        return A, B

    def caf(self, x: Array) -> tuple[Array, Array]:
        """Evaluate the terms `(f, B)` for `dx/dt = f(x) + B(x)u`."""
        g = self.gravity
        ε = self.coupling
        if self.body_fixed:
            _, _, φ, v, w, dφ = x
            cφ, sφ = np.cos(φ), np.sin(φ)
            f = np.array([v*cφ - w*sφ,
                          v*sφ + w*cφ,
                          dφ,
                          w*dφ - g*sφ,
                          -v*dφ - g*cφ,
                          0.])
            B = np.array([[0., 0.],
                          [0., 0.],
                          [0., 0.],
                          [0., ε],
                          [1., 0.],
                          [0., 1.]])
        else:
            _, _, φ, dx, dy, dφ = x
            cφ, sφ = np.cos(φ), np.sin(φ)
            f = np.array([dx, dy, dφ, 0., -g, 0.])
            B = np.array([[0., 0.],
                          [0., 0.],
                          [0., 0.],
                          [-sφ, ε*cφ],
                          [cφ,  ε*sφ],
                          [0.,  1.]])
        return f, B

    def tracking_sdc(self, e, x, u):
        """Compute a tracking SDC form of the system dynamics."""
        φ, dφ, eφ = x[2], x[5], e[2]
        g = self.gravity
        ε = self.coupling
        cφ, sφ = np.cos(φ), np.sin(φ)
        if self.body_fixed:
            v, w = x[3], x[4]
            ev, ew = e[3], e[4]
            A = np.zeros((self.dims[0], self.dims[0]))

            # `x`-error state derivative
            A[0, 2] = (v*cφ - w*sφ)*cosc(eφ) - (v*sφ + w*cφ)*sinc(eφ)
            A[0, 3] = np.cos(φ + eφ)
            A[0, 4] = -np.sin(φ + eφ)

            # `y`-error state derivative
            A[1, 2] = (v*sφ + w*cφ)*cosc(eφ) + (v*cφ - w*sφ)*sinc(eφ)
            A[1, 3] = np.sin(φ + eφ)
            A[1, 4] = np.cos(φ + eφ)

            # `phi`-error state derivative
            A[2, 5] = 1.

            # `v`-error state derivative
            A[3, 2] = -g*(sφ*cosc(eφ) + cφ*sinc(eφ))
            A[3, 4] = dφ
            A[3, 5] = w + ew

            # `w`-error state derivative
            A[4, 2] = -g*(cφ*cosc(eφ) - sφ*sinc(eφ))
            A[4, 3] = -dφ
            A[4, 5] = -(v + ev)

            B = np.array([[0., 0.],
                          [0., 0.],
                          [0., 0.],
                          [0., ε],
                          [1., 0.],
                          [0., 1.]])
        else:
            raise NotImplementedError()
        return A, B


class ThreeLinkManipulator(ControlAffineDynamics):
    """Dynamics of a three-link, open-chain manipulator."""

    gravity:    float = 9.81
    lengths:    np.ndarray = np.array([1., 1., 1.])
    masses:     np.ndarray = np.array([1., 1., 1.])
    inertias:   np.ndarray = np.array([1e-2, 1e-2, 1e-2])

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

    @property
    def equilibrium(self):
        """Return an equilibrium pair `(x_bar, u_bar)` for this system."""
        x_bar = np.zeros(self.state_dim)
        u_bar = np.zeros(self.control_dim)
        return x_bar, u_bar

    def caf(self, x: Array) -> tuple[Array, Array]:
        """Evaluate the terms `(f, B)` for `dx/dt = f(x) + B(x)u`."""
        n_dof = 3
        g, L, m, J = self.gravity, self.lengths, self.masses, self.inertias
        r = L/2
        q, dq = x[:n_dof], x[n_dof:]
        c, s = np.cos(q), np.sin(q)
        c_12 = np.cos(q[1] + q[2])
        s_12 = np.sin(q[1] + q[2])

        # Mass matrix (inverse)
        M_00 = (J[0] + J[1] + J[2] + m[1]*(r[0]**2)*(c[1]**2)
                + m[2]*(L[0]*c[1] + r[1]*c_12)**2)
        M_11 = (J[1] + J[2] + m[1]*r[0]**2
                + m[2]*(L[0]**2 + r[1]**2 + 2*L[0]*r[1]*c[2]))
        M_12 = J[2] + m[2]*r[1]*(r[1] + L[0]*c[2])
        M_22 = J[2] + m[2]*r[1]**2
        # M = np.array([
        #     [M_00, 0.,   0.],
        #     [0.,   M_11, M_12],
        #     [0.,   M_12, M_22],
        # ])
        det = M_11*M_22 - M_12**2
        M_inv = np.array([
            [1/M_00, 0.,        0.],
            [0.,     M_22/det,  -M_12/det],
            [0.,     -M_12/det, M_11/det],
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

        Cdq = np.array([
            (C_001 + C_010)*dq[0]*dq[1] + (C_002 + C_020)*dq[0]*dq[2],
            C_100*dq[0]**2 + (C_112 + C_121)*dq[1]*dq[2] + C_122*dq[2]**2,
            C_200*dq[0]**2 + C_211*dq[1]**2,
        ])

        # Potential vector
        dV = -g * np.array([
            0.,
            (m[1]*r[0] + m[2]*L[0])*c[1] + m[2]*r[1]*c_12,
            m[2]*r[1]*c_12,
        ])

        f = np.concatenate([dq, -M_inv @ (Cdq + dV)])
        B = np.vstack([np.zeros((n_dof, n_dof)),  M_inv])
        return f, B
