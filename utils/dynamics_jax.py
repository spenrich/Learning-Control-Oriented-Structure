"""
Equations of motion for various dynamical systems in JAX.

Author: Spencer M. Richards
        Autonomous Systems Lab (ASL), Stanford
        (GitHub: spenrich)
"""
from abc import ABC, abstractmethod

import equinox as eqx

import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint

from jaxtyping import Array

import scipy

from .misc import broadcast_to_square_matrix


def wrap_to_pi(x):
    """Wrap an angle in radians to the interval (-pi, pi]."""
    return jnp.pi - jnp.mod(-x + jnp.pi, 2*jnp.pi)


def sinc(x):
    """Evaluate the unnormalized sinc function `sin(x)/x`."""
    return jnp.sinc(x/jnp.pi)


def cosc(x):
    """Evaluate the function `(cos(x) - 1)/x`.

    The name `cosc` is inspired by the function `sinc(x) := sin(x)/x`.
    """
    return -jnp.sin(x/2)*sinc(x/2)


def solve_care(A, B, Q, R):
    """Solve the continuous-time algebraic Riccati equation (CARE).

    This function is based on the JAX documentation at:
        https://jax.readthedocs.io/en/latest/notebooks/external_callbacks.html

    Since this is pure callback using a SciPy function, data will always be
    transferred to the host (i.e., the CPU) before doing this computation.
    """
    A, B, Q, R = jnp.asarray(A), jnp.asarray(B), jnp.asarray(Q), jnp.asarray(R)

    # Promote the inputs to be inexact (float/complex)
    # NOTE: `jnp.result_type()` accounts for the `enable_x64` flag
    A = A.astype(jnp.result_type(float, A.dtype))
    B = B.astype(jnp.result_type(float, B.dtype))
    Q = Q.astype(jnp.result_type(float, Q.dtype))
    R = R.astype(jnp.result_type(float, R.dtype))

    # Wrap scipy function to return the expected dtype
    def _scipy_solve_care(A, B, Q, R):
        return scipy.linalg.solve_continuous_are(A, B, Q, R).astype(A.dtype)

    # Define the expected shape and dtype of the output
    n = A.shape[-1]
    result_shape_dtype = jax.ShapeDtypeStruct((n, n), A.dtype)

    # Create and call a pure callback version of the external function
    # NOTE: We set `vectorized=False` since `scipy.linalg.solve_continuous_are`
    #       is not vectorized
    M = jax.pure_callback(_scipy_solve_care, result_shape_dtype,
                          A, B, Q, R, vectorized=False)
    return M


class Dynamics(eqx.Module, ABC):
    """Base class for autonomous continuous-time dynamical systems."""

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
    def dynamics(self, x: Array, u: Array) -> Array:
        """Compute the dynamics `dx/dt = f(x, u)`."""
        raise NotImplementedError()

    def __call__(self, x: Array, u: Array,
                 zoh: bool = False, dt: float = 0.1) -> Array:
        """Evaluate this dynamics model at state `x` and control `u`."""
        x, u = jnp.atleast_1d(x), jnp.atleast_1d(u)
        if zoh:  # `y = x(t + dt)`
            def ode(x, t, u):
                return self.dynamics(x, u)
            y = odeint(ode, x, jnp.array([0., dt]), u)[-1]
        else:  # `y = dx/dt`
            y = self.dynamics(x, u)
        return y


class ControlAffineDynamics(Dynamics):
    """A continuous-time dynamical system in control-affine form."""

    @abstractmethod
    def caf(self, x: Array) -> tuple[Array, Array]:
        """Evaluate the terms `(f, B)` for `dx/dt = f(x) + B(x)u`."""
        raise NotImplementedError()

    def dynamics(self, x: Array, u: Array) -> Array:
        """Evaluate the dynamics `dx/dt = f(x, u)`."""
        f, B = self.caf(x)
        dx = f + B@u
        return dx

    def tracking_controller(self, x, x_bar, u_bar, Q=1., R=1.):
        """Compute the tracking control input."""
        n, m = self.dims
        e = x - x_bar
        A = jax.jacfwd(self.dynamics, argnums=0)(x_bar, u_bar)
        _, B = self.caf(x_bar)
        Q = broadcast_to_square_matrix(Q, n)
        R = broadcast_to_square_matrix(R, m)
        M = solve_care(A, B, Q, R)
        dV = M @ e
        u = u_bar - jax.scipy.linalg.solve(R, B.T @ dV, assume_a='pos')
        return u


class SDCDynamics(ControlAffineDynamics):
    """Base class for dynamics in state-dependent coefficient (SDC) form."""

    @property
    @abstractmethod
    def equilibrium(self) -> tuple[Array, Array]:
        """Return an equilibrium pair `(x_bar, u_bar)` for this system."""
        raise NotImplementedError()

    @abstractmethod
    def sdc(self, x) -> tuple[Array, Array]:
        """Compute an SDC form of the system dynamics at state `x`.

        An SDC form `(A, B)` is such that:
            `dx/dt = A(x - x_bar)@(x - x_bar) + B(x)@(u - u_bar)`,
        where `(x_bar, u_bar)` is the given equilibrium pair for this system.
        """
        raise NotImplementedError()

    def caf(self, x: Array) -> tuple[Array, Array]:
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
        M = solve_care(A, B, Q, R)
        dV = M @ (x - x_bar)
        u = u_bar - jax.scipy.linalg.solve(R, B.T @ dV, assume_a='pos')
        return u

    def tracking_sdc(self, e, x_bar, u_bar) -> tuple[Array, Array]:
        """Compute a tracking SDC form of the system dynamics."""
        raise NotImplementedError()

    def tracking_controller(self, x, x_bar, u_bar, Q=1., R=1.,
                            linearize=False):
        """Compute the tracking control input."""
        n, m = self.dims
        e = x - x_bar
        if linearize:
            try:
                A, B = self.tracking_sdc(jnp.zeros(n), x_bar, u_bar)
            except NotImplementedError:
                A = jax.jacfwd(self.dynamics, argnums=0)(x_bar, u_bar)
                _, B = self.caf(x_bar)
        else:
            A, B = self.tracking_sdc(e, x_bar, u_bar)
        Q = broadcast_to_square_matrix(Q, n)
        R = broadcast_to_square_matrix(R, m)
        M = solve_care(A, B, Q, R)
        dV = M @ e
        u = u_bar - jax.scipy.linalg.solve(R, B.T @ dV, assume_a='pos')
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

    @classmethod
    def wrap_state(cls, x):
        """Adjust the state such that the angle lies in (-pi, pi]."""
        x = x.at[..., 3].set(wrap_to_pi(x[..., 3]))
        return x

    @property
    def equilibrium(self):
        """Return an equilibrium pair `(x_bar, u_bar)` for this system."""
        x_bar = jnp.zeros(self.state_dim)
        u_bar = jnp.zeros(self.control_dim)
        return x_bar, u_bar

    def caf(self, x: Array):
        """Evaluate the terms `(f, B)` for `dx/dt = f(x) + B(x)u`."""
        _, _, θ, v = x
        f = jnp.array([v*jnp.cos(θ), v*jnp.sin(θ), 0., 0.])
        B = jnp.array([[0., 0.],
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
        A = jnp.array([[0., 0., 0.,        jnp.cos(θ)],
                       [0., 0., v*sinc(θ), 0.],
                       [0., 0., 0.,        0.],
                       [0., 0., 0.,        0.]])
        B = jnp.array([[0., 0.],
                       [0., 0.],
                       [1., 0.],
                       [0., 1.]])
        return A, B


class CartPole(SDCDynamics):
    """Dynamics of an inverted pendulum mounted on a cart."""

    gravity:        float = eqx.static_field(default=9.81)
    length:         float = eqx.static_field(default=1.)
    mass_cart:      float = eqx.static_field(default=1.)
    mass_pendulum:  float = eqx.static_field(default=1.)

    @property
    def dims(self) -> tuple[int, int]:
        """Return the dimensions of the state and input for this system."""
        return 4, 1

    @classmethod
    def wrap_state(cls, x):
        """Adjust the state such that the angle lies in (-pi, pi]."""
        x = x.at[..., 1].set(wrap_to_pi(x[..., 1]))
        return x

    @property
    def equilibrium(self):
        """Return an equilibrium pair `(x_bar, u_bar)` for this system."""
        n, m = self.dims
        x_bar = jnp.zeros(n)
        u_bar = jnp.zeros(m)
        return x_bar, u_bar

    def annihilator(self, x: Array) -> Array:
        """Evaluate an annihilator actuation matrix for the dynamics."""
        φ = x[1]
        L, mc, mp = self.length, self.mass_cart, self.mass_pendulum
        h = mc + mp*(jnp.sin(φ)**2)
        B_perp = jnp.array([[1., 0., 0.],
                            [0., 1., 0.],
                            [0., 0., jnp.cos(φ)/(L*h)],
                            [0., 0., -1/h]])
        return B_perp

    def sdc(self, x):
        """Compute an SDC form of the system dynamics at state `x`."""
        φ, dφ = x[1], x[3]
        g, L = self.gravity, self.length
        mc, mp = self.mass_cart, self.mass_pendulum
        h = mc + mp*(jnp.sin(φ)**2)

        A = jnp.array([
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
            [0., mp*g*jnp.cos(φ)*sinc(φ)/h, 0., -mp*L*dφ*jnp.sin(φ)/h],
            [0., (mc + mp)*g*sinc(φ)/L/h, 0., -mp*dφ*jnp.sin(φ)*jnp.cos(φ)/h],
        ])
        B = jnp.array([
            [0.],
            [0.],
            [1/h],
            [jnp.cos(φ)/(L*h)]
        ])
        return A, B


class Crane(Dynamics):
    """Dynamics of a simple crane."""

    damping:    float = eqx.static_field(default=0.001)
    spring:     float = eqx.static_field(default=0.9)

    @property
    def dims(self) -> tuple[int, int]:
        """Return the dimensions of the state and input for this system."""
        return 4, 1

    def dynamics(self, x, u):
        """Evaluate the dynamics `dx/dt = f(x, u)`."""
        β, k = self.damping, self.spring
        p1, v1, p2, v2 = x
        f = jnp.array([v1, u[0], v2, -β*v2 - k*(p2 - p1)])
        return f


class FreemanKokotovic(SDCDynamics):
    """Dynamics of the scalar system in (Freeman, Kokotovic; 1996).

    This system has a non-convex value function for any quadratic cost.
    """

    @property
    def dims(self) -> tuple[int, int]:
        """Return the dimensions of the state and input for this system."""
        return 1, 1

    def sdc(self, x):
        """Compute an SDC form of the system dynamics at state `x`."""
        # dx/dt = (1 - x^2)*x + u
        # l(x,u) = (q*x^2 + r*u^2)/2
        A = jnp.atleast_2d(1. - x**2)
        B = jnp.atleast_2d(1.)
        return A, B

    def grad_lyapunov(self, x, Q=1., R=1.):
        """TODO."""
        A, B = self.sdc(x)
        Rb = R/(B**2)
        dV = jnp.squeeze(Rb*(A + jnp.sqrt(A**2 + Q/Rb))*x)
        return dV

    def lyapunov(self, x, Q=1., R=1.):
        """TODO."""

        def h(x, c=Q/R):
            g1 = x**2 - 1
            g2 = jnp.sqrt(g1**2 + c)
            h = g1*g2 + c*jnp.arctanh(g1/g2)
            return h

        V = jnp.squeeze(R*(h(x) - h(0) - x**4 + 2*(x**2))/4)
        return V

    def controller(self, x, Q=1., R=1.):
        """Compute the control input at state `x`."""
        _, B = self.sdc(x)
        dV = self.grad_lyapunov(x, Q, R)
        u = jnp.reshape(-(B/R)*dV, (self.control_dim,))
        return u


class InvertedPendulum(ControlAffineDynamics):
    """Dynamics of an inverted pendulum."""

    gravity:    float = eqx.static_field(default=9.81)
    mass:       float = eqx.static_field(default=1.)
    length:     float = eqx.static_field(default=1.)

    @property
    def dims(self) -> tuple[int, int]:
        """Return the dimensions of the state and input for this system."""
        return 2, 1

    @classmethod
    def wrap_state(cls, x: Array) -> Array:
        """Adjust the state such that the angle lies in (-pi, pi]."""
        x = x.at[..., 0].set(wrap_to_pi(x[..., 0]))
        return x

    @property
    def equilibrium(self):
        """Return an equilibrium pair `(x_bar, u_bar)` for this system."""
        x_bar = jnp.zeros(self.state_dim)
        u_bar = jnp.zeros(self.control_dim)
        return x_bar, u_bar

    def caf(self, x: Array) -> Array:
        """Evaluate the terms `(f, B)` for `dx/dt = f(x) + B(x)u`."""
        m, g, L = self.mass, self.gravity, self.length
        θ, dθ = x
        f = jnp.array([dθ, (g/L)*jnp.sin(θ)])
        B = jnp.array([[0.],
                       [1./(m*L**2)]])
        return f, B


class PlanarBirotor(SDCDynamics):
    """Dynamics of a planar birotor."""

    gravity:    float = eqx.static_field(default=9.81)
    mass:       float = eqx.static_field(default=0.486)
    length:     float = eqx.static_field(default=0.25)
    inertia:    float = eqx.static_field(default=0.00383)

    @property
    def dims(self) -> tuple[int, int]:
        """Return the dimensions of the state and input for this system."""
        return 6, 2

    @classmethod
    def wrap_state(cls, x: Array) -> Array:
        """Adjust the state such that the angle lies in (-pi, pi]."""
        x = x.at[..., 2].set(wrap_to_pi(x[..., 2]))
        return x

    @property
    def equilibrium(self):
        """Return an equilibrium pair `(x_bar, u_bar)` for this system."""
        m, g = self.mass, self.gravity
        x_bar = jnp.zeros(self.state_dim)
        u_bar = jnp.array([m*g/2, m*g/2])
        return x_bar, u_bar

    def caf(self, x: Array) -> Array:
        """Evaluate the terms `(f, B)` for `dx/dt = f(x) + B(x)u`."""
        g, m, L, J = self.gravity, self.mass, self.length, self.inertia
        _, _, φ, v, w, dφ = x
        cφ, sφ = jnp.cos(φ), jnp.sin(φ)
        f = jnp.array([v*cφ - w*sφ,
                       v*sφ + w*cφ,
                       dφ,
                       w*dφ - g*sφ,
                       -v*dφ - g*cφ,
                       0.])
        B = jnp.array([[0., 0.],
                       [0., 0.],
                       [0., 0.],
                       [0., 0.],
                       [1/m, 1/m],
                       [L/J, -L/J]])
        return f, B

    def annihilator(self, x: Array) -> Array:
        """Evaluate an annihilator actuation matrix for the dynamics."""
        n, m = self.state_dim, self.control_dim
        B_perp = jnp.vstack([jnp.eye(n - m),
                             jnp.zeros((m, n - m))])
        return B_perp

    def sdc(self, x: Array):
        """Compute an SDC form of the system dynamics at state `x`.

        An SDC form `(A, B)` is such that:
            `dx/dt = A(x - x_bar)@(x - x_bar) + B(x)@(u - u_bar)`,
        where `(x_bar, u_bar)` is an equilibrium pair for this system.
        """
        g, m, L, J = self.gravity, self.mass, self.length, self.inertia
        φ = x[2]
        cφ, sφ = jnp.cos(φ), jnp.sin(φ)
        v, w = x[3], x[4]
        A = jnp.array([[0., 0., 0.,         cφ, -sφ, 0.],
                       [0., 0., 0.,         sφ, cφ,  0.],
                       [0., 0., 0.,         0., 0.,  1.],
                       [0., 0., -g*sinc(φ), 0., 0.,  w],
                       [0., 0., -g*cosc(φ), 0., 0.,  -v],
                       [0., 0., 0.,         0., 0.,  0.]])
        B = jnp.array([[0., 0.],
                       [0., 0.],
                       [0., 0.],
                       [0., 0.],
                       [1/m, 1/m],
                       [L/J, -L/J]])
        return A, B

    def tracking_sdc(self, e, x, u):
        """Compute a tracking SDC form of the system dynamics."""
        φ, dφ, eφ = x[2], x[5], e[2]
        g, m, L, J = self.gravity, self.mass, self.length, self.inertia
        cφ, sφ = jnp.cos(φ), jnp.sin(φ)
        v, w = x[3], x[4]
        ev, ew = e[3], e[4]
        A = jnp.zeros((self.state_dim, self.state_dim))

        # `x`-error state derivative
        A = A.at[0, 2].set((v*cφ - w*sφ)*cosc(eφ) - (v*sφ + w*cφ)*sinc(eφ))
        A = A.at[0, 3].set(jnp.cos(φ + eφ))
        A = A.at[0, 4].set(-jnp.sin(φ + eφ))

        # `y`-error state derivative
        A = A.at[1, 2].set((v*sφ + w*cφ)*cosc(eφ) + (v*cφ - w*sφ)*sinc(eφ))
        A = A.at[1, 3].set(jnp.sin(φ + eφ))
        A = A.at[1, 4].set(jnp.cos(φ + eφ))

        # `phi`-error state derivative
        A = A.at[2, 5].set(1.)

        # `v`-error state derivative
        A = A.at[3, 2].set(-g*(sφ*cosc(eφ) + cφ*sinc(eφ)))
        A = A.at[3, 4].set(dφ)
        A = A.at[3, 5].set(w + ew)

        # `w`-error state derivative
        A = A.at[4, 2].set(-g*(cφ*cosc(eφ) - sφ*sinc(eφ)))
        A = A.at[4, 3].set(-dφ)
        A = A.at[4, 5].set(-(v + ev))

        B = jnp.array([[0., 0.],
                       [0., 0.],
                       [0., 0.],
                       [0., 0.],
                       [1/m, 1/m],
                       [L/J, -L/J]])
        return A, B


class PlanarSpacecraft(ControlAffineDynamics):
    """Dynamics of a planar spacecraft with an offset center-of-mass."""

    mass:       float = eqx.static_field(default=30.)
    inertia:    float = eqx.static_field(default=0.5)
    offset:     Array = eqx.static_field(default=jnp.array([0.075, 0.075]))

    def __post_init__(self):
        """TODO."""
        object.__setattr__(self, 'offset', jnp.broadcast_to(self.offset, (2,)))

    @property
    def dims(self) -> tuple[int, int]:
        """Return the dimensions of the state and input for this system."""
        return 6, 3

    @classmethod
    def wrap_state(cls, x: Array) -> Array:
        """Adjust the state such that the angle lies in (-pi, pi]."""
        x = x.at[..., 2].set(wrap_to_pi(x[..., 2]))
        return x

    @property
    def equilibrium(self):
        """Return an equilibrium pair `(x_bar, u_bar)` for this system."""
        x_bar = jnp.zeros(self.state_dim)
        u_bar = jnp.zeros(self.control_dim)
        return x_bar, u_bar

    def caf(self, x: Array) -> Array:
        """Evaluate the terms `(f, B)` for `dx/dt = f(x) + B(x)u`."""
        m, J = self.mass, self.inertia
        pox, poy = self.offset
        _, _, _, vx, vy, ω = x
        f = jnp.array([vx, vy, ω, (ω**2)*pox/m, (ω**2)*poy/m, 0.])
        B = jnp.array([[0.,                 0.,               0.],
                       [0.,                 0.,               0.],
                       [0.,                 0.,               0.],
                       [(1 + (poy**2)/J)/m, -pox*poy/(m*J),   poy/(m*J)],
                       [-pox*poy/(m*J),     (1 + pox**2/J)/m, -pox/(m*J)],
                       [poy/J,              -pox/J,           1./J]])
        return f, B


class PVTOL(SDCDynamics):
    """Dynamics of a planar vertical take-off and landing (PVTOL) vehicle."""

    gravity:    float = eqx.static_field(default=9.81)
    coupling:   float = eqx.static_field(default=1.)
    body_fixed: bool = eqx.static_field(default=True)

    @property
    def dims(self) -> tuple[int, int]:
        """Return the dimensions of the state and input for this system."""
        return 6, 2

    @classmethod
    def wrap_state(cls, x):
        """Adjust the state such that the angle lies in (-pi, pi]."""
        x = x.at[..., 2].set(wrap_to_pi(x[..., 2]))
        return x

    @property
    def equilibrium(self):
        """Return an equilibrium pair `(x_bar, u_bar)` for this system."""
        x_bar = jnp.zeros(self.state_dim)
        u_bar = jnp.array([self.gravity, 0.])
        return x_bar, u_bar

    def caf(self, x: Array):
        """Evaluate the terms `(f, B)` for `dx/dt = f(x) + B(x)u`."""
        g, ε = self.gravity, self.coupling
        if self.body_fixed:
            _, _, φ, v, w, dφ = x
            cφ, sφ = jnp.cos(φ), jnp.sin(φ)
            f = jnp.array([v*cφ - w*sφ,
                           v*sφ + w*cφ,
                           dφ,
                           w*dφ - g*sφ,
                           -v*dφ - g*cφ,
                           0.])
            B = jnp.array([[0., 0.],
                           [0., 0.],
                           [0., 0.],
                           [0., ε],
                           [1., 0.],
                           [0., 1.]])
        else:
            _, _, φ, dx, dy, dφ = x
            cφ, sφ = jnp.cos(φ), jnp.sin(φ)
            f = jnp.array([dx, dy, dφ, 0., -g, 0.])
            B = jnp.array([[0., 0.],
                           [0., 0.],
                           [0., 0.],
                           [-sφ, ε*cφ],
                           [cφ,  ε*sφ],
                           [0.,  1.]])
        return f, B

    def annihilator(self, x: Array) -> Array:
        """Evaluate an annihilator actuation matrix for the dynamics."""
        d = 3   # `n // 2`
        B_perp = jnp.vstack([jnp.eye(d),
                             jnp.zeros((d, d))])
        return B_perp

    def sdc(self, x):
        """Compute an SDC form of the system dynamics at state `x`.

        An SDC form `(A, B)` is such that:
            `dx/dt = A(x - x_bar)@(x - x_bar) + B(x)@(u - u_bar)`,
        where `(x_bar, u_bar)` is an equilibrium pair for this system.
        """
        g, ε = self.gravity, self.coupling
        φ = x[2]
        cφ, sφ = jnp.cos(φ), jnp.sin(φ)
        if self.body_fixed:
            v, w = x[3], x[4]
            A = jnp.array([[0., 0., 0.,         cφ, -sφ, 0.],
                           [0., 0., 0.,         sφ, cφ,  0.],
                           [0., 0., 0.,         0., 0.,  1.],
                           [0., 0., -g*sinc(φ), 0., 0.,  w],
                           [0., 0., -g*cosc(φ), 0., 0.,  -v],
                           [0., 0., 0.,         0., 0.,  0.]])
            B = jnp.array([[0., 0.],
                           [0., 0.],
                           [0., 0.],
                           [0., ε],
                           [1., 0.],
                           [0., 1.]])
        else:
            A = jnp.array([[0., 0., 0.,         1., 0., 0.],
                           [0., 0., 0.,         0., 1., 0.],
                           [0., 0., 0.,         0., 0., 1.],
                           [0., 0., -g*sinc(φ), 0., 0., 0.],
                           [0., 0., g*cosc(φ),  0., 0., 0.],
                           [0., 0., 0.,         0., 0., 0.]])
            B = jnp.array([[0.,  0.],
                           [0.,  0.],
                           [0.,  0.],
                           [-sφ, ε*cφ],
                           [cφ,  ε*sφ],
                           [0.,  1.]])
        return A, B

    def tracking_sdc(self, e, x, u):
        """Compute a tracking SDC form of the system dynamics."""
        φ, dφ, eφ = x[2], x[5], e[2]
        g = self.gravity
        ε = self.coupling
        cφ, sφ = jnp.cos(φ), jnp.sin(φ)
        if self.body_fixed:
            v, w = x[3], x[4]
            ev, ew = e[3], e[4]
            A = jnp.zeros((self.state_dim, self.state_dim))

            # `x`-error state derivative
            A = A.at[0, 2].set((v*cφ - w*sφ)*cosc(eφ) - (v*sφ + w*cφ)*sinc(eφ))
            A = A.at[0, 3].set(jnp.cos(φ + eφ))
            A = A.at[0, 4].set(-jnp.sin(φ + eφ))

            # `y`-error state derivative
            A = A.at[1, 2].set((v*sφ + w*cφ)*cosc(eφ) + (v*cφ - w*sφ)*sinc(eφ))
            A = A.at[1, 3].set(jnp.sin(φ + eφ))
            A = A.at[1, 4].set(jnp.cos(φ + eφ))

            # `phi`-error state derivative
            A = A.at[2, 5].set(1.)

            # `v`-error state derivative
            A = A.at[3, 2].set(-g*(sφ*cosc(eφ) + cφ*sinc(eφ)))
            A = A.at[3, 4].set(dφ)
            A = A.at[3, 5].set(w + ew)

            # `w`-error state derivative
            A = A.at[4, 2].set(-g*(cφ*cosc(eφ) - sφ*sinc(eφ)))
            A = A.at[4, 3].set(-dφ)
            A = A.at[4, 5].set(-(v + ev))

            B = jnp.array([[0., 0.],
                           [0., 0.],
                           [0., 0.],
                           [0., ε],
                           [1., 0.],
                           [0., 1.]])
        else:
            raise NotImplementedError()
        return A, B


class VanDerPolOscillator(SDCDynamics):
    """Dynamics of the controller Van der Pol oscillator."""

    damping:    float = eqx.static_field(default=1.)

    @property
    def dims(self) -> tuple[int, int]:
        """Return the dimensions of the state and input for this system."""
        return 2, 1

    @property
    def equilibrium(self):
        """Return an equilibrium pair `(x_bar, u_bar)` for this system."""
        n, m = self.dims
        x_bar = jnp.zeros(n)
        u_bar = jnp.zeros(m)
        return x_bar, u_bar

    def sdc(self, x):
        """Compute an SDC form of the system dynamics at state `x`."""
        A = jnp.array([[0, 1],
                       [-1, self.damping*(1 - x[0]**2)]])
        B = jnp.array([[0.],
                       [1.]])
        return A, B
