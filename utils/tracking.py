"""
Joint learning of nonlinear dynamics, controllers, and certificates.

Author: Spencer M. Richards
        Autonomous Systems Lab (ASL), Stanford
        (GitHub: spenrich)
"""
from typing import Union

import equinox as eqx

import jax
import jax.numpy as jnp

from jaxtyping import Array

from .dynamics_jax import ControlAffineDynamics, solve_care
from .matrix_params import params_to_posdef, tri_size
from .misc import broadcast_to_square_matrix, identity, solve_pos


def eigvalh_max(M: Array) -> float:
    """Compute the maximum eigenvalue of the Hermitian matrix `M`."""
    eig_max = jnp.linalg.eigvalsh(M)[-1]
    return eig_max


class NeuralLQRController(eqx.Module):
    """LQR controller with learned control-affine dynamics."""

    state_dim:          int = eqx.static_field()
    control_dim:        int = eqx.static_field()
    model:              ControlAffineDynamics = eqx.static_field()
    nn_caf:             eqx.Module
    preconditioner:     callable = eqx.static_field()

    def __init__(self, model:       ControlAffineDynamics,
                 hidden_width:      int,
                 hidden_depth:      int,
                 hidden_activation: callable,
                 final_activation:  callable = identity,
                 preconditioner:    callable = identity,
                 *, key:            jax.random.PRNGKey):
        """Initialize; see `LQRController`."""
        self.state_dim = model.state_dim
        self.control_dim = model.control_dim
        self.model = model
        self.preconditioner = preconditioner
        n, m = self.state_dim, self.control_dim
        self.nn_caf = eqx.nn.MLP(n, (1 + m)*n, hidden_width, hidden_depth,
                                 hidden_activation, final_activation,
                                 key=key)

    def caf(self, x: Array) -> tuple[Array, Array]:
        """Evaluate the control-affine terms `(f, B)`."""
        n, m = self.state_dim, self.control_dim
        f, vec_B = jnp.split(self.nn_caf(self.preconditioner(x)), [n, ])
        B = jnp.reshape(vec_B, (n, m))
        return f, B

    def dynamics(self, x: Array, u: Array) -> Array:
        """Evaluate the dynamics."""
        f, B = self.caf(x)
        dx = f + B@u
        return dx

    def __call__(self, x: Array, x_ref: Array, u_ref: Array,
                 Q: Array | float = 1., R: Array | float = 1.):
        """Compute the tracking control input at state `x`."""
        A = jax.jacfwd(self.dynamics)(x_ref, u_ref)
        _, B = self.caf(x_ref)
        Q = broadcast_to_square_matrix(Q, self.state_dim)
        R = broadcast_to_square_matrix(R, self.control_dim)
        M = solve_care(A, B, Q, R)  # TODO: use pure JAX
        dV = M @ (x - x_ref)
        u = u_ref - solve_pos(R, B.T @ dV)
        return u

    def loss_regression(self, X, U, Y):
        """Compute the loss for training the dynamics model."""
        Y_est = jax.vmap(self.dynamics)(X, U)
        loss = jnp.mean(jnp.sum((Y_est - Y)**2, axis=-1))
        return loss

    def loss_auxiliary(self, x: Array, u: Array, x_ref: Array, u_ref: Array):
        """Compute the loss for training the controller."""
        return 0.


class NeuralSDCController(eqx.Module):
    """Neural network approximator of dynamics in SDC form."""

    state_dim:          int = eqx.static_field()
    control_dim:        int = eqx.static_field()
    model:              ControlAffineDynamics = eqx.static_field()
    nn_sdc:             eqx.Module
    nn_caf:             eqx.Module
    preconditioner:     callable = eqx.static_field()
    bootstrap:          bool = eqx.static_field()
    learn_caf:          bool = eqx.static_field()
    stop_grad:          bool = eqx.static_field()

    def __init__(self, model:       ControlAffineDynamics,
                 hidden_width:      int,
                 hidden_depth:      int,
                 hidden_activation: callable,
                 final_activation:  callable = identity,
                 preconditioner:    callable = identity,
                 learn_caf:         bool = False,
                 bootstrap:         bool = False,
                 stop_grad:         bool = False,
                 *, key:            jax.random.PRNGKey):
        """Initialize; see `LinearizedLQRController`."""
        self.model = model
        self.state_dim = model.state_dim
        self.control_dim = model.control_dim
        self.preconditioner = preconditioner
        self.learn_caf = learn_caf
        self.bootstrap = bootstrap
        self.stop_grad = stop_grad
        n, m = self.state_dim, self.control_dim
        key_sdc, key_caf = jax.random.split(key, 2)
        self.nn_sdc = eqx.nn.MLP(2*n, (1 + m)*(n**2),
                                 hidden_width, hidden_depth, hidden_activation,
                                 final_activation, key=key_sdc)
        if self.learn_caf:
            self.nn_caf = eqx.nn.MLP(n, (1 + m)*n, hidden_width, hidden_depth,
                                     hidden_activation, final_activation,
                                     key=key_caf)
        else:
            self.nn_caf = eqx.nn.Identity()

    def caf(self, x: Array) -> tuple[Array, Array]:
        """Evaluate the control-affine terms `(f, B)`."""
        if self.learn_caf:
            # Learn `(f, B)`
            n, m = self.state_dim, self.control_dim
            f, vec_B = jnp.split(self.nn_caf(self.preconditioner(x)), [n, ])
            B = jnp.reshape(vec_B, (n, m))
        else:
            # Use provided `(f, B)`
            f, B = self.model.caf(x)
        return f, B

    def dynamics(self, x: Array, u: Array) -> Array:
        """Evaluate the dynamics."""
        f, B = self.caf(x)
        dx = f + B@u
        return dx

    def sdc_matrices(self, x: Array, e: Array) -> tuple[Array, ...]:
        """Evaluate the SDC form of the error dynamics."""
        n, m = self.state_dim, self.control_dim
        y = jnp.concatenate([self.preconditioner(x), self.preconditioner(e)])
        z = self.nn_sdc(y)
        Af = jnp.reshape(z[:n**2], (n, n))
        Bf = jnp.reshape(z[n**2:], (m, n, n))
        return Af, Bf

    def __call__(self, x: Array, x_ref: Array, u_ref: Array,
                 Q: Array | float = 1., R: Array | float = 1.,
                 linearize: bool = False) -> Array:
        """Compute the tracking control input at state `x`."""
        e = x - x_ref
        if linearize:
            A = jax.jacfwd(self.dynamics)(x_ref, u_ref)
        else:
            A, Bf = self.sdc_matrices(x_ref, e)
            for j in range(self.control_dim):
                A += u_ref[j]*Bf[j]
        _, B = self.caf(x)
        Q = broadcast_to_square_matrix(Q, self.state_dim)
        R = broadcast_to_square_matrix(R, self.control_dim)
        M = solve_care(A, B, Q, R)
        dV = M @ e
        u = u_ref - solve_pos(R, B.T @ dV)
        return u

    def loss_regression_single(self, x, u, y, x_ref, u_ref, y_ref):
        """Compute the loss for training the dynamics model."""
        # Fit control-affine (CAF) form to data
        f, B = self.caf(x)
        y_caf = f + B@u
        loss = jnp.sum((y_caf - y)**2)

        # Fit generalized SDC form to data
        e, v, z = x - x_ref, u - u_ref, y - y_ref
        Af, Bf = self.sdc_matrices(x_ref, e)
        df_sdc, dB_sdc = Af@e, (Bf@e).T
        z_sdc = df_sdc + dB_sdc@u_ref + B@v
        loss += jnp.sum((z_sdc - z)**2)
        return loss

    def loss_regression(self, X, U, Y):
        """Compute the loss for training the dynamics model."""
        if self.learn_caf:
            # TODO: too much memory required
            # N = X.shape[0]
            # idx = np.array(list(itertools.permutations(range(N), 2)))
            # E = X[idx[:, 1]] - X[idx[:, 0]]
            # dE = Y[idx[:, 1]] - Y[idx[:, 0]]
            # V = U[idx[:, 1]] - U[idx[:, 0]]
            # X, U = X[idx[:, 0]], U[idx[:, 0]]
            X_ref, U_ref, Y_ref = X[:-1], U[:-1], Y[:-1]
            X, U, Y = X[1:], U[1:], Y[1:]
            loss = jnp.mean(
                jax.vmap(self.loss_regression_single)(X, U, Y,
                                                      X_ref, U_ref, Y_ref)
            )
        else:
            # When we know the dynamics, the only contribution to the loss
            # should be from the mismatch already penalized in `loss_auxiliary`
            loss = 0.
        return loss

    def loss_auxiliary(self, x: Array, u: Array, x_ref: Array, u_ref: Array):
        """Compute the loss for training the controller."""
        # Dynamics perturbation based on the SDC form
        e = x - x_ref
        Af, Bf = self.sdc_matrices(x_ref, e)
        df_sdc, dB_sdc = Af@e, (Bf@e).T

        # Dynamics perturbation based on the CAF form
        f_ref, B_ref = self.caf(x_ref)
        f, B = self.caf(x)
        df_caf, dB_caf = f - f_ref, B - B_ref

        # Boot-strap SDC learning onto CAF learning by penalizing mismatch
        if self.stop_grad:
            df_caf = jax.lax.stop_gradient(df_caf)
            dB_caf = jax.lax.stop_gradient(dB_caf)
        loss = jnp.sum((df_sdc - df_caf)**2) + jnp.sum((dB_sdc - dB_caf)**2)
        return loss


class NeuralCCMController(eqx.Module):
    """Parametric CCM-based controller."""

    state_dim:              int = eqx.static_field()
    control_dim:            int = eqx.static_field()
    model:                  ControlAffineDynamics
    nn_metric:              eqx.nn.MLP
    nn_controller:          eqx.nn.MLP
    nn_caf:                 eqx.Module
    contraction_rate:       float = eqx.static_field()
    overshoot:              float = eqx.static_field()
    preconditioner:         callable = eqx.static_field()
    margin:                 float = eqx.static_field()
    eig_lower:              float = eqx.static_field()
    eig_upper:              float = eqx.static_field()
    hard_eig_bound:         bool = eqx.static_field()
    learn_caf:              bool = eqx.static_field()

    def __init__(self, model:           ControlAffineDynamics,
                 hidden_width:          int,
                 hidden_depth:          int,
                 hidden_activation:     callable,
                 final_activation:      callable = identity,
                 preconditioner:        Union[eqx.Module, callable] = identity,
                 contraction_rate:      float = 1.,
                 overshoot:             float = 10.,
                 eig_lower:             float = 0.1,
                 margin:                float = 0.,
                 hard_eig_bound:        bool = False,
                 learn_caf:             bool = False,
                 *, key:                jax.random.PRNGKey):
        """Initialize; see `NeuralCCMController`."""
        self.state_dim = model.state_dim
        self.control_dim = model.control_dim
        self.model = model
        self.contraction_rate = contraction_rate
        self.overshoot = jnp.maximum(1., overshoot)
        self.eig_lower = eig_lower
        self.eig_upper = (self.overshoot**2) * self.eig_lower
        self.preconditioner = preconditioner
        self.margin = margin
        self.hard_eig_bound = hard_eig_bound
        self.learn_caf = learn_caf

        n, m = self.state_dim, self.control_dim
        key_metric, key_ctrl = jax.random.split(key, 2)
        if hard_eig_bound:
            output_dim = tri_size(n)
        else:
            output_dim = n**2
        self.nn_metric = eqx.nn.MLP(n, output_dim, hidden_width,
                                    hidden_depth, hidden_activation,
                                    final_activation, key=key_metric)
        self.nn_controller = eqx.nn.MLP(2*n, n*m, hidden_width, hidden_depth,
                                        hidden_activation, final_activation,
                                        key=key_ctrl)

        if self.learn_caf:
            key_caf, key = jax.random.split(key, 2)
            self.nn_caf = eqx.nn.MLP(n, (1 + m)*n, hidden_width, hidden_depth,
                                     hidden_activation, final_activation,
                                     key=key_caf)
        else:
            self.nn_caf = eqx.nn.Identity()

    def __call__(self, x: Array, x_ref: Array, u_ref: Array) -> Array:
        """Evaluate this controller."""
        x, x_ref, u_ref = [jnp.atleast_1d(a) for a in (x, x_ref, u_ref)]
        y = self.nn_controller(jnp.concatenate([self.preconditioner(x),
                                                self.preconditioner(x_ref)]))
        K = jnp.reshape(y, (self.control_dim, self.state_dim))
        u = u_ref + K@(x - x_ref)
        return u

    def caf(self, x: Array) -> tuple[Array, Array]:
        """Evaluate the control-affine terms `(f, B)`."""
        if self.learn_caf:
            # Learn `(f, B)`
            n, m = self.state_dim, self.control_dim
            f, vec_B = jnp.split(self.nn_caf(self.preconditioner(x)), [n, ])
            B = jnp.reshape(vec_B, (n, m))
        else:
            # Use provided `(f, B)`
            f, B = self.model.caf(x)
        return f, B

    def dynamics(self, x: Array, u: Array) -> Array:
        """Evaluate the dynamics."""
        f, B = self.caf(x)
        dx = f + B@u
        return dx

    def ccm(self, x: Array) -> Array:
        """Evaluate the estimated CCM `M(x)` at state `x`."""
        x = jnp.atleast_1d(x)
        y = self.nn_metric(self.preconditioner(x))
        if self.hard_eig_bound:
            M = params_to_posdef(y, method='householder',
                                 eig_lower=self.eig_lower,
                                 eig_upper=self.eig_upper)
        else:
            # M = params_to_posdef(y, method='cholesky')
            L = jnp.reshape(y, (self.state_dim, self.state_dim))
            M = L@L.T + self.eig_lower*jnp.eye(self.state_dim)
        return M

    def closed_loop(self, x, x_ref, u_ref):
        """Evaluate the closed-loop dynamics."""
        u = self.__call__(x, x_ref, u_ref)
        dx = self.dynamics(x, u)
        return dx

    def loss_regression(self, X, U, Y):
        """Compute the loss for training the dynamics model."""
        if self.learn_caf:
            Y_est = jax.vmap(self.dynamics)(X, U)
            loss = jnp.mean(jnp.sum((Y_est - Y)**2, axis=-1))
        else:
            loss = 0.
        return loss

    def loss_auxiliary(self, x: Array, u: Array, x_ref: Array, u_ref: Array):
        """Compute the loss for training this controller."""
        x_dot = self.closed_loop(x, x_ref, u_ref)
        M, M_dot = jax.jvp(self.ccm, (x,), (x_dot,))
        A = jax.jacfwd(self.closed_loop)(x, x_ref, u_ref)

        # Controlled contraction condition (plus a margin to encourage
        # generalization to nearby points)
        MA = M@A
        C = M_dot + (MA + MA.T) + 2*self.contraction_rate*M
        loss = jnp.maximum(0., eigvalh_max(C) + self.margin)

        # Desired overshoot (via metric eigenvalue upper bound)
        if not self.hard_eig_bound:
            loss += jnp.maximum(0., eigvalh_max(M) - self.eig_upper)
        return loss
