"""
Miscellaneous utility functions.

Author: Spencer M. Richards
        Autonomous Systems Lab (ASL), Stanford
        (GitHub: spenrich)
"""
from functools import partial, wraps

import equinox as eqx

import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint

from jaxtyping import Array, PyTree

import numpy as np

# Solve linear systems `A@X = B` assuming `A` is symmetric positive-definite
solve_pos = partial(jax.scipy.linalg.solve, assume_a='pos')

# Compute numerically stable implementations of:
# Softplus:         `jnp.log(jnp.exp(x) + 1)`
# Softplus inverse: `jnp.log(jnp.exp(x) - 1)`
softplus = jax.nn.softplus


def softplus_inverse(x):
    """Evaluate the inverse of the softplus function."""
    return x + jnp.log(-jnp.expm1(-x))


def identity(x):
    """Evaluate the identity function."""
    return x


def pytree_sos(x_tree: PyTree):
    """Compute sum of squared PyTree elements."""
    def reducer(x, y):
        return x + jnp.sum(y**2)
    x_tree, _ = eqx.partition(x_tree, eqx.is_array, replace=0.)
    sos = jax.tree_util.tree_reduce(reducer, x_tree, 0.)
    return sos


def pytree_permute(key, x_tree: PyTree):
    """Permute PyTree leaves."""
    x_tree, _ = eqx.partition(x_tree, eqx.is_array, replace=0.)
    permuter = partial(jax.random.permutation, key)
    permuted = jax.tree_util.tree_map(permuter, x_tree)
    return permuted


def jax_passthrough(func):
    """Wrap a JAX function to accept and output NumPy arrays."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        jax_args = jax.tree_util.tree_map(jnp.array, args)
        jax_output = func(*jax_args, **kwargs)
        output = jax.tree_util.tree_map(np.array, jax_output)
        return output
    return wrapper


def broadcast_to_square_matrix(A: float | Array, n: int):
    """Broadcast argument `A` to a conformable square matrix."""
    A = jnp.asarray(A)
    if A.ndim > 2:
        raise ValueError('Argument `A` has too many dimensions.')
    elif A.ndim <= 1:
        A = jnp.broadcast_to(A, (n,))
        A = jnp.diag(A)
    return A


def trapezoid(y, x, initial=None):
    """Cumulatively integrate `y(x)` using the trapezoidal rule."""
    dx = jnp.diff(x)
    dy_dx = (y[:-1] + y[1:]) / 2
    int_y = jnp.cumsum(dy_dx * dx)
    if initial is not None:
        initial = jnp.reshape(initial, (1,))
        int_y = jnp.concatenate((initial, int_y + initial))
    return int_y


def interp(x, xp, fp, left=None, right=None, axis=-1):
    """Linearly interpolate data along a given axis."""
    i = jnp.searchsorted(xp, x, side='right')
    i = jnp.clip(i, 1, xp.shape[axis] - 1)
    dx = xp[i] - xp[i-1]
    df = fp.take(i, axis) - fp.take(i - 1, axis)

    # Use numpy for static operations based on shapes
    shape = np.ones(fp.ndim, dtype=int)
    shape[axis] = x.size

    # Interpolate linearly
    scale = jnp.reshape((x - xp[i-1])/dx, shape)
    f = fp.take(i - 1, axis) + scale*df

    # Extrapolate with constant fill values
    left = fp.take(0, axis) if left is None else left
    right = fp.take(-1, axis) if right is None else right
    f = jnp.where(jnp.reshape(x < xp[0], shape), left, f)
    f = jnp.where(jnp.reshape(x > xp[-1], shape), right, f)
    return f


def schedule_const(t, learning_rate):
    """Return a constant learning rate."""
    return learning_rate


def schedule_decay(t, learning_rate, decay=1e-4, period=1000):
    """Compute a learning rate according to a decaying schedule."""
    lr = learning_rate * ((1 - decay)**jnp.floor(t/period))
    return lr


def schedule_cosine(t, learning_rate, floor=1e-4, period=1000):
    """Compute a learning rate according to a cosine-annealing schedule."""
    floor = jnp.minimum(learning_rate, floor)
    phi = jnp.pi*jnp.mod(t, period)/period
    lr = floor + (learning_rate - floor)*(1 + jnp.cos(phi))/2
    return lr


def simulate(system, controller, x0, t, t_ref, x_ref, u_ref,
             u_lb=-jnp.inf, u_ub=jnp.inf, Q=1., R=1.,
             clip_ctrl=False, zoh=False):
    """Simulate the closed-loop system."""
    xc, uc = system.equilibrium
    if not clip_ctrl:
        u_lb, u_ub = -jnp.inf, jnp.inf
    u_lb = jnp.broadcast_to(u_lb, (system.control_dim,))
    u_ub = jnp.broadcast_to(u_ub, (system.control_dim,))
    xr_func = lambda t: interp(t, t_ref, x_ref, axis=0, right=xc)  # noqa: E731
    ur_func = lambda t: interp(t, t_ref, u_ref, axis=0, right=uc)  # noqa: E731
    x_ref_t = xr_func(t)
    u_ref_t = ur_func(t)

    # Simulate, possibly with a zero-order hold
    if zoh:
        def body(carry, scan):
            t, t_next, x_ref, u_ref = scan
            x = carry
            u = jnp.clip(controller(x, x_ref, u_ref), u_lb, u_ub)
            x_next = system(x, u, zoh=True, dt=t_next-t)
            carry = x_next
            output = (x_next, u)
            return carry, output

        scans = (t[:-1], t[1:], x_ref_t[:-1], u_ref_t[:-1])
        _, (x_next, u) = jax.lax.scan(body, x0, scans)
        x = jnp.vstack([x0, x_next])
        u_last = jnp.clip(controller(x[-1], x_ref_t[-1], u_ref_t[-1]),
                          u_lb, u_ub)
        u = jnp.vstack([u, u_last])
    else:
        def ode_cl(x, t):
            xr = xr_func(t).ravel()
            ur = ur_func(t).ravel()
            u = jnp.clip(controller(x, xr, ur), u_lb, u_ub)
            return system(x, u)

        x = odeint(ode_cl, x0, t)
        u = jnp.clip(jax.vmap(controller)(x, x_ref, u_ref), u_lb, u_ub)

    # Integrate tracking cost with the trapezoid rule
    Q = broadcast_to_square_matrix(Q, system.state_dim)
    R = broadcast_to_square_matrix(R, system.control_dim)
    e = x - x_ref_t
    v = u - u_ref_t
    costs = jnp.sum((e@Q)*e, axis=-1)/2 + jnp.sum((v@R)*v, axis=-1)/2
    J = trapezoid(costs, t, initial=0.)
    return x, u, x_ref_t, u_ref_t, J
