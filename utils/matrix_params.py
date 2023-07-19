"""
Unconstrained parameterizations of various types of matrices.

Author: Spencer M. Richards
        Autonomous Systems Lab (ASL), Stanford
        (GitHub: spenrich)
"""
from math import inf, isqrt

import equinox as eqx

import jax
import jax.numpy as jnp

from jaxtyping import Array

from .misc import identity, softplus, softplus_inverse


def tri_size(n: int, k: int = 0, m: int | None = None) -> int:
    """Compute the number of elements in the lower-triangle of an array."""
    if m is None:
        m = n
    if n <= 0 or m <= 0:
        raise ValueError('Matrix dimensions should be strictly positive.')

    if k < -(n-1):
        raise ValueError('Index of diagonal is too low (`k < -(n-1)`).')
    elif k > m-1:
        raise ValueError('Index of diagonal is too high (`k > m-1`).')

    if n <= m - k:
        d = ((n + k)*(n + k + 1))//2
    else:
        d = n*m - ((m - k)*(m - k - 1))//2
    return d


def tri_size_to_dim(d: int) -> int:
    """Compute the square matrix dimension with `d` lower-triangle elements."""
    n = (isqrt(8*d + 1) - 1) // 2
    if d != tri_size(n):
        raise ValueError('Invalid vector length `d = %d` for filling the '
                         'lower-triangle of a square matrix.' % d)
    return n


def params_to_skew(θ: Array) -> Array:
    """Map parameters to a unique skew-symmetric matrix."""
    θ = jnp.atleast_1d(θ)
    d = θ.shape[-1]
    if d == 0:
        n = 1
        S = jnp.zeros((*θ.shape[:-1], n, n))
    else:
        # Fill triangle above the main diagonal with parameters, and the
        # triangle below the main diagonal with negated parameters
        n = 1 + tri_size_to_dim(d)
        S = jnp.zeros((*θ.shape[:-1], n, n))
        triu_rows, triu_cols = jnp.triu_indices(n, 1)
        S = S.at[..., triu_rows, triu_cols].set(θ)
        S = S.at[..., triu_cols, triu_rows].set(-θ)
    return S


def skew_to_params(S: Array) -> Array:
    """Map a skew-symmetric matrix to unique parameters."""
    n = S.shape[-1]
    triu_rows, triu_cols = jnp.triu_indices(n, 1)
    θ = S[..., triu_rows, triu_cols]
    return θ


def params_to_cholesky(θ: Array) -> Array:
    """Map parameters to a unique non-singular Cholesky factor."""
    θ = jnp.atleast_1d(θ)
    d = θ.shape[-1]
    n = tri_size_to_dim(d)
    L = jnp.zeros((*θ.shape[:-1], n, n))

    # Fill main diagonal with exponentials of first `n` parameters
    diag_rows, diag_cols = jnp.diag_indices(n)
    L = L.at[..., diag_rows, diag_cols].set(softplus(θ[..., :n]))

    # Fill triangle below main diagonal with the remaining parameters
    tril_rows, tril_cols = jnp.tril_indices(n, -1)
    L = L.at[..., tril_rows, tril_cols].set(θ[..., n:])

    return L


def cholesky_to_params(L: Array) -> Array:
    """Map a non-singular Cholesky factor to unique parameters."""
    shape = jnp.shape(L)
    if len(shape) < 2:
        raise ValueError('Argument `L` must be at least 2D!')
    if shape[-2] != shape[-1]:
        raise ValueError('Last two dimensions of `L` must be equal!')
    n = shape[-1]
    diag_rows, diag_cols = jnp.diag_indices(n)
    log_L = L.at[..., diag_rows, diag_cols].set(
        softplus_inverse(L[..., diag_rows, diag_cols])
    )
    tril_rows, tril_cols = jnp.tril_indices(n, -1)
    rows = jnp.concatenate([diag_rows, tril_rows])
    cols = jnp.concatenate([diag_cols, tril_cols])
    θ = log_L[..., rows, cols]
    return θ


def params_to_ortho(θ: Array, method: str = 'cayley') -> Array:
    """Map parameters to a unique orthogonal matrix."""
    if method == 'cayley':
        S = params_to_skew(θ)
        I = jnp.eye(S.shape[-1])  # noqa: E741
        Q = jnp.linalg.solve(I - S, I + S)
    elif method == 'householder':
        # TODO: vectorize this
        d = θ.size  # number of free parameters, d := (n * (n - 1)) // 2
        n = tri_size_to_dim(d) + 1
        Q = jnp.eye(n).at[-1, -1].set(
            jnp.where(n % 2 == 0, -1., 1.)   # initialize Q = H(n)
        )
        # H = jnp.zeros((n, n, n))
        # H = H.at[-1].set(I)
        # H = H.at[-1, -1, -1].set(-1)
        offset = 0
        for p in range(n - 1, 0, -1):  # TODO: use LAX loop
            v = jnp.concatenate((jnp.array([1.]), θ[offset:offset + (n - p)]))
            τ = 2. / jnp.sum(v**2)
            UQ = jnp.outer(τ * v, Q[p-1:, p-1:].T @ v)
            Q = Q.at[p-1:, p-1:].add(-UQ)
            offset += (n - p)
            # H = H.at[p-1].set(I)
            # H = H.at[p-1, p-1:, p-1:].add(-U)
    elif method == 'matrix_exp':
        S = params_to_skew(θ)
        Q = jax.scipy.linalg.expm(S)
    else:
        raise ValueError('Argument `method` must be `cayley`, `householder`, '
                         'or `matrix_exp`.')
    return Q


def ortho_to_params(Q: Array, method: str = 'cayley') -> Array:
    """Map an orthogonal matrix to unique parameters."""
    if method == 'cayley':
        I = jnp.eye(Q.shape[-1])  # noqa: E741
        QT = jnp.swapaxes(Q, -2, -1)
        S = -jnp.linalg.solve(QT + I, QT - I)
        θ = skew_to_params(S)
    elif method == 'householder':
        raise NotImplementedError()
    elif method == 'matrix_exp':
        raise NotImplementedError()
    else:
        raise ValueError('Argument `method` must be `cayley`, `householder`, '
                         'or `matrix_exp`.')
    return θ


def params_to_posdef(θ: Array, method: str = 'cholesky',
                     eig_lower: float = 0., eig_upper: float = 1.) -> Array:
    """Map parameters to a unique positive-definite matrix."""
    if method == 'cholesky':
        L = params_to_cholesky(θ)
        LT = jnp.swapaxes(L, -2, -1)
        P = L @ LT
    elif method in ['cayley', 'householder', 'matrix_exp']:
        n = tri_size_to_dim(θ.shape[-1])
        Q = params_to_ortho(θ[..., n:], method)
        slope = (eig_upper - eig_lower)/2
        bias = (eig_upper + eig_lower)/2
        λ = slope*jnp.tanh(θ[..., :n]/slope) + bias
        P = Q @ (jnp.expand_dims(λ, -1) * Q.T)
    else:
        raise ValueError('Argument `method` must be `cholesky`, `cayley`, '
                         '`householder`, or `matrix_exp`.')
    return P


def posdef_to_params(P: Array, method: str = 'cholesky',
                     eig_lower: float = 0., eig_upper: float = 1.) -> Array:
    """Map a positive-definite matrix to unique parameters."""
    if method == 'cholesky':
        L = jnp.linalg.cholesky(P)
        θ = cholesky_to_params(L)
    elif method in ['cayley', 'householder', 'matrix_exp']:
        raise NotImplementedError()
    else:
        raise ValueError('Argument `method` must be `cholesky`, `cayley`, '
                         '`householder`, or `matrix_exp`.')
    return θ


class PosDefMatrixNN(eqx.Module):
    """Parametric positive-definite matrix function."""

    input_dim:      int = eqx.static_field()
    output_dim:     int = eqx.static_field()
    eig_lower:      float = eqx.static_field()
    eig_upper:      float = eqx.static_field()
    nn:             eqx.Module

    def __init__(self,
                 input_dim:             int,
                 output_dim:            int,
                 hidden_width:          int,
                 hidden_depth:          int,
                 hidden_activation:     callable,
                 final_activation:      callable = identity,
                 eig_lower:             float = 0.,
                 eig_upper:             float = inf,
                 *,
                 key:                   jax.random.PRNGKey):
        """Initialize; see `PosDefMatrixNN`."""
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.eig_lower = eig_lower
        self.eig_upper = eig_upper
        self.nn = eqx.nn.MLP(input_dim, tri_size(output_dim), hidden_width,
                             hidden_depth, hidden_activation, final_activation,
                             key=key)

    def __call__(self, x: Array) -> Array:
        """Evaluate this function."""
        x = jnp.atleast_1d(x)
        y = self.nn(x)
        if self.eig_upper < inf:
            M = params_to_posdef(y, 'householder',
                                 self.eig_lower, self.eig_upper)
        else:
            M = params_to_posdef(y, 'cholesky')
            if self.eig_lower >= 0:
                M += self.eig_lower*jnp.eye(self.output_dim)
        return M
