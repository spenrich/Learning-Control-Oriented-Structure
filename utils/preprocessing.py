"""
Utility functions for pre-processing datasets.

Author: Spencer M. Richards
        Autonomous Systems Lab (ASL), Stanford
        (GitHub: spenrich)
"""
from abc import ABC, abstractmethod

import jax.numpy as jnp


class AffineScaler(ABC):
    """Base class for affine data processing transforms."""

    @abstractmethod
    def transform(self, X):
        """Scale the data."""
        raise NotImplementedError()

    @abstractmethod
    def inverse_transform(self, X):
        """Un-scale the data."""
        raise NotImplementedError()

    def __call__(self, X, invert=False):
        """Apply this scaling."""
        Y = self.inverse_transform(X) if invert else self.transform(X)
        return Y


class IdentityScaler(AffineScaler):
    """TODO."""

    def __init__(self, *args, **kwargs):
        """TODO."""
        pass

    def transform(self, X):
        """TODO."""
        return X

    def inverse_transform(self, Y):
        """TODO."""
        return Y


class StandardScaler(AffineScaler):
    """TODO."""

    def __init__(self, X):
        """TODO."""
        self.mean = jnp.mean(X, axis=0)
        self.std = jnp.std(X, axis=0)

    def transform(self, X):
        """TODO."""
        return (X - self.mean) / self.std

    def inverse_transform(self, Y):
        """TODO."""
        return self.std*Y + self.mean


class MinMaxScaler(AffineScaler):
    """TODO."""

    def __init__(self, X):
        """TODO."""
        self.min = jnp.max(X, axis=0)
        self.max = jnp.min(X, axis=0)

    def transform(self, X):
        """TODO."""
        return (X - self.min) / (self.max - self.min)

    def inverse_transform(self, Y):
        """TODO."""
        return (self.max - self.min)*Y + self.min
