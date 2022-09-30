from typing import Optional
from jax import vmap
from jax.interpreters.xla import DeviceArray
import jax.numpy as jnp
from .basis_fns import BasisFunction


class Spline(BasisFunction):
    def __init__(
        self,
        s: Optional[DeviceArray] = None,
        order: Optional[int] = None,
        k: Optional[int] = None,
    ):
        """Construct Spline class.

        Parameters
        ----------
        s:
            Optional knot points for spline basis.
            Defaults to 'k' evenly spaced points if absent.

        order:
            Optional order for splines.
            Defaults to 4 corresponding to cubic splines.

        k:
            Optional number of basis functions.
            Must pass either 's' or 'k'.

        Returns
        -------
        Spline
        """
        self.s = s
        self.order = order if order else 4
        self.k = k  # Need to handle error if neither s and k are passed

    @staticmethod
    def _omega(s1, s2, t):
        return jnp.where(s1 == s2, jnp.zeros_like(t), (t - s1) / (s2 - s1))

    @staticmethod
    def _basis(t, s, order, i):
        if order == 1:
            return jnp.where(
                (t >= s[i]) * (t < s[i + 1]),
                jnp.ones_like(t),
                jnp.zeros_like(t),
            )

        # Recurse left
        w1 = Spline._omega(s[i], s[i + order - 1], t)
        B1 = Spline._basis(t, s, order - 1, i)

        # Recurse right
        w2 = Spline._omega(s[i + 1], s[i + order], t)
        B2 = Spline._basis(t, s, order - 1, i + 1)
        return w1 * B1 + (1 - w2) * B2

    @staticmethod
    def matrix(t, s, order):
        """Construct matrix for spline of
        order 'order' with knots 's' at points 't'.
        """
        _s = jnp.pad(s, mode="edge", pad_width=(order - 1))  # Extend knots
        X = vmap(lambda i: Spline._basis(t, _s, order, i))(
            jnp.arange(0, len(s) + order - 2)
        )  # Make spline basis
        return X.T

    def make_features(
        self, data: Optional[dict] = None, T: Optional[float] = None
    ) -> DeviceArray:
        # Check for maximum time
        if T is None and data is not None:
            T = data["N"].shape[0]

        # If pivots not defined, make self.k equally spaced splines
        if self.s is None and self.k:
            self.s = jnp.linspace(0, T, self.k)

        return self.matrix(jnp.arange(T), self.s, self.order)


class SplineDeriv:
    def __init__(
        self,
        s: Optional[DeviceArray] = None,
        order: Optional[int] = None,
        k: Optional[int] = None,
    ):
        """Construct SplineDeriv class. Represents the derivative of the Spline class.

        Parameters
        ----------
        s:
            Optional knot points for spline basis.
            Defaults to 'k' evenly spaced points if absent.

        order:
            Optional order for splines.
            Defaults to 4 corresponding to cubic splines.

        k:
            Optional number of basis functions.
            Must pass either 's' or 'k'.

        Returns
        -------
        SplineDeriv
        """

        self.s = s
        self.order = order if order else 4
        self.k = k  # Need to handle error if neither s and k are passed

    @staticmethod
    def _omegap(s1, s2, t):
        return jnp.where(s1 == s2, jnp.zeros_like(t), jnp.reciprocal(s2 - s1))

    @staticmethod
    def _basis(t, s, order, i):
        if order == 1:
            return jnp.where(
                (t >= s[i]) * (t < s[i + 1]),
                jnp.ones_like(t),
                jnp.zeros_like(t),
            )

        # Recurse left
        w1 = SplineDeriv._omegap(s[i], s[i + order - 1], t)
        B1 = Spline._basis(t, s, order - 1, i)

        # Recurse right
        w2 = SplineDeriv._omegap(s[i + 1], s[i + order], t)
        B2 = Spline._basis(t, s, order - 1, i + 1)
        return (order - 1) * (w1 * B1 - w2 * B2)

    @staticmethod
    def matrix(t, s, order):
        """Construct matrix for spline derivative of
        order 'order' with knots 's' at points 't'.
        """

        _s = jnp.pad(s, mode="edge", pad_width=(order - 1))  # Extend knots
        X = vmap(lambda i: SplineDeriv._basis(t, _s, order, i))(
            jnp.arange(0, len(s) + order - 2)
        )  # Make spline basis
        return X.T

    def make_features(
        self, data: Optional[dict] = None, T: Optional[float] = None
    ) -> DeviceArray:
        # Check for maximum time
        if T is None and data is not None:
            T = data["N"].shape[0]

        # If pivots not defined, make self.k equally spaced splines
        if self.s is None and self.k:
            self.s = jnp.linspace(0, T, self.k)

        return self.matrix(jnp.arange(T), self.s, self.order)
