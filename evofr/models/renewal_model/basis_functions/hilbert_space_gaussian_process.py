from typing import Optional
from jax import vmap
from jax.interpreters.xla import DeviceArray
import jax.numpy as jnp
from jax.scipy.special import gammaln
from .basis_fns import BasisFunction


class HSGaussianProcess(BasisFunction):
    """
    Implementation of basis approximation to Gaussian processes.
    """

    def __init__(self):
        pass

    @staticmethod
    def lam(L: float, m: int):
        return jnp.square(m * jnp.pi / (2 * L))

    @staticmethod
    def phi(L: float, m: int, x: float):
        lam = HSGaussianProcess.lam(L, m)
        arg = jnp.sqrt(lam) * (x + L)
        return jnp.sqrt(1 / L) * jnp.sin(arg)

    @staticmethod
    def phi_matrix(L: float, m: int, x: float):
        phi_mapped = vmap(
            HSGaussianProcess.phi, in_axes=(None, 0, None), out_axes=-1
        )
        return phi_mapped(L, m, x)


class SquaredExponential(HSGaussianProcess):
    def __init__(
        self,
        alpha: Optional[float] = None,
        rho: Optional[float] = None,
        L: Optional[float] = None,
        m: Optional[int] = None,
    ):
        self.alpha = alpha if alpha else 1.0
        self.rho = rho if rho else 1.0
        self.L = L if L else 10.0
        self.m = m if m else 5

    @staticmethod
    def spd(alpha: float, rho: float, w: float):
        return (
            alpha
            * jnp.sqrt(2 * jnp.pi)
            * rho
            * jnp.exp(-0.5 * jnp.square(rho) * jnp.square(w))
        )

    def make_features(self, data: dict) -> DeviceArray:
        T = data["N"].shape[0]

        # Make time period
        ts = jnp.arange(T)

        # Make eigenvalues
        ms = jnp.arange(1, self.m + 1)
        lams = self.lam(self.L, ms)

        # Make eigenvectors
        phi = self.phi_matrix(self.L, ms, ts)

        # Make spectral density
        delta = self.spd(self.alpha, self.rho, jnp.sqrt(lams))

        # Return feature matrix
        return phi * jnp.sqrt(delta)


class Matern(HSGaussianProcess):
    def __init__(
        self,
        alpha: Optional[float] = None,
        rho: Optional[float] = None,
        nu: Optional[float] = None,
        L: Optional[float] = None,
        m: Optional[int] = None,
    ):
        self.alpha = alpha if alpha else 1.0
        self.rho = rho if rho else 1.0
        self.nu = nu if nu else 1.5
        self.L = L if L else 10.0
        self.m = m if m else 5

    @staticmethod
    def spd(alpha: float, rho: float, nu: float, w: float):
        gammanu = jnp.exp(gammaln(nu))
        gammanuhalf = jnp.exp(gammaln(nu + 0.5))
        coef = (
            alpha
            * 2
            * jnp.sqrt(jnp.pi)
            * gammanuhalf
            * jnp.power(2 * nu, nu)
            / (gammanu * jnp.power(rho, 2 * nu))
        )
        base = 2 * nu * jnp.power(rho, -2) + 4 * jnp.square(jnp.pi * w)
        expon = -nu - 0.5
        return coef * jnp.power(base, expon)

    def make_features(self, data: dict) -> DeviceArray:
        T = data["N"].shape[0]

        # Make time period
        ts = jnp.arange(T)

        # Make eigenvalues
        ms = jnp.arange(1, self.m + 1)
        lams = self.lam(self.L, ms)

        # Make eigenvectors
        phi = self.phi_matrix(self.L, ms, ts)

        # Make spectral density
        delta = self.spd(self.alpha, self.rho, self.nu, jnp.sqrt(lams))

        # Return feature matrix
        return phi * jnp.sqrt(delta)
