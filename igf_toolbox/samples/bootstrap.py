import random
import numpy as np
from typing import Tuple, Callable
import warnings
from dataclasses import dataclass, field

@dataclass
class Bootstrap:
    data: np.ndarray
    num_iters: int = 10000
    distribution_name: str = "empirical"
    bootstrap_means: np.ndarray = field(init=False)
    empiric_mean: float = field(init=False)

    def __post_init__(self):
        distribution = self._fit_distribution()
        self.bootstrap_means = [
            np.mean(distribution(size=len(self.data)))
            for _ in range(self.num_iters)
        ]
        self.empiric_mean = float(np.mean(self.bootstrap_means))

    def _fit_distribution(self) -> Callable:
        """
        Fit the distribution specified by dist_name and return a sampling function.
        """
        if self.distribution_name == "empirical":
            return lambda size: np.random.choice(self.data,
                                                size=size,
                                                replace=True)
        elif self.distribution_name == "normal":
            mu, sigma = np.mean(self.data), np.std(self.data, ddof=1)
            return lambda size: np.random.normal(loc=mu, scale=sigma, size=size)
        elif self.distribution_name == "poisson":
            lam = np.mean(self.data)
            return lambda size: np.random.poisson(lam=lam, size=size)
        elif self.distribution_name == "uniform":
            a, b = np.min(self.data), np.max(self.data)
            return lambda size: np.random.uniform(low=a, high=b, size=size)
        else:
            raise ValueError(f"Unsupported distribution : {self.distribution_name}")

    def compute_empiric_mean(self) -> float:
        """
        Return the bootstrap estimate of the sample mean.
        """
        return self.empiric_mean

    def compute_empiric_variance(self) -> float:
        """
        Return the variance of the bootstrap estimate of the sample mean.
        """
        self.empiric_variance = np.mean((np.array(self.bootstrap_means) - self.empiric_mean) ** 2)
        return float(self.empiric_variance)

    def compute_confidence_interval(self, alpha: float = .05) -> Tuple[float, float]:
        """
        Return the lower and upper bounds of he bootstrap estimate
        of the sample mean for a given level.

        Args:
            alpha (float): Level of the confidence interval, defaults to 5%.

        Returns: 
            Tuple[float, float]: A tuple of lower and upper bounds.
        """
        
        if self.num_iters < 1000:
            warnings.warn("`num_iters` should be superior to 1000 to have good confidence intervals", 
                          UserWarning)

        self.lower_bound_ci, self.upper_bound_ci = np.percentile(self.bootstrap_means, 
                                                                 [100*(alpha/2), 100*(1-alpha/2)])

        return float(self.lower_bound_ci), float(self.upper_bound_ci)