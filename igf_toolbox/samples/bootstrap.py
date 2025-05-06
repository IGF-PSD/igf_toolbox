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
    statistic: Callable = np.mean  
    bootstrap_stats: np.ndarray = field(init=False)
    empiric_stat: float = field(init=False)

    def __post_init__(self):

        self.data = np.asarray(self.data).flatten()
        
        distribution = self._fit_distribution()
        self.bootstrap_stats = np.array([
            self.statistic(distribution(size=len(self.data)))
            for _ in range(self.num_iters)
        ])
        self.sample_stat = self.statistic(self.data)
        self.empiric_mean = float(np.mean(self.bootstrap_stats))

    def _fit_distribution(self) -> Callable:
        """
        Fit the distribution specified by distribution_name and return a sampling function.
        """
        if self.distribution_name == "empirical":
            return lambda size: np.random.choice(self.data, size=size, replace=True)
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
            raise ValueError(f"Unsupported distribution: {self.distribution_name}")

    def return_bootstraped_statistic(self) -> np.ndarray:
        """
        Return the bootstraped statistics.
        """
        return self.bootstrap_stats

    def compute_empiric_mean(self) -> float:
        """
        Return the bootstrap mean of the statistic.
        """
        return self.empiric_mean

    def compute_empiric_variance(self) -> float:
        """
        Return the variance of the bootstrap estimate of the statistic.
        """
        return float(np.mean((self.bootstrap_stats - self.empiric_mean) ** 2))

    def compute_confidence_interval(self, 
                                    alpha: float = .05,
                                    type_ci: str = "percentile") -> Tuple[float, float]:
        """
        Return the lower and upper bounds of the bootstrap estimate
        of the statistic for a given confidence level.

        Args:
            alpha (float): Level of the confidence interval, defaults to 5%.
            type_ci (str): Type of confidence interval, one of "percentile" or "basic".

        Returns: 
            Tuple[float, float]: A tuple of lower and upper bounds.
        """
        if self.num_iters < 1000:
            warnings.warn("`num_iters` should be greater than 1000 to ensure reliable confidence intervals",
                          UserWarning)

        if type_ci == "percentile":
            lower, upper = np.percentile(self.bootstrap_stats, [100 * (alpha / 2), 100 * (1 - alpha / 2)])
        elif type_ci == "basic":
            lower_p, upper_p = np.percentile(self.bootstrap_stats, [100 * (alpha / 2), 100 * (1 - alpha / 2)])
            lower = 2 * self.sample_stat - upper_p
            upper = 2 * self.sample_stat - lower_p
        else:
            raise ValueError(f"Unsupported confidence interval type: {type_ci}")

        return float(lower), float(upper)
