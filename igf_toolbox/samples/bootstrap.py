import random
import numpy as np
from typing import Tuple, Callable, Sequence
import warnings
from dataclasses import dataclass, field

from .utils import bootstrap_preprocess_data

@dataclass
class Bootstrap:
    """
    A class for performing bootstrap resampling and statistical inference.

    Args:
        data (Sequence[float]): Input data sample to bootstrap.
        num_iters (int): Number of bootstrap iterations. Defaults to 10000.
        distribution_name (str): Distribution type - "empirical", "normal", "poisson", or "uniform".
        statistic (Callable): Statistical function to apply. Defaults to numpy.mean.

    Attributes:
        bootstrap_stats (np.ndarray): Statistics from bootstrap samples.
        sample_stat (float): Statistic on the original sample.
        empiric_mean (float): Mean of the bootstrap statistics.

    Examples:
        >>> from igf_toolbox.samples.bootstrap import Bootstrap
        >>> data = [1.2, 2.3, 3.1, 4.0, 5.2]
        >>> bs = Bootstrap(data, num_iters=5000, distribution_name="empirical", statistic = np.mean)
        >>> bs.compute_sample_statistic()
        3.16
        >>> bs.compute_confidence_interval(alpha=0.05)
        (2.1, 4.2)
    """

    data: Sequence[float]
    num_iters: int = 10000
    distribution_name: str = "empirical"
    statistic: Callable = np.mean  
    bootstrap_stats: np.ndarray = field(init=False)
    sample_stat: float = field(init=False)
    empiric_mean: float = field(init=False)

    def __post_init__(self):
        """
        Preprocess the data, perform bootstrap resampling, and compute initial statistics.
        """
        self.data = bootstrap_preprocess_data(self.data)
        distribution = self._fit_distribution()
        self.bootstrap_stats = np.array([
            self.statistic(distribution(size=len(self.data)))
            for _ in range(self.num_iters)
        ])
        self.sample_stat = self.statistic(self.data)
        self.empiric_mean = float(np.mean(self.bootstrap_stats))

    def _fit_distribution(self) -> Callable:
        """
        Fit the specified distribution and return a function that generates random samples.

        Returns:
            Callable: A function that takes a size argument and returns a sample.
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
            
    def compute_sample_statistic(self) -> float:
        """
        Return the statistic computed on the original data sample.

        Returns:
            float: The sample statistic.
        """
        return float(self.sample_stat)

    def compute_bootstraped_statistics(self) -> np.ndarray:
        """
        Return the array of statistics computed on the bootstrap samples.

        Returns:
            np.ndarray: Bootstrap statistics.
        """
        return self.bootstrap_stats

    def compute_empiric_mean(self) -> float:
        """
        Return the mean of the bootstrap statistics.

        Returns:
            float: Empirical mean from bootstrap samples.
        """
        return self.empiric_mean

    def compute_empiric_variance(self) -> float:
        """
        Return the variance of the bootstrap statistics.

        Returns:
            float: Empirical variance from bootstrap samples.
        """
        return float(np.mean((self.bootstrap_stats - self.empiric_mean) ** 2))

    def compute_confidence_interval(self, 
                                    alpha: float = .05,
                                    type_ci: str = "percentile") -> Tuple[float, float]:
        """
        Compute a confidence interval for the statistic based on bootstrap results.

        Args:
            alpha (float): Significance level for the confidence interval. Default is 0.05.
            type_ci (str): Type of confidence interval. Options:
                - "percentile": Uses empirical percentiles of the bootstrap distribution.
                - "basic": Reflects percentiles around the sample statistic.

        Returns:
            Tuple[float, float]: Lower and upper bounds of the confidence interval.

        Raises:
            ValueError: If the type_ci is not supported.
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
