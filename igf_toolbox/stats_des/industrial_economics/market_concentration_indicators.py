import warnings
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from numba import njit

from utils import validate_market_shares_data

@njit(return_type=float)
def _gini_index(x: np.ndarray) -> float:
    """
    Computes the Gini Index using numba.

    Args:
        x (np.ndarray): A 1D numpy array of floats.

    Returns:
        float: The Gini Index.
    """

    n=len(x)
    total_x = np.sum(x)
    x = np.sort(x)
    indices = np.arange(1, n + 1)
    weighted_total_x = np.sum(indices * x)
    return float((2 * weighted_total_x) / (n * total_x) - (n + 1) / n)


def gini_index(x: Sequence[float], verbose: bool = False) -> float:
    """
    Returns the Gini Index of a sequence of floats.

    Args:
        x (Sequence[float]): Sequence of market shares.
        verbose (bool, optional): If True, additional information is printed during the computation.
                                  Defaults to False.

    Returns:
        float: The Gini Index between 0 for perfect equality and 1 for perfect inequality.

    Raises:
        ValueError: If `x` is not a 1D array, is empty, contains negative values, or NaN values.
    """

    x = validate_market_shares_data(x)

    if verbose:
        print("Gini index ranges from 0 (perfect equality) to 1 (perfect inequality).")

    total_x = x.sum()
    if total_x == 0:
        if verbose:
            print("Gini index is 0 because all market shares are 0.")
        return float(0.0)
    else:
        return _gini_index(x)


def lorenz_curve(x: Sequence[float], verbose: bool = False) -> None:
    """
    Plots the Lorenz Curve of an array.

    Args:
        x (Sequence[float]): Sequence of market shares.
        verbose (bool, optional): If True, additional information is printed during the computation.
                                  Defaults to False.

    Returns:
        None.
    """

    x = validate_market_shares_data(x)

    total_x = x.sum()
    if total_x == 0:
        if verbose:
            print("All market shares are zero. Lorenz Curve is a straight line at y=0.")
        plt.plot([0, 1], [0, 0], color="#eb7d0f", label="Lorenz Curve (Zero Market)")
        plt.plot(
            [0, 1], [0, 1], color="#096c45", linestyle="--", label="Perfect Equality"
        )
        plt.xlabel("Cumulative Share of Population")
        plt.ylabel("Cumulative Share of Market")
        plt.legend()
        plt.show()
        return

    # Compute Lorenz curve
    x = np.sort(x)
    lorenz_curve = np.cumsum(x) / total_x
    lorenz_curve = np.insert(lorenz_curve, 0, 0)  # Insert 0 at the beginning

    # Population proportion (x-axis)
    pop_share = np.linspace(0, 1, len(lorenz_curve))

    # Plot Lorenz Curve
    fig, ax = plt.subplots(figsize=[6, 6])
    ax.plot(pop_share, lorenz_curve, color="#eb7d0f", linewidth=2, label="Lorenz Curve")
    ax.plot([0, 1], [0, 1], color="#096c45", linestyle="--", label="Perfect Equality")

    # Labels & legend
    ax.set_xlabel("Cumulative Share of Population")
    ax.set_ylabel("Cumulative Share of Market")
    ax.set_title("Lorenz Curve")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)

    plt.show()

@njit(return_type=float)
def _hhi(x: np.ndarray) -> float:
    """
    Computes the Herfindahl-Hirschman Index using numba.

    Args:
        x (np.ndarray): A 1D numpy array of floats.

    Returns:
        float: The HHI.
    """
    return np.sum(x**2)
    
def hhi(x: Sequence[float], normalize: bool = False, verbose: bool = False) -> float:
    """
    Computes the Herfindahl-Hirschman Index (HHI) for market concentration.

    Args:
        x (Sequence[float]): A 1D sequence of market shares (values between 0 and 1).
        normalize (bool, optional): If True, HHI is normalized to range from 0 to 1.
                                    Defaults to False (returns standard HHI).
        verbose (bool, optional): If True, additional information is printed during the computation.
                                  Defaults to False.

    Returns:
        float: The HHI value.
               - If `normalize=False`, HHI ranges from 0 (perfect competition) to 1 (monopoly).
               - If `normalize=True`, HHI is adjusted to account for the number of firms.
               - If shares are mistakenly given as percentages (>1 sum), a warning is raised.

    Raises:
        ValueError: If `x` is not a 1D array, is empty, contains negative values, or NaN values.
        ValueError: If normalization is applied with regard to only one company.
    """

    x = validate_market_shares_data(x)

    # Warn about shares in percentage
    if np.sum(x) > 1:
        warnings.warn(
            """Market shares seem expressed in % as the sum
                      of market shares is strictly higher than 1.""",
            UserWarning,
        )

    if verbose:
        print(
            """
            The sum of market shares should be 1, or 100 when market
            shares are provided in %. 
            HHI higher than 0.25 (or 2500 for percentage data)
            means high level of concentration
            """
        )
        print(f"Sum of market shares : {np.sum(x)}.")

    # Compute HHI with normalization if necessary
    hhi = _hhi(x)

    if normalize:

        if verbose:
            print(f"Unnormalized HHI : {hhi}.")

        n = len(x)
        if n > 1:
            hhi = (hhi - 1 / n) / (1 - 1 / n)
        else:
            raise ValueError(
                """Normalization is not meaningful with only 1 company (zero-division in formula)."""
            )

        if verbose:
            print(f"Normalized HHI : {hhi}.")

    return float(hhi)

def concentration_ratio(x: Sequence[float], k: int = 3, verbose: bool = False) -> float:
    """
    Returns the concentration ratio of a market for a parameter n.

    Args:
        x (Sequence[float]): Sequence of market shares.
        k (int, optional): Number of companies to consider as top k market shares.
        verbose (bool, optional): If True, additional information is printed during the computation.
                                  Defaults to False.

    Returns:
        float: Concentration ratio from 0 to 0.4 competitive market,
                from 0.4 to 0.7 medium concentration, from 0.7 to 1 high concentration.

    Raises:
        ValueError: If `x` is not a 1D array, is empty, contains negative values, or NaN values.
        ValueError: If `k` is not a strictly positive integer.
    """

    x = validate_market_shares_data(x)

    # Check if k is a strictly positive integer
    if not isinstance(k, int) or k <= 0:
        raise ValueError(f"Parameter `k` must be a strictly positive integer. Got {k}.")

    # Limit k to the total number of companies
    # in case where k is higher than this total number
    if k > x.size and verbose:
        print(
            f"Parameter `k` exceeds the total number of companies ({x.size}). Limiting `k` to {x.size}."
        )

    k = min(k, x.size)

    # Check if market shares are given in %
    if np.sum(x) > 1:
        warnings.warn(
            """The sum of market shares exceeds 1, suggesting that they may be in percentage form.
                         Please verify the data.""",
            UserWarning,
        )

    # Get the k largest market shares
    # We use np.partition for O(n) time complexity
    top_k = np.partition(x, -k)[-k:]

    # Return the sum of the k-largest market shares.
    return float(np.sum(top_k))

@njit(return_type=float)
def _shannon_entropy(x: np.ndarray) -> float:
    """
    Computes the Shannon Entropy using numba.

    Args:
        x (np.ndarray): A 1D numpy array of floats.

    Returns:
        float: The Shannon Entropy.
    """
    
    # Normalize data for Shannon entropy to be a probability distribution
    # As we assume all data is non negative it should sum up to 1
    x = x / np.sum(x)

    # When 0 are in the data don't comput log only return 0
    return float(-np.sum(np.where(x > 0, x * np.log(x), 0)))

def shannon_entropy(x: Sequence[float], verbose: bool = False) -> float:
    """
    Computes the Shannon Entropy for market concentration.

    Args:
        x (Sequence[float]): A 1D sequence of market shares.
        verbose (bool, optional): If True, additional information is printed during the computation.
                                  Defaults to False.

    Returns:
        float: Shannon Entropy, from 0 for monopoly, to log(n) for diversified market,
                where n is the number of competitors.

    Raises:
        ValueError: If `x` is not a 1D array, is empty, contains negative values, or NaN values.
        ValueError: If Shannon Entropy is not defined as all values are zero.
    """

    x = validate_market_shares_data(x)

    # Check if all values are equal to zero
    if np.sum(x) == 0:
        raise ValueError(
            """All market shares are equal to zero
                        Shannon Entropy is not defined."""
        )

    if verbose:
        print(f"The sum of the market shares is {np.sum(x)}.")
        print("The lower the Shannon Entropy the higher the market concentration is.")

    return _shannon_entropy(x)

@njit(return_type=float)
def _theil_index(x: np.ndarray) -> float:
    """
    Computes the Theil Index using numba.
    
    Args:
        x (np.ndarray): A 1D numpy array of floats.

    Returns:
        float: The Theil Index.
    """

    mean_x=np.mean(x)
    return float(np.sum(np.where(x > 0, (x / mean_x) * np.log(x / mean_x), 0)) / x.size)

def theil_index(x: Sequence[float], verbose: bool = False) -> float:
    """
    Computes the Theil Index for market concentration.

    Args:
        x (Sequence[float]): A 1D sequence of market shares.
        verbose (bool, optional): If True, additional information is printed during the computation.
                                  Defaults to False.

    Returns:
        float: Theil Index :
                - T=0: perfect competition
                - 0<T<0.2: high compettion
                - 0.2<=T<0.5: moderate concentration
                - 0.5<=T<1: high concentration
                - 1<=T: very high concentration

    Raises:
        ValueError: If `x` is not a 1D array, is empty, contains negative values, or NaN values.
    """

    x = validate_market_shares_data(x)

    if verbose:
        print(f"The sum of the market shares is {np.sum(x)}.")
        print(
            """The lower the Theil Index is the lower the market concentration is.
               - T=0 -> perfect competition
               - 0<T<0.2 -> high competition
               - 0.2<=T<0.5 -> moderate concentration
               - 0.5<=T<1 -> high concentration
               - 1 <= T -> very high concentration"""
        )

    mean_x = np.mean(x)
    if mean_x == 0:
        if verbose:
            print("Theil Index is 0, as all market shares are 0.")
        return float(0.0)
    else:
        return _theil_index(x)