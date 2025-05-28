from typing import Sequence

import numpy as np


def validate_market_shares_data(x: Sequence[float]) -> np.ndarray:
    """
    Validates a sequence of market shares by checking:
    - If it's a 1D array
    - If it's non-empty
    - If it contains only non-negative values
    - If it doesn't contain NaN values

    Args:
        x (Sequence[float]): Input sequence of market shares.

    Returns:
        np.ndarray: The validated market shares as a NumPy array.

    Raises:
        ValueError: If any validation condition is violated.
    """

    x = np.array(x, dtype=np.float64)

    # Check if x is 1D array
    if x.ndim != 1:
        raise ValueError(
            """The market shares data provided should be one-dimensional."""
        )

    # Check if x is empty
    if x.size == 0:
        raise ValueError("""Market shares array is empty.""")

    # Check if all market shares are positive
    if (x < 0).any():
        raise ValueError("""Some market shares provided are strictly negative.""")

    # Check if data is missing
    if np.isnan(x).any():
        raise ValueError("""Some market shares provided are missing (NaN values).""")

    return x