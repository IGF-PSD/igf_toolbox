from typing import Sequence

import numpy as np


def bootstrap_preprocess_data(x: Sequence[float]) -> np.ndarray:
    """
    Validates a sequence of floats for bootstraping statistics:
    - If it's a 1D array
    - If it's non-empty
    - If it doesn't contain NaN values

    Args:
        x (Sequence[float]): Input sequence of floats.

    Returns:
        np.ndarray: The validated data as a NumPy array.

    Raises:
        ValueError: If any validation condition is violated.
    """

    x = np.array(x, dtype=np.float64)

    # Check if x is 1D array
    if x.ndim != 1:
        raise ValueError(
            """The data provided should be one-dimensional."""
        )

    # Check if x is empty
    if x.size == 0:
        raise ValueError("""The array is empty.""")

    # Check if data is missing
    if np.isnan(x).any():
        raise ValueError("""Some data is (NaN values).""")

    return x