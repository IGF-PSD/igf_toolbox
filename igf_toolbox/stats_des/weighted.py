## Importation des modules
# Modules de base
# Modules du typage
from numbers import Number
from typing import Iterable, List, Union

import numpy as np
import pandas as pd
# Module de compilation
from numba import njit


# Fonction de calcul de l'écart-type pondéré
def weighted_std(
    values: Union[np.ndarray, pd.DataFrame, pd.Series],
    axis: Union[int, None] = None,
    weights: Union[List[Union[int, float]], None] = None,
) -> pd.Series:
    """
    Computes the weighted standard deviation for a given set of values.

    Parameters:
    - values (array-like): Input data for which the weighted standard deviation is computed.
    - axis (int, optional): Axis along which the weighted standard deviation is computed.
        - If axis=0, compute the standard deviation for each column.
        - If axis=1, compute the standard deviation for each row.
        - Default is None, which computes the standard deviation of the flattened array.
    - weights (array-like, optional): An array of weights of the same shape as `values`. Default is None, which gives equal weight to all values.

    Returns:
    - pd.Series: Series containing the computed weighted standard deviation.

    Raises:
    - ValueError: If the input values contain NaN.

    Notes:
    The function expects no NaN values in the input. Ensure NaN values are handled before calling this function.
    """

    if np.isnan(values).any():
        raise ValueError(
            "Input values contain NaN. Please remove them before computing the weighted standard deviation."
        )

    # Calcul de la moyenne pondérée
    average = np.average(values, axis=axis, weights=weights)
    # Calcul de la déviation standard
    std = np.sqrt(
        np.average(np.subtract(values, average) ** 2, axis=axis, weights=weights)
    )
    # Mise sous forme de Series
    if axis == 0:
        std = pd.Series(data=std, index=values.columns)
    elif axis == 1:
        std = pd.Series(data=std, index=values.index)

    return std


# Calcul des différents quantiles associés à plusieurs variables d'intérêt au sein d'un jeu de données
# Peut éventuellement être regroupé dans une même classe avec la fonction suivante
def weighted_quantile(
    data: pd.DataFrame,
    vars_of_interest: Union[List[str], str],
    var_weights: Union[str, None],
    q: Union[int, float, List[Union[int, float]]],
) -> pd.Series:
    """
    Compute the weighted quantile(s) for multiple variables of interest from the given dataset.

    Parameters:
    - data (pd.DataFrame): The input dataset containing the values and their respective weights.
    - vars_of_interest (list of str): List of column names in `data` representing the variables for which quantiles will be computed.
    - var_weights (str or None): Column name in `data` representing the weights of the values.
                                 If None, simple quantile will be computed for each variable of interest.
    - q (Number or array-like): Quantile or sequence of quantiles to compute, which must be between 0 and 1 inclusive.

    Returns:
    - pd.Series: The computed weighted quantile(s) for each variable of interest. The index of the series corresponds to
                 the variables of interest, and the values are the computed quantiles.

    Notes:
    This function relies on the `_weighted_quantile_array` compiled function.
    Ensure that the `data` does not contain NaN values in the specified columns.
    """

    # Retraitement des variables d'intérêt et conversion en liste
    if isinstance(vars_of_interest, str):
        vars_of_interest = [vars_of_interest]

    # Retraitement des quantiles
    if isinstance(q, Number):
        q_work = np.array([q])
    else:
        q_work = np.asarray(q)

    # Initialisation de la liste résultat
    list_res = []
    # Parcours des variables d'intérêt
    for var_of_interest in vars_of_interest:
        # Distinction de la méthode a appliquer suivant que
        if var_weights is not None:
            res_value = _weighted_quantile_array(
                array=data[[var_of_interest, var_weights]].values,
                values_pos=0,
                weights_pos=1,
                q=q_work,
            )
            if len(res_value) == 1:
                res_value = res_value[0]
        else:
            res_value = data[var_of_interest].quantile(q)
        # Ajout à la liste résultat
        list_res.append(res_value)

    return pd.Series(data=list_res, index=vars_of_interest)


# Calcul des quantiles pondérés sur des array afin d'utiliser Numba
@njit
def _weighted_quantile_array(
    array: np.ndarray, values_pos: int, weights_pos: int, q: np.ndarray
) -> np.ndarray:
    """
    Calculates weighted quantiles for a given 2D array with values and weights.

    Parameters:
    - array (2D numpy.ndarray): Input array containing values and their corresponding weights.
    - values_pos (int): Column index in `array` where the values are stored.
    - weights_pos (int): Column index in `array` where the weights are stored.
    - q (1D array-like): List or array of quantiles to compute. Values must be between 0 and 1.

    Returns:
    - numpy.ndarray: Array containing computed weighted quantiles corresponding to `q`.

    Notes:
    This function is optimized with Numba's Just-in-Time (JIT) compiler for improved performance.
    Ensure that the `array` does not contain NaN values.
    """
    # Tri selon les valeurs (en première colonne)
    array_work = array[array[:, values_pos].argsort()]
    # Somme cumulative des poids
    array_work[:, weights_pos] = np.cumsum(array_work[:, weights_pos]) / np.sum(
        array_work[:, weights_pos]
    )
    # Initialisation de l'array résultat
    res_value = np.zeros(len(q))
    for i, qi in enumerate(q):
        res_value[i] = array_work[array_work[:, weights_pos] <= qi, values_pos][-1]

    return res_value


# Association d'un quantile à chaque observation
# Peut éventuellement être regroupé dans une même classe avec la fonction suivante
def assign_quantile(
    serie: pd.Series, quantiles: List[Union[int, float]], is_threshold: bool
) -> pd.Series:
    """
    Assigns quantiles or thresholds to each observation in the input series.

    Parameters:
    - serie (pd.Series): Serie containing values for which quantiles or thresholds are to be assigned.
    - quantiles (1D array-like): List or array of quantiles or thresholds. Does not need to be sorted.
    - is_threshold (bool): If True, the function assigns thresholds to each observation in `array_values`.
                           If False, the function assigns quantile labels (e.g., 1 for first quantile, 2 for second, etc.).

    Returns:
    - pd.Series: Series containing assigned quantiles or thresholds for each observation in `serie`.

    Notes:
    This function relies on `_assign_quantile_array` optimized with Numba's Just-in-Time (JIT) compiler for improved performance.
    Ensure that the `array_values` does not contain NaN values.
    """

    return pd.Series(
        data=_assign_quantile_array(
            array_values=np.asarray(serie),
            quantiles=np.asarray(quantiles),
            is_threshold=is_threshold,
        ),
        index=serie.index,
    )


# Association d'un quantile à chaque observation sur des array afin d'utiliser Numba
@njit
def _assign_quantile_array(
    array_values: np.ndarray, quantiles: np.ndarray, is_threshold: bool
) -> np.ndarray:
    """
    Assigns quantiles or thresholds to each observation in the input array.

    Parameters:
    - array_values (1D numpy.ndarray): Array containing values for which quantiles or thresholds are to be assigned.
    - quantiles (1D array-like): List or array of quantiles or thresholds. Does not need to be sorted.
    - is_threshold (bool): If True, the function assigns thresholds to each observation in `array_values`.
                           If False, the function assigns quantile labels (e.g., 1 for first quantile, 2 for second, etc.).

    Returns:
    - numpy.ndarray: Array containing assigned quantiles or thresholds for each observation in `array_values`.

    Notes:
    This function is optimized with Numba's Just-in-Time (JIT) compiler for improved performance.
    Ensure that the `array_values` does not contain NaN values.
    """

    # Tri des quantiles
    quantiles_array = np.sort(quantiles)

    # Construction du jeu de données résultat
    if is_threshold:
        array_res = np.full(shape=len(array_values), fill_value=np.min(array_values))
    else:
        array_res = np.zeros(shape=len(array_values))

    # Application successive des quantiles (triés par ordre croissance)
    for i, qi in enumerate(quantiles_array):
        if is_threshold:
            array_res[array_values >= qi] = qi
        else:
            array_res[array_values >= qi] = i + 1

    return array_res


# Création d'un jeu de données pondéré
def create_pond_data(
    data: pd.DataFrame,
    list_var_of_interest: List[str],
    list_var_groupby: Union[List[str], None],
    var_weights: str,
) -> pd.DataFrame:
    """
    Create a weighted dataset by multiplying the variables of interest with the specified weights.

    Parameters
    ----------
    data : pandas.DataFrame
        The source dataset that contains the variables of interest, grouping variables, and weights.
    list_var_of_interest : list of str
        The list of column names in `data` that are of interest for weighting.
    list_var_groupby : list of str or None
        The list of column names in `data` used for grouping. If None, no grouping will be performed.
    var_weights : str
        The column name in `data` that contains the weights for the variables of interest.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the variables of interest weighted by `var_weights`. If `list_var_groupby` is provided,
        the returned DataFrame will also contain the groupby variables.

    Notes
    -----
    The function resets the index of the input dataframe to ensure safe merging.

    Examples
    --------
    >>> data = pd.DataFrame({
    ...     'A': [1, 2, 3],
    ...     'B': [4, 5, 6],
    ...     'weights': [0.5, 1, 1.5]
    ... })
    >>> create_pond_data(data, ['A'], None, 'weights')
       A
    0  0.5
    1  2.0
    2  4.5
    """

    # Réinitialisation de l'index pour pouvoir faire un merge "safe"
    data_work = data.copy().reset_index()

    # Multiplication des variables d'intérêt par des poids
    if list_var_groupby is not None:
        data_pond = pd.merge(
            left=data_work[list_var_groupby],
            right=data_work[list_var_of_interest].multiply(
                data_work[var_weights], axis=0
            ),
            how="outer",
            left_index=True,
            right_index=True,
            validate="one_to_one",
        )
    else:
        data_pond = data_work[list_var_of_interest].multiply(
            data_work[var_weights], axis=0
        )

    return data_pond
