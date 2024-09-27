""" 
TO DO :
- faire en sorte que divide_by_total puisse tolérer un dictionnaire de modalités
"""

from typing import List, Optional, Tuple

# Importation des modules
# Modules de base
import numpy as np
import pandas as pd

try:
    from collections import Mapping
except:
    from collections.abc import Mapping


# Immutable dictionnary class
class FrozenDict(Mapping):
    """
    An immutable dictionary that can be hashed.
    """

    def __init__(self, *args, **kwargs) -> None:

        self._d = dict(*args, **kwargs)
        self._hash = None

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)

    def __getitem__(self, key):
        return self._d[key]

    def __hash__(self):
        if self._hash is None:
            hash_ = 0
            for pair in self.items():
                hash_ ^= hash(pair)
            self._hash = hash_
        return self._hash

    def __setitem__(self, key, value):
        raise TypeError("FrozenDict is immutable; changes are not allowed")

    def __repr__(self):
        return f"FrozenDict({self._d})"


# Fonction de division par le total
def divide_by_total(
    data_stat_des: pd.DataFrame,
    list_var_groupby: List[str],
    list_var_divide: List[str],
    list_var_of_interest: List[str],
    modality: Optional[str] = "Total",
) -> pd.DataFrame:
    """
    Divide values in the dataset by their respective total, based on specified groupby and divide-by columns.

    Parameters:
    - data_stat_des (pd.DataFrame): Input dataset containing statistics.
    - list_var_groupby (list of str): List of columns to group data by.
    - list_var_divide (list of str): List of columns used to identify the 'Total' or other reference rows.
    - list_var_of_interest (list of str): List of columns containing the values to be divided.
    - modality (str or dict, optional): Value used to identify the 'Total' or reference rows in the `list_var_divide` columns.
                                         Default is 'Total'. If it's a dictionary, keys should be columns from `list_var_divide` and
                                         values are the respective modalities to be used as reference for each column.

    Returns:
    - pd.DataFrame: The resulting dataset after dividing the specified values by their respective totals.

    Notes:
    This function is useful to compute relative statistics or proportions. Ensure that the `data_stat_des` does not contain
    NaN values in the specified columns, and the 'Total' or other reference rows are unique for each combination
    in `list_var_groupby`.
    """
    # Tolérer aussi que modality soit un dictionnaire

    # Le groupby ne doit pas être en index

    # Initialisation du jeu de données résultat
    data_res = data_stat_des.copy()

    # Construction du jeu de données par lequel diviser
    data_condition = pd.concat(
        [
            data_res[col].apply(func=lambda x: True if x == modality else False)
            for col in list_var_divide
        ],
        axis=1,
        join="outer",
    )
    data_divide = data_res.loc[
        data_condition.all(axis=1), list_var_groupby + list_var_of_interest
    ].set_index(list_var_groupby)

    data_res = data_res.set_index(list_var_groupby + list_var_divide) / data_divide

    return data_res


# Application des étoiles suivant les p-valeurs
def apply_stars(coef: str, p_value: float) -> str:
    """
    Append asterisks to a coefficient based on its p-value.

    Parameters:
    - coef (str): The coefficient to which asterisks will be appended.
    - p_value (float): The p-value corresponding to the coefficient.

    Returns:
    - str: The coefficient with appended asterisks indicating its significance level.
      Three asterisks (***) for p < 0.01, two asterisks (**) for p < 0.05,
      one asterisk (*) for p < 0.1, and no asterisks for p >= 0.1.

    Example:
    >>> apply_stars('0.25', 0.02)
    '0.25 (**)'
    """

    if p_value < 0.01:
        return coef + " (***)"
    elif p_value < 0.05:
        return coef + " (**)"
    elif p_value < 0.1:
        return coef + " (*)"
    else:
        return coef + " ()"


# Fonction associant une p-valeur à chaque coefficient du jeu de données
def convert_pvalues_to_stars(
    data_source: pd.DataFrame, col_coef: str, col_pvalues: str, is_percent: bool
) -> pd.DataFrame:
    """
    Convert p-values to asterisks and append them to the coefficients in a dataframe.

    This function takes in a dataframe and based on provided coefficient and p-value columns,
    it will convert the p-values to asterisks using the `apply_stars` function and append them
    to the coefficients. The coefficients can also be converted to percentage if required.

    Parameters:
    - data_source (pd.DataFrame): Source dataframe containing the coefficient and p-value columns.
    - col_coef (str): Name of the column containing coefficients.
    - col_pvalues (str): Name of the column containing p-values.
    - is_percent (bool): If True, coefficients are converted to percentages. Otherwise, they remain as is.

    Returns:
    - pd.DataFrame: A dataframe with a new column 'Coefficients - P-valeurs' containing coefficients
      appended with asterisks indicating significance level based on p-values.
      The coefficients are rounded and the decimal points are replaced with commas.

    Example:
    >>> df = pd.DataFrame({'coef': [0.25, 0.1], 'p_value': [0.02, 0.5]})
    >>> convert_pvalues_to_stars(df, 'coef', 'p_value', True)
       coef  p_value Coefficients - P-valeurs
    0  0.25     0.02                   25,0% (**)
    1  0.10     0.50                   10,0% ()
    """

    # Copie indépendante du jeu de données
    data_res = data_source.copy()

    # Ajout de la colonne résultats
    if is_percent:
        data_res["Coefficients - P-valeurs"] = (
            np.round(data_res[col_coef] * 100, decimals=1)
            .astype(str)
            .apply(lambda x: x + "%")
        )
    else:
        data_res["Coefficients - P-valeurs"] = np.round(
            data_res[col_coef], decimals=2
        ).astype(str)

    # Correction des points
    data_res["Coefficients - P-valeurs"] = data_res[
        "Coefficients - P-valeurs"
    ].str.replace(".", ",", regex=False)

    # Ajout des petites étoiles
    data_res["Coefficients - P-valeurs"] = data_res[
        ["Coefficients - P-valeurs", col_pvalues]
    ].apply(
        lambda x: apply_stars(
            coef=x["Coefficients - P-valeurs"], p_value=x[col_pvalues]
        ),
        axis=1,
    )

    return data_res


# Fonction permettant de compter le nombre d'individus dans la modalité de référence
def count_effectif_modalite(
    liste_fix_no_trap: List[Tuple[str, str]], data_source: pd.DataFrame, var_id: str
) -> pd.DataFrame:
    """
    Counts the number of unique individuals within a reference modality for a given list of variables.

    Parameters
    ----------
    liste_fix_no_trap : list of tuples
        List of variable-modalities pairs. Each tuple consists of a variable name and its reference modality.
    data_source : pd.DataFrame
        Source DataFrame containing the data.
    var_id : str
        Name of the variable that identifies unique individuals within `data_source`.

    Returns
    -------
    data_res : pd.DataFrame
        DataFrame containing:
        - 'variable': Variable names from `liste_fix_no_trap`.
        - 'modalite_ref': Corresponding modalities from `liste_fix_no_trap`.
        - 'nombre_individus': Count of unique individuals within each modality.
        The last row contains the intersection count for all modalities provided in `liste_fix_no_trap`.

    Notes
    -----
    If a modality for a given variable is not found in `data_source`, a warning is issued, and the modality is skipped.
    """

    # Initialisation du jeu de données résultat
    data_res = pd.DataFrame(
        data=liste_fix_no_trap, columns=["variable", "modalite_ref"]
    )

    # Initialisation du nombre d'individus correspondant à chaque modalité
    data_res["nombre_individus"] = 0
    data_intersect = data_source.copy()

    # Remplissage du jeu de données résultats
    for i, (variable, modalite_ref) in enumerate(liste_fix_no_trap):
        # Ajout du nombre d'individus
        data_res.loc[i, "nombre_individus"] = data_source.loc[
            data_source[variable] == modalite_ref, var_id
        ].nunique()
        # Réduction de l'intersection
        try:
            data_intersect = data_intersect.loc[
                data_intersect[variable] == modalite_ref
            ]
        except:
            warnings.warn(
                "Could not find the modality {} of the variable {}".format(
                    variable, modalite_ref
                )
            )
            pass

    # Ajout de l'intersection
    data_res = pd.concat(
        [
            data_res,
            pd.DataFrame(
                [["Intersection", "Intersection", data_intersect[var_id].nunique()]],
                columns=["variable", "modalite_ref", "nombre_individus"],
            ),
        ],
        axis=0,
        join="outer",
        ignore_index=True,
    )

    return data_res
