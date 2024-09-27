# Importation des modules
# Modules de base
import os
from typing import Dict, List

import numpy as np
import pandas as pd


# Fonction auxiliaire de séparation des modalités des variables catégorielles
def _separate_category_modality_from_dummy(value: Dict[str, str]) -> Dict[str, str]:
    """
    Auxiliary function to separate category names and their modalities from dummy variable names.

    Given a dummy variable name typically in the format "category - modality",
    this function separates the name into its individual category and modality components.

    Parameters:
    -----------
    value : dict
        Dictionary containing a key 'dummies' whose associated value is the dummy variable name to be split.

    Returns:
    --------
    dict
        Updated dictionary with two new keys:
        'categories' : the extracted category name
        'modalities' : the extracted modality name

    Examples:
    ---------
    >>> value = {'dummies': 'Color - Blue'}
    >>> _separate_category_modality_from_dummy(value)
    {'dummies': 'Color - Blue', 'categories': 'Color', 'modalities': 'Blue'}

    Notes:
    ------
    The function assumes the dummy variable name contains only one ' - ' separator and splits only at the first occurrence.
    """

    # Séparation selon la première occurence du caractère ' - ' de la catégorie de ses modalités
    res = value["dummies"].split(" - ", 1)

    # Complétion du jeu de données
    value["categories"] = res[0]
    value["modalities"] = res[1]

    return value


# Création d'un dictionnaire ajoutant un suffixe à chaque clé
def create_dict_suffix(list_name: List[str], suffix: str) -> Dict[str, str]:
    """
    Creates a dictionary where each key from the given list has a corresponding value with the specified suffix.

    Parameters:
    -----------
    list_name : list
        List of strings that will serve as keys in the resulting dictionary.
    suffix : str
        Suffix to append to each string in the list_name to create corresponding dictionary values.

    Returns:
    --------
    dict
        Dictionary with keys from list_name and values as original strings appended with the provided suffix.

    Example:
    --------
    >>> create_dict_suffix(['apple', 'banana'], '_fruit')
    {'apple': 'apple_fruit', 'banana': 'banana_fruit'}
    """
    # Initialisation du dictionnaire résultat
    dict_suffix = {}
    # Remplissage du dictionnaire
    for name in list_name:
        dict_suffix[name] = name + suffix
    return dict_suffix


# Fonction de tri de l'index d'un DataFrame en plaçant le "Total" à la fin
def _sort_index_with_total(data_source: pd.DataFrame) -> pd.DataFrame:
    """
    Sorts the index of a DataFrame, moving the 'Total' label to the end of each level.

    Parameters:
        data_source (pd.DataFrame): The DataFrame to be sorted.

    Returns:
        pd.DataFrame: The sorted DataFrame with 'Total' labels moved to the end of each index level.

    Note:
        This function is intended for internal use and is used to sort the index of a DataFrame by moving 'Total' labels to the end.
    """

    # Tri de chaque index
    for level in range(data_source.index.nlevels):
        if level == 0:
            data_source = _sort_level_with_total(data_source=data_source, level=level)
        else:
            data_source = data_source.groupby(level=np.arange(level).tolist()).apply(
                func=lambda x: _sort_level_with_total(data_source=x, level=level)
            )
    return data_source


# Fonction de tri d'un "level" en plçant le "Total" à la fin
def _sort_level_with_total(data_source: pd.DataFrame, level: int) -> pd.DataFrame:
    """
    Sorts a specific index level of a DataFrame, moving the 'Total' label to the end of that level.

    Parameters:
        data_source (pd.DataFrame): The DataFrame to be sorted.
        level (int): The index level to be sorted.

    Returns:
        pd.DataFrame: The DataFrame with the specified index level sorted, moving 'Total' labels to the end.

    Note:
        This function is intended for internal use and is used to sort a specific index level of a DataFrame.
    """

    # Extraction des données de total
    mask_total = data_source.index.get_level_values(level).map(
        mapper=lambda x: x == "Total"
    )
    # Tri des données qui ne sont pas des totaux et concaténation
    if level == 0:
        data_source = pd.concat(
            [
                data_source.loc[~mask_total].sort_index(level=level),
                data_source.loc[mask_total],
            ],
            axis=0,
        )
    else:
        data_source = pd.concat(
            [
                data_source.loc[~mask_total]
                .sort_index(level=level)
                .droplevel(level=np.arange(level).tolist()),
                data_source.loc[mask_total].droplevel(level=np.arange(level).tolist()),
            ],
            axis=0,
        )

    return data_source
