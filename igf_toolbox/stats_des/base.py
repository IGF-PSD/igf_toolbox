"""
TO DO :
- Vérifier la Gestion des Nan dans la classe StatDesGroupby
"""

from itertools import combinations
from typing import List, Optional, Union

# Importation des modules
# Modules de base
import numpy as np
import pandas as pd

# Utilitaire
from ..utils._auxiliary import _sort_index_with_total, create_dict_suffix
from .weighted import create_pond_data, weighted_quantile


# Classe de statistiques descriptives
class StatDesGroupBy(object):
    """
    A class to compute descriptive statistics grouped by specified variables.
    Compared to the pandas.DataFrame.groupby method it handles weighted operations, and can add totals.

    Attributes:
    -----------
    list_var_groupby : list
        List of variables to group data by.
    list_var_of_interest : list
        List of variables for which descriptive statistics will be computed.
    var_individu : str, optional
        Variable representing individual data.
    var_entreprise : str, optional
        Variable representing enterprise data.
    var_weights : str, optional
        Variable representing the weights for each data entry.
    data_source : DataFrame
        Data source after initialization and cleaning.

    Methods:
    --------
    iterate_with_total(iterable_operations)
        Computes descriptive statistics and returns results with subtotals and a grand total.
    iterate_without_total(iterable_operations)
        Computes descriptive statistics without subtotals or a grand total and returns results.
    add_under_total(iterable_operations)
        Generates and returns subtotals for combinations of grouping variables.
    """

    def __init__(
        self,
        data_source: pd.DataFrame,
        list_var_groupby: List[str],
        list_var_of_interest: List[str],
        var_individu: Optional[Union[str, None]] = None,
        var_entreprise: Optional[Union[str, None]] = None,
        var_weights: Optional[Union[str, None]] = None,
        dropna: Optional[bool] = False,
    ) -> None:
        """
        Initialize the StatDesGroupBy class with data and parameters.

        Parameters:
        -----------
        data_source : DataFrame
            Source data for computing statistics.
        list_var_groupby : list
            List of variables to group data by.
        list_var_of_interest : list
            List of variables for which descriptive statistics will be computed.
        var_individu : str, optional
            Variable representing individual data.
        var_entreprise : str, optional
            Variable representing enterprise data.
        var_weights : str, optional
            Variable representing the weights for each data entry.
        """
        # Initialisation des paramètres
        self.list_var_groupby = list_var_groupby
        self.list_var_of_interest = list_var_of_interest
        self.var_individu = var_individu
        self.var_entreprise = var_entreprise
        self.var_weights = var_weights

        # Initialisation de la liste des variables d'intérêt
        if var_weights is not None:
            list_var_of_interest_weights = list_var_of_interest + [var_weights]
        else:
            list_var_of_interest_weights = list_var_of_interest

        # Copie indépendante du jeu de données
        if (
            (var_individu not in list_var_groupby)
            & (var_individu is not None)
            & (var_entreprise not in list_var_groupby)
            & (var_entreprise is not None)
        ):
            list_var_keep = np.unique(
                list_var_groupby
                + list_var_of_interest_weights
                + [var_individu, var_entreprise]
            ).tolist()
        elif (
            (var_individu not in list_var_groupby)
            & (var_individu is not None)
            & ((var_entreprise in list_var_groupby) | (var_entreprise is None))
        ):
            list_var_keep = np.unique(
                list_var_groupby + list_var_of_interest_weights + [var_individu]
            ).tolist()
        elif (
            ((var_individu in list_var_groupby) | (var_individu is None))
            & (var_entreprise not in list_var_groupby)
            & (var_entreprise is not None)
        ):
            list_var_keep = np.unique(
                list_var_groupby + list_var_of_interest_weights + [var_entreprise]
            ).tolist()
        else:
            list_var_keep = np.unique(
                list_var_groupby + list_var_of_interest_weights
            ).tolist()

        # Suppression des Nan
        if dropna:
            self.data_source = data_source[list_var_keep].copy().dropna(how="any")
        else:
            self.data_source = data_source[list_var_keep].copy()

    def iterate_with_total(
        self, iterable_operations: Union[dict, List[str]]
    ) -> pd.DataFrame:
        """
        Computes descriptive statistics and returns results with subtotals and a grand total.

        Parameters:
        -----------
        iterable_operations : iterable
            Operations or functions to apply to the grouped data.

        Returns:
        --------
        DataFrame
            Descriptive statistics with subtotals and a grand total.
        """
        # Disjonction de cas suivant la longueur de la liste de groupby
        if len(self.list_var_groupby) == 0:
            data_res = self.add_total(iterable_operations=iterable_operations)
        elif len(self.list_var_groupby) == 1:
            # Itération des statistiques descriptives
            data_stat_des = self.iterate_operations(
                iterable_operations=iterable_operations,
                data=self.data_source,
                list_var_groupby=self.list_var_groupby,
            )
            # Itération du total
            data_total = self.add_total(iterable_operations=iterable_operations)
            # Concaténation des jeux de données
            data_res = pd.concat([data_stat_des, data_total], axis=0, join="outer")
        else:
            # Itération des statistiques descriptives
            data_stat_des = self.iterate_operations(
                iterable_operations=iterable_operations,
                data=self.data_source,
                list_var_groupby=self.list_var_groupby,
            )
            # Itération des sous-totaux
            data_sub_total = self.add_under_total(
                iterable_operations=iterable_operations
            )
            # Itération du total
            data_total = self.add_total(iterable_operations=iterable_operations)
            # Concaténation des jeux de données
            data_res = pd.concat(
                [data_stat_des, data_sub_total, data_total], axis=0, join="outer"
            )

        # Tri de l'indice
        data_res = _sort_index_with_total(data_source=data_res)

        return data_res

    def iterate_without_total(
        self, iterable_operations: Union[dict, List[str]]
    ) -> pd.DataFrame:
        """
        Computes descriptive statistics without subtotals or a grand total and returns results.

        Parameters:
        -----------
        iterable_operations : iterable
            Operations or functions to apply to the grouped data.

        Returns:
        --------
        DataFrame
            Descriptive statistics without subtotals or a grand total.
        """
        return self.iterate_operations(
            iterable_operations=iterable_operations,
            data=self.data_source,
            list_var_groupby=self.list_var_groupby,
        ).sort_index()

    def add_under_total(
        self, iterable_operations: Union[dict, List[str]]
    ) -> pd.DataFrame:
        """
        Generates and returns subtotals for combinations of grouping variables.

        Parameters:
        -----------
        iterable_operations : iterable
            Operations or functions to apply to the grouped data to compute subtotals.

        Returns:
        --------
        DataFrame
            Descriptive statistics with subtotals for combinations of grouping variables.
        """

        # Parcours de toutes les combinaisons possibles de niveaux
        # Création des combinaisons
        list_combinations = []
        for i in range(1, len(self.list_var_groupby)):
            list_combinations += list(
                combinations(np.arange(len(self.list_var_groupby)), r=i)
            )
        # Conversion en liste
        list_combinations = [list(combination) for combination in list_combinations]
        # Initialisation de la liste résultat
        list_sub_total = []
        for combination in list_combinations:
            if len(combination) > 0:
                # Construction de la liste avec les sous-ensembles de variables
                list_var_sub_groupby = [
                    self.list_var_groupby[e]
                    for e in range(len(self.list_var_groupby))
                    if e not in combination
                ]
                # Itération des statistiques descriptives
                data_sub_total = self.iterate_operations(
                    iterable_operations=iterable_operations,
                    data=self.data_source,
                    list_var_groupby=list_var_sub_groupby,
                )
                # Modification de l'index
                sub_total_index_res = []
                for sub_total_index in data_sub_total.index:
                    for i_remove in combination:
                        if isinstance(sub_total_index, tuple):
                            sub_total_index = (
                                list(sub_total_index)[:i_remove]
                                + ["Total"]
                                + list(sub_total_index[i_remove:])
                            )
                        elif isinstance(sub_total_index, list):
                            sub_total_index = (
                                sub_total_index[:i_remove]
                                + ["Total"]
                                + sub_total_index[i_remove:]
                            )
                        elif i_remove == 0:
                            sub_total_index = ["Total"] + [sub_total_index]
                        elif i_remove == 1:
                            sub_total_index = [sub_total_index] + ["Total"]
                        else:
                            raise ValueError("Unknown index")
                    sub_total_index_res.append(sub_total_index)
                data_sub_total.index = pd.MultiIndex.from_tuples(
                    sub_total_index_res, names=self.list_var_groupby
                )
                # Ajout à la liste résultat
                list_sub_total.append(data_sub_total)
        # Concaténation des jeux de données résultats
        data_sub_total = pd.concat(list_sub_total, axis=0, join="outer")

        return data_sub_total

    def add_total(self, iterable_operations: Union[dict, List[str]]) -> pd.DataFrame:
        """
        Perform a series of global aggregate operations on the data and concatenate the results.

        The function goes through the specified operations (either as strings for basic operations or tuples for
        operations with extra parameters) and applies them on the data_source. The result of each operation is
        appended to a list. At the end, the results are concatenated and returned as a single DataFrame.

        Parameters:
        - iterable_operations (dict or list): Operations to be applied on the data. If a dictionary, the keys
        represent the operation and values represent variables of interest. If a list, it contains either
        operations as strings or tuples where the first element is the operation and the second is a dictionary
        of parameters.

        Returns:
        - pd.DataFrame: A DataFrame containing aggregated results after applying all the operations, indexed by
        either 'Total' or a MultiIndex version of it based on the length of list_var_groupby.

        Raises:
        - ValueError: If the provided type for iterable_operations is neither dict nor list.

        Example:
        ```python
        # Example usage (assuming it's a method within a class)
        aggregated_data = StatDesGroupby.add_total({'sum': ['var1', 'var2']})
        ```

        Notes:
        - The function uses internal methods of the class, such as sum, mean, median, etc.
        - The operations are performed without groupby, intended to get overall aggregates.
        - The 'count_effectif' and 'max_sum_effectif' operations have special handling based on the presence
        of var_individu and var_entreprise class attributes.
        """

        # Initialisation de la liste résultat
        list_total = []
        # Parcours des différentes opérations
        # Faire avec agg pour limiter le nombre de groupby
        if isinstance(iterable_operations, dict):
            for operation, list_var_of_interest_work in iterable_operations.items():
                if isinstance(operation, str):
                    if operation == "sum":
                        list_total.append(
                            self.sum(
                                data=self.data_source,
                                list_var_groupby=None,
                                list_var_of_interest=np.intersect1d(
                                    ar1=self.list_var_of_interest,
                                    ar2=list_var_of_interest_work,
                                ).tolist(),
                                var_weights=self.var_weights,
                            )
                        )
                    elif operation == "mean":
                        list_total.append(
                            self.mean(
                                data=self.data_source,
                                list_var_groupby=None,
                                list_var_of_interest=np.intersect1d(
                                    ar1=self.list_var_of_interest,
                                    ar2=list_var_of_interest_work,
                                ).tolist(),
                                var_weights=self.var_weights,
                            )
                        )
                    elif operation == "median":
                        list_total.append(
                            self.median(
                                data=self.data_source,
                                list_var_groupby=None,
                                list_var_of_interest=np.intersect1d(
                                    ar1=self.list_var_of_interest,
                                    ar2=list_var_of_interest_work,
                                ).tolist(),
                                var_weights=self.var_weights,
                            )
                        )
                    elif operation == "nunique":
                        list_total.append(
                            self.nunique(
                                data=self.data_source,
                                list_var_groupby=None,
                                list_var_of_interest=np.intersect1d(
                                    ar1=self.list_var_of_interest,
                                    ar2=list_var_of_interest_work,
                                ).tolist(),
                            )
                        )
                    elif operation == "count":
                        list_total.append(
                            self.count(
                                data=self.data_source,
                                list_var_groupby=None,
                                list_var_of_interest=np.intersect1d(
                                    ar1=self.list_var_of_interest,
                                    ar2=list_var_of_interest_work,
                                ).tolist(),
                            )
                        )
                    elif operation == "any":
                        list_total.append(
                            self.any(
                                data=self.data_source,
                                list_var_groupby=None,
                                list_var_of_interest=np.intersect1d(
                                    ar1=self.list_var_of_interest,
                                    ar2=list_var_of_interest_work,
                                ).tolist(),
                            )
                        )
                    elif operation == "all":
                        list_total.append(
                            self.all(
                                data=self.data_source,
                                list_var_groupby=None,
                                list_var_of_interest=np.intersect1d(
                                    ar1=self.list_var_of_interest,
                                    ar2=list_var_of_interest_work,
                                ).tolist(),
                            )
                        )
                    elif operation == "majority":
                        list_total.append(
                            self.majority(
                                data=self.data_source,
                                list_var_groupby=None,
                                list_var_of_interest=np.intersect1d(
                                    ar1=self.list_var_of_interest,
                                    ar2=list_var_of_interest_work,
                                ).tolist(),
                                var_weights=self.var_weights,
                            )
                        )
                    elif (
                        (operation == "count_effectif")
                        & (self.var_individu is not None)
                        & (self.var_entreprise is not None)
                    ):
                        list_total.append(
                            self.nunique(
                                data=self.data_source,
                                list_var_groupby=None,
                                list_var_of_interest=[
                                    self.var_individu,
                                    self.var_entreprise,
                                ],
                            )
                        )
                    elif (
                        (operation == "count_effectif")
                        & (self.var_individu is not None)
                        & (self.var_entreprise is None)
                    ):
                        list_total.append(
                            self.nunique(
                                data=self.data_source,
                                list_var_groupby=None,
                                list_var_of_interest=[self.var_individu],
                            )
                        )
                    elif (
                        (operation == "count_effectif")
                        & (self.var_individu is None)
                        & (self.var_entreprise is not None)
                    ):
                        list_total.append(
                            self.nunique(
                                data=self.data_source,
                                list_var_groupby=None,
                                list_var_of_interest=[self.var_entreprise],
                            )
                        )
                    elif (
                        (operation == "max_sum_effectif")
                        & (self.var_individu is not None)
                        & (self.var_entreprise is not None)
                    ):
                        list_total.append(
                            self.max_sum_effectif(
                                data=self.data_source,
                                list_var_groupby=None,
                                list_var_of_interest=np.intersect1d(
                                    ar1=self.list_var_of_interest,
                                    ar2=list_var_of_interest_work,
                                ).tolist(),
                                var_weights=self.var_weights,
                                var_id=self.var_individu,
                            )
                        )
                        list_total.append(
                            self.max_sum_effectif(
                                data=self.data_source,
                                list_var_groupby=None,
                                list_var_of_interest=np.intersect1d(
                                    ar1=self.list_var_of_interest,
                                    ar2=list_var_of_interest_work,
                                ).tolist(),
                                var_weights=self.var_weights,
                                var_id=self.var_entreprise,
                            )
                        )
                    elif (
                        (operation == "max_sum_effectif")
                        & (self.var_individu is not None)
                        & (self.var_entreprise is None)
                    ):
                        list_total.append(
                            self.max_sum_effectif(
                                data=self.data_source,
                                list_var_groupby=None,
                                list_var_of_interest=np.intersect1d(
                                    ar1=self.list_var_of_interest,
                                    ar2=list_var_of_interest_work,
                                ).tolist(),
                                var_weights=self.var_weights,
                                var_id=self.var_individu,
                            )
                        )
                    elif (
                        (operation == "max_sum_effectif")
                        & (self.var_individu is None)
                        & (self.var_entreprise is not None)
                    ):
                        list_total.append(
                            self.max_sum_effectif(
                                data=self.data_source,
                                list_var_groupby=None,
                                list_var_of_interest=np.intersect1d(
                                    ar1=self.list_var_of_interest,
                                    ar2=list_var_of_interest_work,
                                ).tolist(),
                                var_weights=self.var_weights,
                                var_id=self.var_entreprise,
                            )
                        )
                elif isinstance(operation, tuple):
                    # Le tuple est composé d'un string en première position et d'un dictionnaire de paramètres en deuxième
                    if operation[0] == "quantile":
                        list_total.append(
                            self.quantile(
                                data=self.data_source,
                                list_var_groupby=None,
                                list_var_of_interest=np.intersect1d(
                                    ar1=self.list_var_of_interest,
                                    ar2=list_var_of_interest_work,
                                ).tolist(),
                                var_weights=self.var_weights,
                                q=operation[1]["q"],
                            )
                        )
                    elif operation[0] == "prop":
                        list_total.append(
                            self.prop(
                                data=self.data_source,
                                list_var_groupby=None,
                                list_var_of_interest=np.intersect1d(
                                    ar1=self.list_var_of_interest,
                                    ar2=list_var_of_interest_work,
                                ).tolist(),
                                var_weights=self.var_weights,
                                var_ref=operation[1]["var_ref"],
                            )
                        )
                    elif operation[0] == "inf_threshold":
                        list_total.append(
                            self.inf_seuil(
                                data=self.data_source,
                                list_var_groupby=None,
                                list_var_of_interest=np.intersect1d(
                                    ar1=self.list_var_of_interest,
                                    ar2=list_var_of_interest_work,
                                ).tolist(),
                                var_threshold=operation[1]["var_threshold"],
                                seuil=operation[1]["threshold"],
                            )
                        )
        elif isinstance(iterable_operations, list):
            for operation in iterable_operations:
                if isinstance(operation, str):
                    if operation == "sum":
                        list_total.append(
                            self.sum(
                                data=self.data_source,
                                list_var_groupby=None,
                                list_var_of_interest=self.list_var_of_interest,
                                var_weights=self.var_weights,
                            )
                        )
                    elif operation == "mean":
                        list_total.append(
                            self.mean(
                                data=self.data_source,
                                list_var_groupby=None,
                                list_var_of_interest=self.list_var_of_interest,
                                var_weights=self.var_weights,
                            )
                        )
                    elif operation == "median":
                        list_total.append(
                            self.median(
                                data=self.data_source,
                                list_var_groupby=None,
                                list_var_of_interest=self.list_var_of_interest,
                                var_weights=self.var_weights,
                            )
                        )
                    elif operation == "nunique":
                        list_total.append(
                            self.nunique(
                                data=self.data_source,
                                list_var_groupby=None,
                                list_var_of_interest=self.list_var_of_interest,
                            )
                        )
                    elif operation == "count":
                        list_total.append(
                            self.count(
                                data=self.data_source,
                                list_var_groupby=None,
                                list_var_of_interest=self.list_var_of_interest,
                            )
                        )
                    elif operation == "any":
                        list_total.append(
                            self.any(
                                data=self.data_source,
                                list_var_groupby=None,
                                list_var_of_interest=self.list_var_of_interest,
                            )
                        )
                    elif operation == "all":
                        list_total.append(
                            self.all(
                                data=self.data_source,
                                list_var_groupby=None,
                                list_var_of_interest=self.list_var_of_interest,
                            )
                        )
                    elif operation == "majority":
                        list_total.append(
                            self.majority(
                                data=self.data_source,
                                list_var_groupby=None,
                                list_var_of_interest=self.list_var_of_interest,
                                var_weights=self.var_weights,
                            )
                        )
                    elif (
                        (operation == "count_effectif")
                        & (self.var_individu is not None)
                        & (self.var_entreprise is not None)
                    ):
                        list_total.append(
                            self.nunique(
                                data=self.data_source,
                                list_var_groupby=None,
                                list_var_of_interest=[
                                    self.var_individu,
                                    self.var_entreprise,
                                ],
                            )
                        )
                    elif (
                        (operation == "count_effectif")
                        & (self.var_individu is not None)
                        & (self.var_entreprise is None)
                    ):
                        list_total.append(
                            self.nunique(
                                data=self.data_source,
                                list_var_groupby=None,
                                list_var_of_interest=[self.var_individu],
                            )
                        )
                    elif (
                        (operation == "count_effectif")
                        & (self.var_individu is None)
                        & (self.var_entreprise is not None)
                    ):
                        list_total.append(
                            self.nunique(
                                data=self.data_source,
                                list_var_groupby=None,
                                list_var_of_interest=[self.var_entreprise],
                            )
                        )
                    elif (
                        (operation == "max_sum_effectif")
                        & (self.var_individu is not None)
                        & (self.var_entreprise is not None)
                    ):
                        list_total.append(
                            self.max_sum_effectif(
                                data=self.data_source,
                                list_var_groupby=None,
                                list_var_of_interest=self.list_var_of_interest,
                                var_weights=self.var_weights,
                                var_id=self.var_individu,
                            )
                        )
                        list_total.append(
                            self.max_sum_effectif(
                                data=self.data_source,
                                list_var_groupby=None,
                                list_var_of_interest=self.list_var_of_interest,
                                var_weights=self.var_weights,
                                var_id=self.var_entreprise,
                            )
                        )
                    elif (
                        (operation == "max_sum_effectif")
                        & (self.var_individu is not None)
                        & (self.var_entreprise is None)
                    ):
                        list_total.append(
                            self.max_sum_effectif(
                                data=self.data_source,
                                list_var_groupby=None,
                                list_var_of_interest=self.list_var_of_interest,
                                var_weights=self.var_weights,
                                var_id=self.var_individu,
                            )
                        )
                    elif (
                        (operation == "max_sum_effectif")
                        & (self.var_individu is None)
                        & (self.var_entreprise is not None)
                    ):
                        list_total.append(
                            self.max_sum_effectif(
                                data=self.data_source,
                                list_var_groupby=None,
                                list_var_of_interest=self.list_var_of_interest,
                                var_weights=self.var_weights,
                                var_id=self.var_entreprise,
                            )
                        )
                elif isinstance(operation, tuple):
                    # Le tuple est composé d'un string en première position et d'un dictionnaire de paramètres en deuxième
                    if operation[0] == "quantile":
                        list_total.append(
                            self.quantile(
                                data=self.data_source,
                                list_var_groupby=None,
                                list_var_of_interest=self.list_var_of_interest,
                                var_weights=self.var_weights,
                                q=operation[1]["q"],
                            )
                        )
                    elif operation[0] == "prop":
                        list_total.append(
                            self.prop(
                                data=self.data_source,
                                list_var_groupby=None,
                                list_var_of_interest=self.list_var_of_interest,
                                var_weights=self.var_weights,
                                var_ref=operation[1]["var_ref"],
                            )
                        )
                    elif operation[0] == "inf_threshold":
                        list_total.append(
                            self.inf_seuil(
                                data=self.data_source,
                                list_var_groupby=None,
                                list_var_of_interest=self.list_var_of_interest,
                                var_threshold=operation[1]["var_threshold"],
                                seuil=operation[1]["threshold"],
                            )
                        )
        else:
            raise ValueError("Unsupported type for iterable_operations")

        # Concaténation des jeux de données
        data_total = pd.concat(list_total, axis=1, join="outer")

        if len(self.list_var_groupby) > 1:
            data_total.index = pd.MultiIndex.from_tuples(
                [["Total"] * len(self.list_var_groupby)], names=self.list_var_groupby
            )
        else:
            data_total.index = ["Total"]

        return data_total

    def iterate_operations(
        self,
        iterable_operations: Union[dict, List[str]],
        data: pd.DataFrame,
        list_var_groupby: List[str],
    ) -> pd.DataFrame:
        """
        Iteratively perform a set of operations on the data grouped by given variables.

        This function first creates a nomenclature with categorical variables and then, based on the type
        of operations provided, executes those operations on the dataset. It utilizes the internal methods
        of the class to which this function belongs, such as sum, mean, median, etc.

        Parameters:
        - iterable_operations (dict or list): Operations to be applied on the data. If it's a dictionary, the
        keys represent the operation and values represent the variables of interest. If it's a list, it only
        contains operations.
        - data (pd.DataFrame): Input dataset on which the operations need to be applied.
        - list_var_groupby (list of str): List of variables based on which the data needs to be grouped.

        Returns:
        - pd.DataFrame: Dataset after applying all the operations, indexed by the list_var_groupby.

        Raises:
        - ValueError: If the provided type for iterable_operations is neither dict nor list.

        Example:
        ```python
        # Example usage (assuming it's a method within a class)
        result = StatDesGroupby.iterate_operations({'sum': ['var1', 'var2']}, data, ['group_var'])
        ```

        Notes:
        - The function merges the resultant data with the data_nomenc to ensure the categorical variables are retained.
        - The internal operations like sum, mean, etc., are methods of the same class and are not global functions.
        """
        # Création d'une nomenclature avec variables catégorielles
        data_nomenc = data[list_var_groupby].drop_duplicates().reset_index()
        # Conversion de la colonne en variable catégorielle
        data_nomenc["index"] = pd.Categorical(data_nomenc["index"])
        # Ajout au jeu de données
        data_work = pd.merge(
            left=data,
            right=data_nomenc,
            how="left",
            on=list_var_groupby,
            validate="many_to_one",
        ).drop(list_var_groupby, axis=1)

        # Initialisation de la liste résultat
        list_res = []
        # Parcours des différentes opérations
        # Faire avec agg pour limiter le nombre de groupby
        if isinstance(iterable_operations, dict):
            for operation, list_var_of_interest_work in iterable_operations.items():
                if isinstance(operation, str):
                    if operation == "sum":
                        list_res.append(
                            self.sum(
                                data=data_work,
                                list_var_groupby=["index"],
                                list_var_of_interest=np.intersect1d(
                                    ar1=self.list_var_of_interest,
                                    ar2=list_var_of_interest_work,
                                ).tolist(),
                                var_weights=self.var_weights,
                            )
                        )
                    elif operation == "mean":
                        list_res.append(
                            self.mean(
                                data=data_work,
                                list_var_groupby=["index"],
                                list_var_of_interest=np.intersect1d(
                                    ar1=self.list_var_of_interest,
                                    ar2=list_var_of_interest_work,
                                ).tolist(),
                                var_weights=self.var_weights,
                            )
                        )
                    elif operation == "median":
                        list_res.append(
                            self.median(
                                data=data_work,
                                list_var_groupby=["index"],
                                list_var_of_interest=np.intersect1d(
                                    ar1=self.list_var_of_interest,
                                    ar2=list_var_of_interest_work,
                                ).tolist(),
                                var_weights=self.var_weights,
                            )
                        )
                    elif operation == "nunique":
                        list_res.append(
                            self.nunique(
                                data=data_work,
                                list_var_groupby=["index"],
                                list_var_of_interest=np.intersect1d(
                                    ar1=self.list_var_of_interest,
                                    ar2=list_var_of_interest_work,
                                ).tolist(),
                            )
                        )
                    elif operation == "count":
                        list_res.append(
                            self.count(
                                data=data_work,
                                list_var_groupby=["index"],
                                list_var_of_interest=np.intersect1d(
                                    ar1=self.list_var_of_interest,
                                    ar2=list_var_of_interest_work,
                                ).tolist(),
                            )
                        )
                    elif operation == "any":
                        list_res.append(
                            self.any(
                                data=data_work,
                                list_var_groupby=["index"],
                                list_var_of_interest=np.intersect1d(
                                    ar1=self.list_var_of_interest,
                                    ar2=list_var_of_interest_work,
                                ).tolist(),
                            )
                        )
                    elif operation == "all":
                        list_res.append(
                            self.all(
                                data=data_work,
                                list_var_groupby=["index"],
                                list_var_of_interest=np.intersect1d(
                                    ar1=self.list_var_of_interest,
                                    ar2=list_var_of_interest_work,
                                ).tolist(),
                            )
                        )
                    elif operation == "majority":
                        list_res.append(
                            self.majority(
                                data=data_work,
                                list_var_groupby=["index"],
                                list_var_of_interest=np.intersect1d(
                                    ar1=self.list_var_of_interest,
                                    ar2=list_var_of_interest_work,
                                ).tolist(),
                                var_weights=self.var_weights,
                            )
                        )
                    elif (
                        (operation == "count_effectif")
                        & (self.var_individu is not None)
                        & (self.var_entreprise is not None)
                    ):
                        list_res.append(
                            self.nunique(
                                data=data_work,
                                list_var_groupby=["index"],
                                list_var_of_interest=[
                                    self.var_individu,
                                    self.var_entreprise,
                                ],
                            )
                        )
                    elif (
                        (operation == "count_effectif")
                        & (self.var_individu is not None)
                        & (self.var_entreprise is None)
                    ):
                        list_res.append(
                            self.nunique(
                                data=data_work,
                                list_var_groupby=["index"],
                                list_var_of_interest=[self.var_individu],
                            )
                        )
                    elif (
                        (operation == "count_effectif")
                        & (self.var_individu is None)
                        & (self.var_entreprise is not None)
                    ):
                        list_res.append(
                            self.nunique(
                                data=data_work,
                                list_var_groupby=["index"],
                                list_var_of_interest=[self.var_entreprise],
                            )
                        )
                    elif (
                        (operation == "max_sum_effectif")
                        & (self.var_individu is not None)
                        & (self.var_entreprise is not None)
                    ):
                        list_res.append(
                            self.max_sum_effectif(
                                data=data_work,
                                list_var_groupby=["index"],
                                list_var_of_interest=np.intersect1d(
                                    ar1=self.list_var_of_interest,
                                    ar2=list_var_of_interest_work,
                                ).tolist(),
                                var_weights=self.var_weights,
                                var_id=self.var_individu,
                            )
                        )
                        list_res.append(
                            self.max_sum_effectif(
                                data=data_work,
                                list_var_groupby=["index"],
                                list_var_of_interest=np.intersect1d(
                                    ar1=self.list_var_of_interest,
                                    ar2=list_var_of_interest_work,
                                ).tolist(),
                                var_weights=self.var_weights,
                                var_id=self.var_entreprise,
                            )
                        )
                    elif (
                        (operation == "max_sum_effectif")
                        & (self.var_individu is not None)
                        & (self.var_entreprise is None)
                    ):
                        list_res.append(
                            self.max_sum_effectif(
                                data=data_work,
                                list_var_groupby=["index"],
                                list_var_of_interest=np.intersect1d(
                                    ar1=self.list_var_of_interest,
                                    ar2=list_var_of_interest_work,
                                ).tolist(),
                                var_weights=self.var_weights,
                                var_id=self.var_individu,
                            )
                        )
                    elif (
                        (operation == "max_sum_effectif")
                        & (self.var_individu is None)
                        & (self.var_entreprise is not None)
                    ):
                        list_res.append(
                            self.max_sum_effectif(
                                data=data_work,
                                list_var_groupby=["index"],
                                list_var_of_interest=np.intersect1d(
                                    ar1=self.list_var_of_interest,
                                    ar2=list_var_of_interest_work,
                                ).tolist(),
                                var_weights=self.var_weights,
                                var_id=self.var_entreprise,
                            )
                        )
                elif isinstance(operation, tuple):
                    # Le tuple est composé d'un string en première position et d'un dictionnaire de paramètres en deuxième
                    if operation[0] == "quantile":
                        list_res.append(
                            self.quantile(
                                data=data_work,
                                list_var_groupby=["index"],
                                list_var_of_interest=np.intersect1d(
                                    ar1=self.list_var_of_interest,
                                    ar2=list_var_of_interest_work,
                                ).tolist(),
                                var_weights=self.var_weights,
                                q=operation[1]["q"],
                            )
                        )
                    elif operation[0] == "prop":
                        list_res.append(
                            self.prop(
                                data=data_work,
                                list_var_groupby=["index"],
                                list_var_of_interest=np.intersect1d(
                                    ar1=self.list_var_of_interest,
                                    ar2=list_var_of_interest_work,
                                ).tolist(),
                                var_weights=self.var_weights,
                                var_ref=operation[1]["var_ref"],
                            )
                        )
                    elif operation[0] == "inf_threshold":
                        list_res.append(
                            self.inf_seuil(
                                data=data_work,
                                list_var_groupby=["index"],
                                list_var_of_interest=np.intersect1d(
                                    ar1=self.list_var_of_interest,
                                    ar2=list_var_of_interest_work,
                                ).tolist(),
                                var_threshold=operation[1]["var_threshold"],
                                seuil=operation[1]["threshold"],
                            )
                        )
        elif isinstance(iterable_operations, list):
            for operation in iterable_operations:
                if isinstance(operation, str):
                    if operation == "sum":
                        list_res.append(
                            self.sum(
                                data=data_work,
                                list_var_groupby=["index"],
                                list_var_of_interest=self.list_var_of_interest,
                                var_weights=self.var_weights,
                            )
                        )
                    elif operation == "mean":
                        list_res.append(
                            self.mean(
                                data=data_work,
                                list_var_groupby=["index"],
                                list_var_of_interest=self.list_var_of_interest,
                                var_weights=self.var_weights,
                            )
                        )
                    elif operation == "median":
                        list_res.append(
                            self.median(
                                data=data_work,
                                list_var_groupby=["index"],
                                list_var_of_interest=self.list_var_of_interest,
                                var_weights=self.var_weights,
                            )
                        )
                    elif operation == "nunique":
                        list_res.append(
                            self.nunique(
                                data=data_work,
                                list_var_groupby=["index"],
                                list_var_of_interest=self.list_var_of_interest,
                            )
                        )
                    elif operation == "count":
                        list_res.append(
                            self.count(
                                data=data_work,
                                list_var_groupby=["index"],
                                list_var_of_interest=self.list_var_of_interest,
                            )
                        )
                    elif operation == "any":
                        list_res.append(
                            self.any(
                                data=data_work,
                                list_var_groupby=["index"],
                                list_var_of_interest=self.list_var_of_interest,
                            )
                        )
                    elif operation == "all":
                        list_res.append(
                            self.all(
                                data=data_work,
                                list_var_groupby=["index"],
                                list_var_of_interest=self.list_var_of_interest,
                            )
                        )
                    elif operation == "majority":
                        list_res.append(
                            self.majority(
                                data=data_work,
                                list_var_groupby=["index"],
                                list_var_of_interest=self.list_var_of_interest,
                                var_weights=self.var_weights,
                            )
                        )
                    elif (
                        (operation == "count_effectif")
                        & (self.var_individu is not None)
                        & (self.var_entreprise is not None)
                    ):
                        list_res.append(
                            self.nunique(
                                data=data_work,
                                list_var_groupby=["index"],
                                list_var_of_interest=[
                                    self.var_individu,
                                    self.var_entreprise,
                                ],
                            )
                        )
                    elif (
                        (operation == "count_effectif")
                        & (self.var_individu is not None)
                        & (self.var_entreprise is None)
                    ):
                        list_res.append(
                            self.nunique(
                                data=data_work,
                                list_var_groupby=["index"],
                                list_var_of_interest=[self.var_individu],
                            )
                        )
                    elif (
                        (operation == "count_effectif")
                        & (self.var_individu is None)
                        & (self.var_entreprise is not None)
                    ):
                        list_res.append(
                            self.nunique(
                                data=data_work,
                                list_var_groupby=["index"],
                                list_var_of_interest=[self.var_entreprise],
                            )
                        )
                    elif (
                        (operation == "max_sum_effectif")
                        & (self.var_individu is not None)
                        & (self.var_entreprise is not None)
                    ):
                        list_res.append(
                            self.max_sum_effectif(
                                data=data_work,
                                list_var_groupby=["index"],
                                list_var_of_interest=self.list_var_of_interest,
                                var_weights=self.var_weights,
                                var_id=self.var_individu,
                            )
                        )
                        list_res.append(
                            self.max_sum_effectif(
                                data=data_work,
                                list_var_groupby=["index"],
                                list_var_of_interest=self.list_var_of_interest,
                                var_weights=self.var_weights,
                                var_id=self.var_entreprise,
                            )
                        )
                    elif (
                        (operation == "max_sum_effectif")
                        & (self.var_individu is not None)
                        & (self.var_entreprise is None)
                    ):
                        list_res.append(
                            self.max_sum_effectif(
                                data=data_work,
                                list_var_groupby=["index"],
                                list_var_of_interest=self.list_var_of_interest,
                                var_weights=self.var_weights,
                                var_id=self.var_individu,
                            )
                        )
                    elif (
                        (operation == "max_sum_effectif")
                        & (self.var_individu is None)
                        & (self.var_entreprise is not None)
                    ):
                        list_res.append(
                            self.max_sum_effectif(
                                data=data_work,
                                list_var_groupby=["index"],
                                list_var_of_interest=self.list_var_of_interest,
                                var_weights=self.var_weights,
                                var_id=self.var_entreprise,
                            )
                        )
                elif isinstance(operation, tuple):
                    # Le tuple est composé d'un string en première position et d'un dictionnaire de paramètres en deuxième
                    if operation[0] == "quantile":
                        list_res.append(
                            self.quantile(
                                data=data_work,
                                list_var_groupby=["index"],
                                list_var_of_interest=self.list_var_of_interest,
                                var_weights=self.var_weights,
                                q=operation[1]["q"],
                            )
                        )
                    elif operation[0] == "prop":
                        list_res.append(
                            self.prop(
                                data=data_work,
                                list_var_groupby=["index"],
                                list_var_of_interest=self.list_var_of_interest,
                                var_weights=self.var_weights,
                                var_ref=operation[1]["var_ref"],
                            )
                        )
                    elif operation[0] == "inf_threshold":
                        list_res.append(
                            self.inf_seuil(
                                data=data_work,
                                list_var_groupby=["index"],
                                list_var_of_interest=self.list_var_of_interest,
                                var_threshold=operation[1]["var_threshold"],
                                seuil=operation[1]["threshold"],
                            )
                        )
        else:
            raise ValueError("Unsupported type for iterable_operations")

        # Construction du jeu de données résultat
        data_res = (
            pd.merge(
                left=pd.concat(list_res, axis=1, join="outer").reset_index(),
                right=data_nomenc,
                how="left",
                on="index",
                validate="one_to_one",
            )
            .drop("index", axis=1)
            .set_index(list_var_groupby)
        )

        return data_res

    def nunique(
        self,
        data: pd.DataFrame,
        list_var_groupby: Union[List[str], None],
        list_var_of_interest: List[str],
    ) -> pd.DataFrame:
        """
        Calculate the number of unique values in the given columns of a DataFrame, optionally grouped by specific columns.

        Parameters:
        - data (pd.DataFrame): Input DataFrame to compute the number of unique values on.
        - list_var_groupby (List[str], optional): List of column names to group by. If None, no grouping is performed.
        - list_var_of_interest (List[str]): List of column names for which the number of unique values is computed.

        Returns:
        - pd.DataFrame: A DataFrame containing the count of unique values. The output column names are appended with "_nunique".

        Example:
        ```
        df = pd.DataFrame({
            'A': ['a', 'b', 'a', 'c'],
            'B': [1, 2, 3, 4]
        })
        nunique(df, list_var_groupby=['A'], list_var_of_interest=['B'])
        ```

        """
        if list_var_groupby is not None:
            return (
                data.groupby(list_var_groupby, as_index=True, observed=True)[
                    list_var_of_interest
                ]
                .nunique()
                .rename(
                    create_dict_suffix(
                        list_name=list_var_of_interest, suffix="_nunique"
                    ),
                    axis=1,
                )
            )
        else:
            return (
                data[list_var_of_interest]
                .nunique()
                .to_frame()
                .transpose()
                .rename(
                    create_dict_suffix(
                        list_name=list_var_of_interest, suffix="_nunique"
                    ),
                    axis=1,
                )
            )

    def count(
        self,
        data: pd.DataFrame,
        list_var_groupby: Union[List[str], None],
        list_var_of_interest: List[str],
    ) -> pd.DataFrame:
        """
        Count the number of non-missing values in the given columns of a DataFrame, optionally grouped by specific columns.

        Parameters:
        - data (pd.DataFrame): Input DataFrame to compute the count on.
        - list_var_groupby (List[str], optional): List of column names to group by. If None, no grouping is performed.
        - list_var_of_interest (List[str]): List of column names for which the count is computed.

        Returns:
        - pd.DataFrame: A DataFrame containing the count. The output column names are appended with "_count".

        Example:
        ```
        df = pd.DataFrame({
            'A': ['a', 'b', 'a', None],
            'B': [1, 2, 3, 4]
        })
        count(df, list_var_groupby=['A'], list_var_of_interest=['B'])
        ```

        """
        if list_var_groupby is not None:
            return (
                data.groupby(list_var_groupby, as_index=True, observed=True)[
                    list_var_of_interest
                ]
                .count()
                .rename(
                    create_dict_suffix(list_name=list_var_of_interest, suffix="_count"),
                    axis=1,
                )
            )
        else:
            return (
                data[list_var_of_interest]
                .count()
                .to_frame()
                .transpose()
                .rename(
                    create_dict_suffix(list_name=list_var_of_interest, suffix="_count"),
                    axis=1,
                )
            )

    def any(
        self,
        data: pd.DataFrame,
        list_var_groupby: Union[List[str], None],
        list_var_of_interest: List[str],
    ) -> pd.DataFrame:
        """
        Determine if any element in the given columns of a DataFrame is True, optionally grouped by specific columns.

        Parameters:
        - data (pd.DataFrame): Input DataFrame to check.
        - list_var_groupby (List[str], optional): List of column names to group by. If None, no grouping is performed.
        - list_var_of_interest (List[str]): List of column names to check for any True values.

        Returns:
        - pd.DataFrame: A DataFrame indicating if any value is True. The output column names are appended with "_any".

        Example:
        ```
        df = pd.DataFrame({
            'A': ['a', 'b', 'a', 'c'],
            'B': [True, False, True, False]
        })
        any(df, list_var_groupby=['A'], list_var_of_interest=['B'])
        ```

        """
        if list_var_groupby is not None:
            return (
                data.groupby(list_var_groupby, as_index=True, observed=True)[
                    list_var_of_interest
                ]
                .any()
                .rename(
                    create_dict_suffix(list_name=list_var_of_interest, suffix="_any"),
                    axis=1,
                )
            )
        else:
            return (
                data[list_var_of_interest]
                .any()
                .to_frame()
                .transpose()
                .rename(
                    create_dict_suffix(list_name=list_var_of_interest, suffix="_any"),
                    axis=1,
                )
            )

    def all(
        self,
        data: pd.DataFrame,
        list_var_groupby: Union[List[str], None],
        list_var_of_interest: List[str],
    ) -> pd.DataFrame:
        """
        Determine if all elements in the given columns of a DataFrame are True, optionally grouped by specific columns.

        Parameters:
        - data (pd.DataFrame): Input DataFrame to check.
        - list_var_groupby (List[str], optional): List of column names to group by. If None, no grouping is performed.
        - list_var_of_interest (List[str]): List of column names to check if all values are True.

        Returns:
        - pd.DataFrame: A DataFrame indicating if all values are True. The output column names are appended with "_all".

        Example:
        ```
        df = pd.DataFrame({
            'A': ['a', 'b', 'a', 'c'],
            'B': [True, True, True, False]
        })
        all(df, list_var_groupby=['A'], list_var_of_interest=['B'])
        ```

        """
        if list_var_groupby is not None:
            return (
                data.groupby(list_var_groupby, as_index=True, observed=True)[
                    list_var_of_interest
                ]
                .all()
                .rename(
                    create_dict_suffix(list_name=list_var_of_interest, suffix="_all"),
                    axis=1,
                )
            )
        else:
            return (
                data[list_var_of_interest]
                .all()
                .to_frame()
                .transpose()
                .rename(
                    create_dict_suffix(list_name=list_var_of_interest, suffix="_all"),
                    axis=1,
                )
            )

    def sum(
        self,
        data: pd.DataFrame,
        list_var_groupby: Union[List[str], None],
        list_var_of_interest: List[str],
        var_weights: Union[str, None],
    ) -> pd.DataFrame:
        """
        Compute the sum aggregation on a given pandas DataFrame, with options for weighted sum.

        Parameters:
        - data (pd.DataFrame): The input DataFrame to compute aggregation on.
        - list_var_groupby (List[str], optional): List of column names to group by. If None, no grouping is performed.
        - list_var_of_interest (List[str]): List of column names on which the aggregation is performed.
        - var_weights (str, optional): Column name representing the weights for weighted sum.
                                    If it is in list_var_of_interest, the function will compute both weighted
                                    and non-weighted sum. If None, a simple sum is performed.

        Returns:
        - pd.DataFrame: A DataFrame containing the aggregated data. The output DataFrame will have a multi-index if
                        list_var_groupby is provided and more than one type of aggregation (weighted and non-weighted)
                        is performed. The column names in the output DataFrame will be appended with suffixes like "_sum".

        Notes:
        - The function uses helper functions like `create_pond_data` and `create_dict_suffix` which should be present in the
        same context.
        - It handles different combinations of list_var_groupby and var_weights, giving flexibility in the type of aggregation
        performed.

        Examples:
        ```
        df = pd.DataFrame({
            'A': [1, 2, 3, 4],
            'B': [10, 20, 30, 40],
            'C': [100, 200, 300, 400],
            'weights': [0.1, 0.2, 0.3, 0.4]
        })

        sum(df, list_var_groupby=['A'], list_var_of_interest=['B', 'C'], var_weights='weights')
        ```

        """
        # Initialisation de la liste à retourner
        list_data_return = []

        # Calcul de la somme des poids non pondérée et suppression de la liste des variables d'intérêt
        # Sans doute faire la même chose pour les autres opérations
        # L'inconvénient c'est qu'on ne peut plus faire de somme pondérée des poids (alors qu'avant on pouvait déjà le faire avec var_weights=None)
        if (var_weights is not None) & (var_weights in list_var_of_interest):
            # Ajout de la somme des poids
            if list_var_groupby is not None:
                list_data_return.append(
                    data.groupby(list_var_groupby, as_index=True, observed=True)[
                        [var_weights]
                    ]
                    .sum()
                    .rename(
                        create_dict_suffix(list_name=[var_weights], suffix="_sum"),
                        axis=1,
                    )
                )
            else:
                list_data_return.append(
                    data[[var_weights]]
                    .sum()
                    .to_frame()
                    .transpose()
                    .rename(
                        create_dict_suffix(list_name=[var_weights], suffix="_sum"),
                        axis=1,
                    )
                )

            # Initialisation de la liste de travail sans les poids
            list_var_of_interest_work = [
                var_of_interest
                for var_of_interest in list_var_of_interest
                if var_of_interest != var_weights
            ]
        else:
            list_var_of_interest_work = list_var_of_interest

        if (
            (len(list_var_of_interest_work) > 0)
            & (list_var_groupby is not None)
            & (var_weights is not None)
        ):
            data_pond = create_pond_data(
                data=data,
                list_var_of_interest=list_var_of_interest_work,
                list_var_groupby=list_var_groupby,
                var_weights=var_weights,
            )
            list_data_return.append(
                data_pond.groupby(list_var_groupby, as_index=True, observed=True)[
                    list_var_of_interest_work
                ]
                .sum()
                .rename(
                    create_dict_suffix(
                        list_name=list_var_of_interest_work, suffix="_sum"
                    ),
                    axis=1,
                )
            )
        elif (
            (len(list_var_of_interest_work) > 0)
            & (list_var_groupby is not None)
            & (var_weights is None)
        ):
            list_data_return.append(
                data.groupby(list_var_groupby, as_index=True, observed=True)[
                    list_var_of_interest_work
                ]
                .sum()
                .rename(
                    create_dict_suffix(
                        list_name=list_var_of_interest_work, suffix="_sum"
                    ),
                    axis=1,
                )
            )
        elif (
            (len(list_var_of_interest_work) > 0)
            & (list_var_groupby is None)
            & (var_weights is not None)
        ):
            data_pond = create_pond_data(
                data=data,
                list_var_of_interest=list_var_of_interest_work,
                list_var_groupby=list_var_groupby,
                var_weights=var_weights,
            )
            list_data_return.append(
                data_pond.sum()
                .to_frame()
                .transpose()
                .rename(
                    create_dict_suffix(
                        list_name=list_var_of_interest_work, suffix="_sum"
                    ),
                    axis=1,
                )
            )
        elif (
            (len(list_var_of_interest_work) > 0)
            & (list_var_groupby is None)
            & (var_weights is None)
        ):
            list_data_return.append(
                data[list_var_of_interest_work]
                .sum()
                .to_frame()
                .transpose()
                .rename(
                    create_dict_suffix(
                        list_name=list_var_of_interest_work, suffix="_sum"
                    ),
                    axis=1,
                )
            )

        if list_var_groupby is not None:
            return pd.concat(list_data_return, axis=1, join="outer")
        else:
            return pd.concat(list_data_return, axis=0, join="outer")

    def mean(
        self,
        data: pd.DataFrame,
        list_var_groupby: Union[List[str], None],
        list_var_of_interest: List[str],
        var_weights: Union[str, None],
    ) -> pd.DataFrame:
        """
        Computes the weighted or unweighted mean of the specified variables,
        possibly grouped by specified variables.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset containing the data to be processed.
        list_var_groupby : list of str, optional
            List of variable names to group by.
        list_var_of_interest : list of str
            List of variable names to compute the mean for.
        var_weights : str, optional
            Name of the column containing weights for weighted computation.

        Returns:
        --------
        pd.Series or pd.DataFrame : The computed mean for the specified variables.
        """
        if (list_var_groupby is not None) & (var_weights is not None):
            data_pond = create_pond_data(
                data=data,
                list_var_of_interest=list_var_of_interest,
                list_var_groupby=list_var_groupby,
                var_weights=var_weights,
            )
            return (
                data_pond.groupby(list_var_groupby, as_index=True, observed=True)[
                    list_var_of_interest
                ]
                .sum()
                .divide(
                    other=data.groupby(list_var_groupby, as_index=True, observed=True)[
                        var_weights
                    ].sum(),
                    axis=0,
                )
                .rename(
                    create_dict_suffix(list_name=list_var_of_interest, suffix="_mean"),
                    axis=1,
                )
            )
        elif (list_var_groupby is not None) & (var_weights is None):
            return (
                data.groupby(list_var_groupby, as_index=True, observed=True)[
                    list_var_of_interest
                ]
                .mean()
                .rename(
                    create_dict_suffix(list_name=list_var_of_interest, suffix="_mean"),
                    axis=1,
                )
            )
        elif (list_var_groupby is None) & (var_weights is not None):
            data_pond = create_pond_data(
                data=data,
                list_var_of_interest=list_var_of_interest,
                list_var_groupby=list_var_groupby,
                var_weights=var_weights,
            )
            return (
                data_pond.sum()
                .divide(other=data[var_weights].sum())
                .to_frame()
                .transpose()
                .rename(
                    create_dict_suffix(list_name=list_var_of_interest, suffix="_mean"),
                    axis=1,
                )
            )
        elif (list_var_groupby is None) & (var_weights is None):
            return (
                data[list_var_of_interest]
                .mean()
                .to_frame()
                .transpose()
                .rename(
                    create_dict_suffix(list_name=list_var_of_interest, suffix="_mean"),
                    axis=1,
                )
            )

    def median(
        self,
        data: pd.DataFrame,
        list_var_groupby: Union[List[str], None],
        list_var_of_interest: List[str],
        var_weights: Union[str, None],
    ) -> pd.DataFrame:
        """
        Computes the median value for the specified variables,
        possibly grouped by specified variables.

        Parameters are the same as the mean method.

        Returns:
        --------
        pd.Series or pd.DataFrame
            The median values for the specified variables.
        """
        return self.quantile(
            data=data,
            list_var_groupby=list_var_groupby,
            list_var_of_interest=list_var_of_interest,
            var_weights=var_weights,
            q=0.5,
        )

    def quantile(
        self,
        data: pd.DataFrame,
        list_var_groupby: Union[List[str], None],
        list_var_of_interest: List[str],
        var_weights: Union[str, None],
        q: float,
    ) -> pd.DataFrame:
        """
        Computes the q-th quantile for the specified variables,
        possibly grouped by specified variables.

        Parameters:
        -----------
        q : float
            Quantile to compute, which must be between 0 and 1 inclusive.

        Other parameters are the same as the mean method.

        Returns:
        --------
        pd.Series or pd.DataFrame
            The q-th quantile for the specified variables.
        """
        if (list_var_groupby is not None) & (var_weights is not None):
            return (
                data.groupby(list_var_groupby, as_index=True, observed=True)[
                    list_var_of_interest + [var_weights]
                ]
                .apply(
                    func=lambda x: weighted_quantile(
                        data=x,
                        vars_of_interest=list_var_of_interest,
                        var_weights=var_weights,
                        q=q,
                    )
                )
                .rename(
                    create_dict_suffix(
                        list_name=list_var_of_interest, suffix="_q" + str(q)
                    ),
                    axis=1,
                )
            )
        elif (list_var_groupby is not None) & (var_weights is None):
            return (
                data.groupby(list_var_groupby, as_index=True, observed=True)[
                    list_var_of_interest
                ]
                .quantile(q=q)
                .rename(
                    create_dict_suffix(
                        list_name=list_var_of_interest, suffix="_q" + str(q)
                    ),
                    axis=1,
                )
            )
        elif (list_var_groupby is None) & (var_weights is not None):
            return (
                weighted_quantile(
                    data=data,
                    vars_of_interest=list_var_of_interest,
                    var_weights=var_weights,
                    q=q,
                )
                .to_frame()
                .transpose()
                .rename(
                    create_dict_suffix(
                        list_name=list_var_of_interest, suffix="_q" + str(q)
                    ),
                    axis=1,
                )
            )
        elif (list_var_groupby is None) & (var_weights is None):
            return (
                data[list_var_of_interest]
                .quantile(q=q)
                .rename(0)
                .to_frame()
                .transpose()
                .rename(
                    create_dict_suffix(
                        list_name=list_var_of_interest, suffix="_q" + str(q)
                    ),
                    axis=1,
                )
            )

    def prop(
        self,
        data: pd.DataFrame,
        list_var_groupby: Union[List[str], None],
        list_var_of_interest: List[str],
        var_weights: Union[str, None],
        var_ref: str,
    ) -> pd.DataFrame:
        """
        Computes the proportion of the specified variables based on a reference variable,
        possibly grouped by specified variables.

        Parameters:
        -----------
        var_ref : str
            The reference variable used to compute the proportion.

        Other parameters are the same as the mean method.

        Returns:
        --------
        pd.Series or pd.DataFrame
            The computed proportions for the specified variables.
        """
        if (list_var_groupby is not None) & (var_weights is not None):
            data_pond = create_pond_data(
                data=data,
                list_var_of_interest=list_var_of_interest,
                list_var_groupby=list_var_groupby,
                var_weights=var_weights,
            )
            data_pond_ref = create_pond_data(
                data=data,
                list_var_of_interest=[var_ref],
                list_var_groupby=list_var_groupby,
                var_weights=var_weights,
            )
            return (
                data_pond.groupby(list_var_groupby)
                .sum()
                .divide(data_pond_ref.groupby(list_var_groupby)[var_ref].sum(), axis=0)
                .rename(
                    create_dict_suffix(
                        list_name=list_var_of_interest, suffix=f"_{var_ref}_prop"
                    ),
                    axis=1,
                )
            )
        elif (list_var_groupby is not None) & (var_weights is None):
            return (
                data.groupby(list_var_groupby)[list_var_of_interest]
                .sum()
                .divide(other=data.groupby(list_var_groupby)[var_ref].sum(), axis=0)
                .rename(
                    create_dict_suffix(
                        list_name=list_var_of_interest, suffix=f"_{var_ref}_prop"
                    ),
                    axis=1,
                )
            )
        elif (list_var_groupby is None) & (var_weights is not None):
            data_pond = create_pond_data(
                data=data,
                list_var_of_interest=list_var_of_interest,
                list_var_groupby=list_var_groupby,
                var_weights=var_weights,
            )
            data_pond_ref = create_pond_data(
                data=data,
                list_var_of_interest=[var_ref],
                list_var_groupby=list_var_groupby,
                var_weights=var_weights,
            )
            return (
                data_pond.sum()
                .divide(data_pond_ref[var_ref].sum())
                .to_frame()
                .transpose()
                .rename(
                    create_dict_suffix(
                        list_name=list_var_of_interest, suffix=f"_{var_ref}_prop"
                    ),
                    axis=1,
                )
            )
        elif (list_var_groupby is None) & (var_weights is None):
            data_res = (
                (data[list_var_of_interest].sum() / data[var_ref].sum())
                .to_frame()
                .transpose()
                .rename(
                    create_dict_suffix(
                        list_name=list_var_of_interest, suffix=f"_{var_ref}_prop"
                    ),
                    axis=1,
                )
            )
            return data_res

    def majority(
        self,
        data: pd.DataFrame,
        list_var_groupby: Union[List[str], None],
        list_var_of_interest: List[str],
        var_weights: Union[str, None],
    ) -> pd.DataFrame:
        """
        Determine the majority value for the specified variables, optionally weighted and grouped.

        Parameters:
            data (pd.DataFrame): The data frame containing the data.
            list_var_groupby (list or None): The variables to group by. If None, no grouping is performed.
            list_var_of_interest (list): The variables for which to determine the majority value.
            var_weights (str or None): The variable to use for weighting. If None, no weighting is applied.

        Returns:
            pd.DataFrame: A data frame with the majority value for each specified variable of interest, optionally grouped and weighted.
        """
        if (list_var_groupby is not None) & (var_weights is not None):
            return pd.concat(
                [
                    data.groupby(list_var_groupby)[[var_of_interest, var_weights]]
                    .apply(
                        func=lambda x: x.groupby(var_of_interest)[var_weights]
                        .sum()
                        .idxmax()
                    )
                    .to_frame()
                    .rename({0: var_of_interest}, axis=1)
                    for var_of_interest in list_var_of_interest
                ],
                axis=1,
                join="outer",
            ).rename(
                create_dict_suffix(list_name=list_var_of_interest, suffix="_majority"),
                axis=1,
            )
        elif (list_var_groupby is not None) & (var_weights is None):
            return pd.concat(
                [
                    data.groupby(list_var_groupby)[[var_of_interest]]
                    .apply(func=lambda x: x[var_of_interest].value_counts().idxmax())
                    .to_frame()
                    .rename({0: var_of_interest}, axis=1)
                    for var_of_interest in list_var_of_interest
                ],
                axis=1,
                join="outer",
            ).rename(
                create_dict_suffix(list_name=list_var_of_interest, suffix="_majority"),
                axis=1,
            )
        elif (list_var_groupby is None) & (var_weights is not None):
            return (
                pd.Series(
                    [
                        data.groupby(var_of_interest)[var_weights].sum().idxmax()
                        for var_of_interest in list_var_of_interest
                    ],
                    index=list_var_of_interest,
                )
                .to_frame()
                .transpose()
                .rename(
                    create_dict_suffix(
                        list_name=list_var_of_interest, suffix="_majority"
                    ),
                    axis=1,
                )
            )
        elif (list_var_groupby is None) & (var_weights is None):
            return (
                pd.Series(
                    [
                        data[var_of_interest].value_counts().idxmax()
                        for var_of_interest in list_var_of_interest
                    ],
                    index=list_var_of_interest,
                )
                .to_frame()
                .transpose()
                .rename(
                    create_dict_suffix(
                        list_name=list_var_of_interest, suffix="_majority"
                    ),
                    axis=1,
                )
            )

    def inf_threshold(
        self,
        data: pd.DataFrame,
        list_var_groupby: Union[List[str], None],
        list_var_of_interest: List[str],
        var_threshold: str,
        seuil: Union[int, float],
    ) -> pd.DataFrame:
        """
        Computes the proportion of unique values for the specified variables below a certain threshold.

        Parameters:
        -----------
        var_threshold : str
            The variable based on which the thresholding will be done.
        seuil : int or float
            The threshold value.

        Other parameters are the same as the mean method.

        Returns:
        --------
        pd.Series or pd.DataFrame
            The proportion of unique values for the specified variables below the threshold.
        """
        # A terminer en autorisant plusieurs opérations
        if list_var_groupby is not None:
            return (
                data.loc[data[var_threshold] < seuil]
                .groupby(list_var_groupby)[list_var_of_interest]
                .nunique()
                / data.groupby(list_var_groupby)[list_var_of_interest].nunique()
            ).rename(
                create_dict_suffix(
                    list_name=list_var_of_interest, suffix="_inf_" + str(seuil)
                ),
                axis=1,
            )
        else:
            return (
                (
                    data.loc[data[var_threshold] < seuil][
                        list_var_of_interest
                    ].nunique()
                    / data[list_var_of_interest].nunique()
                )
                .to_frame()
                .transpose()
                .rename(
                    create_dict_suffix(
                        list_name=list_var_of_interest, suffix="_inf_" + str(seuil)
                    ),
                    axis=1,
                )
            )

    def max_sum_effectif(
        self,
        data: pd.DataFrame,
        list_var_groupby: Union[List[str], None],
        list_var_of_interest: List[str],
        var_weights: Union[str, None],
        var_id: str,
    ) -> pd.DataFrame:
        """
        Computes the maximum proportion of a variable relative to its sum, possibly grouped by specified variables.

        Parameters:
        -----------
        var_id : str
            The identifier for renaming the resulting variables.

        Other parameters are the same as the mean method.

        Returns:
        --------
        pd.Series or pd.DataFrame
            The computed maximum proportions for the specified variables relative to their sum.
        """
        if (list_var_groupby is not None) & (var_weights is not None):
            data_pond = create_pond_data(
                data=data,
                list_var_of_interest=list_var_of_interest,
                list_var_groupby=list_var_groupby,
                var_weights=var_weights,
            )
            return (
                data_pond.groupby(list_var_groupby)
                .max()
                .divide(other=data_pond.groupby(list_var_groupby).sum())
                .rename(
                    create_dict_suffix(
                        list_name=list_var_of_interest, suffix=f"_{var_id}_max/sum"
                    ),
                    axis=1,
                )
            )
        elif (list_var_groupby is not None) & (var_weights is None):
            return (
                data.groupby(list_var_groupby)[list_var_of_interest]
                .max()
                .divide(data.groupby(list_var_groupby)[list_var_of_interest].sum())
                .rename(
                    create_dict_suffix(
                        list_name=list_var_of_interest, suffix=f"_{var_id}_max/sum"
                    ),
                    axis=1,
                )
            )
        elif (list_var_groupby is None) & (var_weights is not None):
            data_pond = create_pond_data(
                data=data,
                list_var_of_interest=list_var_of_interest,
                list_var_groupby=list_var_groupby,
                var_weights=var_weights,
            )
            return (
                data_pond.max()
                .divide(other=data_pond.sum())
                .to_frame()
                .transpose()
                .rename(
                    create_dict_suffix(
                        list_name=list_var_of_interest, suffix=f"_{var_id}_max/sum"
                    ),
                    axis=1,
                )
            )
        elif (list_var_groupby is None) & (var_weights is None):
            return (
                data[list_var_of_interest]
                .max()
                .divide(other=data[list_var_of_interest].sum())
                .to_frame()
                .transpose()
                .rename(
                    create_dict_suffix(
                        list_name=list_var_of_interest, suffix=f"_{var_id}_max/sum"
                    ),
                    axis=1,
                )
            )


# Fonction d'apurement des groupby successifs : S'il n'y a qu'une modalité en plus du "Total" dans le groupby précédent, seul la ligne "Total" est conservée
def nest_groupby(
    data_stat_des: pd.DataFrame,
    list_var_groupby: List[str],
    modality: Optional[str] = "Total",
) -> pd.DataFrame:
    """
    Refine a dataset based on successive groupby operations. If there is only one modality in addition to the "Total" in the previous
    groupby, only the "Total" row is preserved.

    Parameters:
    - data_stat_des (pd.DataFrame): Input dataset to refine.
    - list_var_groupby (list of str): List of columns to perform successive groupby operations on.
    - modality (str, optional): Reference modality to identify specific rows. Default is 'Total'.

    Returns:
    - pd.DataFrame: Refined dataset after successive groupby operations.

    Notes:
    This function helps in data summarization where, for each group defined by the previous groupby columns, if only one unique
    value exists in addition to the 'Total' or specified modality, then only the 'Total' or specified modality row is retained.
    Ensure that there are no NaN values in the specified columns of the input dataframe. The function makes use of a helper
    `StatDesGroupBy` class to perform the iterative groupby operations.
    """
    # Copie indépendante du jeu de données
    data_res = data_stat_des.copy()

    if len(list_var_groupby) > 1:
        for i in range(1, len(list_var_groupby)):
            # Restriction au sous ensemble de variable de groupby
            list_var_groupby_work = list_var_groupby[:i]
            # Identification de la variable d'intérêt
            var_of_interest = list_var_groupby[i]
            # Création d'un jeu de données auxiliaire pour l'implémentation du critère
            data_criteria = data_stat_des[
                list_var_groupby_work + [var_of_interest]
            ].copy()
            # Ajout du test si la modalité est présente dans le jeu de données
            data_criteria["is_modality"] = data_criteria[var_of_interest].apply(
                func=lambda x: True if x == modality else False
            )
            # Dénombrement des items de groupby et de présence de la modalité
            # Initialisation du module de statistiques descriptives
            data_source = data_criteria.copy()
            list_var_of_interest = [var_of_interest, "is_modality"]
            var_individu = None
            var_entreprise = None
            var_weights = None
            iterable_operations = {"nunique": [var_of_interest], "any": ["is_modality"]}
            stat_des_generator = StatDesGroupBy(
                data_source=data_source,
                list_var_groupby=list_var_groupby_work,
                list_var_of_interest=list_var_of_interest,
                var_individu=var_individu,
                var_entreprise=var_entreprise,
                var_weights=var_weights,
            )
            data_stat_aux = (
                stat_des_generator.iterate_without_total(
                    iterable_operations=iterable_operations
                )
                .reset_index()
                .rename({"index": list_var_groupby_work[0]}, axis=1)
            )
            data_criteria = pd.merge(
                left=data_criteria,
                right=data_stat_aux,
                on=list_var_groupby_work,
                how="left",
                validate="many_to_one",
            )
            # Restriction aux lignes d'intérêt
            # Les index de data_criteria et data_res sont identiques du fait du left_merge
            data_res.loc[
                (data_criteria[var_of_interest + "_nunique"] == 2)
                & (data_criteria["is_modality_any"]),
                var_of_interest,
            ] = modality
    elif len(list_var_groupby) == 1:
        if (data_res[list_var_groupby[0]].nunique() == 2) & (
            modality in data_res[list_var_groupby[0]]
        ):
            data_res[list_var_groupby[0]] = modality

    # Supression des doublons en indice
    data_res.drop_duplicates(subset=list_var_groupby, inplace=True)

    return data_res
