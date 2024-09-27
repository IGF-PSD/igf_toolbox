"""
TO DO :
- Pour diminuer la mémoire, le jeu de données peut sans doute n'être renseigné qu'en argument des méthodes de SecretStatController
"""

# Importation des modules
# Modules de bases
import os
from copy import deepcopy
from itertools import combinations
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Classe de statistiques descriptives
from .base import StatDesGroupBy, nest_groupby

# from igf_toolbox_python.stats_des.general import StatDesGroupBy, nest_groupby
# Utilitaire
# from igf_toolbox_python.utils.general import nest_groupby


# Contrôle du secret statistique
class PrimarySecretStatController(object):
    """
    Control primary statistical secrets in a dataset.

    This class provides methods to ensure that specific statistical criteria (secret statistics) are maintained
    in a dataset. The criteria are based on predefined thresholds.

    Attributes:
    - var_individu (str or None): The variable representing individuals.
    - var_entreprise (str or None): The variable representing companies.
    - threshold_secret_stat_effectif_individu (int or None): The threshold for primary statistical secrecy control (individuals).
    - threshold_secret_stat_effectif_entreprise (int or None): The threshold for primary statistical secrecy control (companies).
    - threshold_secret_stat_contrib_individu (float or None): The threshold for contribution statistical secrecy control (individuals).
    - threshold_secret_stat_contrib_entreprise (float or None): The threshold for contribution statistical secrecy control (companies).

    Methods:
    - _control_secret_stat_effectif(data, var_of_interest, threshold): Check if the number of unique values of a variable in the dataset meet
      a minimum threshold. Adds a boolean column indicating whether the data in each row respects the secret
      statistic criteria or not.
    - _control_secret_stat_contrib(data, var_of_interest, threshold): Check if the values of a variable in the dataset
      exceed a maximum threshold. Adds a boolean column indicating the status.
    - control_primary_statistic_secret(data, iterable_operations, list_var_of_interest_max_sum_effectif): Control primary statistical secrecy for the given dataset.
      This function controls primary and contribution statistical secrecy for a dataset by comparing the results with
      predefined thresholds. It modifies the dataset as needed to maintain the secrecy of statistical information.
    """

    def __init__(
        self,
        var_individu: Union[str, None],
        var_entreprise: Union[str, None],
        threshold_secret_stat_effectif_individu: Union[int, None],
        threshold_secret_stat_effectif_entreprise: Union[int, None],
        threshold_secret_stat_contrib_individu: Union[float, None],
        threshold_secret_stat_contrib_entreprise: Union[float, None],
    ) -> None:
        """
        Initialize the PrimarySecretStatController class.

        Parameters:
        - var_individu (str or None): Variable representing individuals.
        - var_entreprise (str or None): Variable representing companies.
        - threshold_secret_stat_effectif_individu (int or None): The threshold for primary statistical secrecy control (individuals).
        - threshold_secret_stat_effectif_entreprise (int or None): The threshold for primary statistical secrecy control (companies).
        - threshold_secret_stat_contrib_individu (float or None): The threshold for contribution statistical secrecy control (individuals).
        - threshold_secret_stat_contrib_entreprise (float or None): The threshold for contribution statistical secrecy control (companies).
        """
        # Initialisation des paramètres
        # Variables de contrôle du secret statistique
        self.var_individu = var_individu
        self.var_entreprise = var_entreprise
        # Seuils de contrôle du secret statistique
        self.threshold_secret_stat_effectif_individu = (
            threshold_secret_stat_effectif_individu
        )
        self.threshold_secret_stat_effectif_entreprise = (
            threshold_secret_stat_effectif_entreprise
        )
        self.threshold_secret_stat_contrib_individu = (
            threshold_secret_stat_contrib_individu
        )
        self.threshold_secret_stat_contrib_entreprise = (
            threshold_secret_stat_contrib_entreprise
        )

    def _control_secret_stat_effectif(
        self, data: pd.DataFrame, var_of_interest: str, threshold: int
    ) -> pd.DataFrame:
        """
        Control the primary secret statistic on the number of observations for a variable of interest based on a threshold.

        Adds a column indicating whether the data in each row respects the secret statistic criteria or not.

        Parameters:
        - data (pd.DataFrame) : Dataset with the variable to be controlled
        - var_of_interest (str): Column name of the variable to be controlled.
        - threshold (int): The threshold for primary statistical secrecy control.

        Returns:
        - pd.DataFrame: Dataset with an added column indicating the status of the secret statistic.
        """

        # Ajout d'une colonne secret stat
        data[var_of_interest + "_secret_stat"] = True
        data.loc[
            data[var_of_interest] < threshold, var_of_interest + "_secret_stat"
        ] = False

        return data

    def _control_secret_stat_contrib(
        self, data: pd.DataFrame, var_of_interest: str, threshold: int
    ) -> pd.DataFrame:
        """
        Control the secondary secret statistic for a variable of interest based on a threshold.

        Adds a column indicating whether the data in each row respects the secret statistic criteria or not.

        Parameters:
        - data (pd.DataFrame) : Dataset with the variable to be controlled
        - var_of_interest (str): Column name of the variable to be controlled.
        - threshold (int): The threshold for contribution statistical secrecy control.

        Returns:
        - pd.DataFrame: Dataset with an added column indicating the status of the secret statistic.
        """
        # Ajout d'une colonne secret stat
        data[var_of_interest + "_secret_stat_second"] = True
        data.loc[
            data[var_of_interest] > threshold, var_of_interest + "_secret_stat_second"
        ] = False

        return data

    # Fonction liminaire de contrôle du secret statistique primaire sur les effectifs et les contributions
    def control_primary_statistic_secret(
        self,
        data: pd.DataFrame,
        iterable_operations: Union[list, dict],
        list_var_of_interest_max_sum_effectif: List[str],
    ) -> pd.DataFrame:
        """
        Control primary statistical secrecy for the given dataset.

        This function controls primary and contribution statistical secrecy for a dataset by comparing the results with
        predefined thresholds. It modifies the dataset as needed to maintain the secrecy of statistical information.

        Parameters:
            data (pd.DataFrame): The dataset containing descriptive statistics to control the statistical secret on.
            iterable_operations (dict or list): The list of statistical operations to be applied.
            list_var_of_interest_max_sum_effectif (list): List of variables related to contribution control operations.

        Returns:
            DataFrame: The modified dataset after applying statistical secrecy controls.

        Note:
            - Primary statistical secrecy control compares the number of unique values in specific variables to predefined thresholds.
            - Contribution statistical secrecy control may include further checks based on specific variables.
            - The function updates the dataset to ensure that statistical secrecy is maintained based on the provided thresholds.
        """
        # Contrôle du secret statistique sur les effectifs
        if (self.var_individu is not None) & (self.var_entreprise is not None):
            data = self._control_secret_stat_effectif(
                data=data,
                var_of_interest=self.var_individu + "_nunique",
                threshold=self.threshold_secret_stat_effectif_individu,
            )
            data = self._control_secret_stat_effectif(
                data=data,
                var_of_interest=self.var_entreprise + "_nunique",
                threshold=self.threshold_secret_stat_effectif_entreprise,
            )
        elif (self.var_individu is not None) & (self.var_entreprise is None):
            data = self._control_secret_stat_effectif(
                data=data,
                var_of_interest=self.var_individu + "_nunique",
                threshold=self.threshold_secret_stat_effectif_individu,
            )
        elif (self.var_individu is None) & (self.var_entreprise is not None):
            data = self._control_secret_stat_effectif(
                data=data,
                var_of_interest=self.var_entreprise + "_nunique",
                threshold=self.threshold_secret_stat_effectif_entreprise,
            )

        # Définition de la nécessité de contrôler le secret statistique secondaire
        if isinstance(iterable_operations, dict):
            if any(x in iterable_operations.keys() for x in ["sum", "mean"]):
                need_secret_stat_second = True
            else:
                need_secret_stat_second = False
        elif isinstance(iterable_operations, list):
            if any(x in iterable_operations for x in ["sum", "mean"]):
                need_secret_stat_second = True
            else:
                need_secret_stat_second = False

        # Contrôle du secret statistique secondaire
        if (
            need_secret_stat_second
            & (self.var_individu is not None)
            & (self.var_entreprise is not None)
        ):
            for var_of_interest in list_var_of_interest_max_sum_effectif:
                data = self._control_secret_stat_contrib(
                    data=data,
                    var_of_interest=var_of_interest + f"_{self.var_individu}_max/sum",
                    threshold=self.threshold_secret_stat_contrib_individu,
                )
                data = self._control_secret_stat_contrib(
                    data=data,
                    var_of_interest=var_of_interest + f"_{self.var_entreprise}_max/sum",
                    threshold=self.threshold_secret_stat_contrib_entreprise,
                )
        elif (
            need_secret_stat_second
            & (self.var_individu is not None)
            & (self.var_entreprise is None)
        ):
            for var_of_interest in list_var_of_interest_max_sum_effectif:
                data = self._control_secret_stat_contrib(
                    data=data,
                    var_of_interest=var_of_interest + f"_{self.var_individu}_max/sum",
                    threshold=self.threshold_secret_stat_contrib_individu,
                )
        elif (
            need_secret_stat_second
            & (self.var_individu is None)
            & (self.var_entreprise is not None)
        ):
            for var_of_interest in list_var_of_interest_max_sum_effectif:
                data = self._control_secret_stat_contrib(
                    data=data,
                    var_of_interest=var_of_interest + f"_{self.var_entreprise}_max/sum",
                    threshold=self.threshold_secret_stat_contrib_entreprise,
                )

        return data


# Classe de construction d'une statistique descriptive en vérifiant le secret statistique
class SecondarySecretStatController(object):
    """
    A class for controlling secondary statistical secrecy in a dataset.

    This class provides methods to control secondary statistical secrecy in a dataset based on specified grouping variables
    and a secrecy strategy. It allows for the identification and flagging of non-secret observations based on the strategy.

    Attributes:
        list_var_groupby (List[str]): A list of variables used for grouping.
        var_individu (Union[str, None]): The variable representing individual-level data.
        var_entreprise (Union[str, None]): The variable representing enterprise-level data.
        strategy (Optional[str]): The strategy for controlling secrecy. Either 'min' or 'total'.

    Methods:
        var_effectif: A property that returns the variable representing the number of observations or entities.
        _control_local_ss2: A method for controlling secondary statistical secrecy at the local level using the specified strategy.
        _iterate_local_ss2: A method for iteratively controlling the secondary statistical secrecy at the local level for different subgroups.
        control_ss2: A method for controlling secondary statistical secrecy for a given dataset.
    """

    # Initialisation
    def __init__(
        self,
        list_var_groupby: List[str],
        var_individu: Union[str, None],
        var_entreprise: Union[str, None],
        strategy: Optional[str] = "total",
    ) -> None:
        """
        Initialize the SecondarySecretStatController class.

        Parameters:
            list_var_groupby (List[str]): A list of variables used for grouping.
            var_individu (Union[str, None]): The variable representing individual-level data.
            var_entreprise (Union[str, None]): The variable representing enterprise-level data.
            strategy (Optional[str]): The strategy for controlling secrecy. Either 'min' or 'total'. Defaults to 'total'.
        """
        # Initialisation des paramètres
        self.list_var_groupby = list_var_groupby
        self.var_individu = var_individu
        self.var_entreprise = var_entreprise
        self.strategy = strategy

    # Initialisation de la variable d'effectifs
    @property
    def var_effectif(self) -> Union[str, None]:
        """
        Get the variable representing the number of observations or entities.

        Returns:
            Union[str, None]: The variable representing the number of observations or entities, or None if not defined.
        """
        if self.var_individu is not None:
            var_effectif = f"{self.var_individu}_nunique"
        elif self.var_entreprise is not None:
            var_effectif = f"{self.var_entreprise}_nunique"
        else:
            var_effectif = None
        return var_effectif

    # Fonction liminaire de contrôle du secret statistique secondaire sur un sous-ensemble des variables
    def _control_local_ss2(
        self,
        data: pd.DataFrame,
        list_var_subgroupby: List[str],
        var_secret_primary: str,
    ) -> pd.DataFrame:
        """
        Control for secondary statistical secrecy at the local level using the specified strategy.

        This function performs secondary statistical secrecy control at the local level using the specified strategy, where only specific
        data points are flagged as non-secret based on a given strategy.

        Parameters:
            data (DataFrame): The dataset containing descriptive statistics.
            list_var_subgroupby (List[str]): The list of variables used in the sub-groupby.
            var_secret_primary (str): The primary secret variable to control.

        Returns:
            DataFrame: A DataFrame with a boolean column indicating whether each observation is considered non-secret (True) or secret (False).

        Raises:
            ValueError: If the 'var_effectif' parameter is not provided when the strategy is 'min'.
        """
        # Extraction du level d'intérêt
        level_of_interest = np.setdiff1d(data.index.names, list_var_subgroupby)[0]
        # Initialisation de la série résultat
        serie_res = pd.Series(
            True, index=data.index.get_level_values(level_of_interest)
        )
        # Si le nombre de False vaut le nombre d'observations -1, il faut controler le secret
        if len(data) - data[var_secret_primary].sum() == 1:
            # Différenciation suivant la stratégie
            if self.strategy == "min":
                if self.var_effectif is not None:
                    # Extraction de l'indice d'intérêt
                    idx_of_interest = (
                        data.loc[data[var_secret_primary], self.var_effectif]
                        .droplevel(list_var_subgroupby)
                        .idxmin()
                    )
                    serie_res.loc[idx_of_interest] = False
                else:
                    raise ValueError("'self.var_effectif' is undefined")
            elif self.strategy == "total":
                serie_res.loc["Total"] = False

        return serie_res.to_frame()

    # Fonction liminaire d'itération du contrôle local du secret statistique secondaire
    def _iterate_local_ss2(
        self,
        data: pd.DataFrame,
        subgroupby_iterable: Iterable,
        var_secret_primary: str,
        var_secret_secondary: str,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Iterate the secondary statistical secrecy control at the local level.

        This function iteratively controls the secondary statistical secrecy at the local level for different subgroups
        defined by combinations of grouping variables. It generates secondary secrecy flags for these subgroups based on
        the specified strategy.

        Parameters:
            data (DataFrame): The dataset to perform secrecy control on.
            subgroupby_iterable (Iterable): An iterable containing combinations of grouping variables.
            var_secret_primary (str): The primary secret variable to control.
            var_secret_secondary (str): The secondary secret variable to control.

        Returns:
            Tuple[DataFrame, List[str]]: A tuple containing the modified dataset with secondary secrecy flags added for subgroups
                and a list of combinations of grouping variables that need further secrecy control.
        """
        # Copie indépendante du jeu de données
        data_work = data.copy()
        # Initialisation de la liste des combinaisons à reparcourir
        list_reiterate = []
        for e in subgroupby_iterable:
            # Initialisation de la liste du groupby
            list_var_subgroupby = list(e)
            # Nom de la colonne
            name_col_ss2 = "_".join(list_var_subgroupby) + "_ss2"
            # Construction de la colonne de secret stat secondaire
            ss2_series = data_work.groupby(list_var_subgroupby).apply(
                func=lambda x: self._control_local_ss2(
                    data=x,
                    list_var_subgroupby=list_var_subgroupby,
                    var_secret_primary=var_secret_primary,
                )
            )
            data_work[name_col_ss2] = ss2_series
            # Combinaisons à reparcourir
            if data_work[name_col_ss2].sum() < len(data_work):
                # Construction de la liste des variables à ajouter
                if len(list_var_subgroupby) == 1:
                    list_combinations_add = np.setdiff1d(
                        self.list_var_groupby, list_var_subgroupby
                    ).tolist()
                elif len(list_var_subgroupby) >= 2:
                    # Extraction du degré de liberté
                    list_var_compl = np.setdiff1d(
                        self.list_var_groupby, list_var_subgroupby
                    ).tolist()
                    list_combinations_add = [
                        list(e) + list_var_compl
                        for e in combinations(
                            list_var_subgroupby, len(self.list_var_groupby) - 2
                        )
                    ]
                # Ajout à la liste résultat
                list_reiterate.append(list_combinations_add)

        # Synthèse du secret statistique primaire et des différents secrets statistiques secondaires avant une éventuelle nouovelle itération sur les nouveaux Nan ajoutés
        if self.var_effectif is not None:
            list_col_ss2 = np.setdiff1d(
                data_work.columns.tolist(),
                [var_secret_primary, var_secret_secondary, self.var_effectif],
            ).tolist()
        else:
            list_col_ss2 = np.setdiff1d(
                data_work.columns.tolist(), [var_secret_primary, var_secret_secondary]
            ).tolist()
        data_work[var_secret_secondary] &= data_work[list_col_ss2].all(axis=1)
        data_work[var_secret_primary] = data_work[
            [var_secret_primary, var_secret_secondary]
        ].all(axis=1)

        return data_work.drop(list_col_ss2, axis=1), list_reiterate

    # Fonction liminaire de contrôle du secret statistique
    def control_ss2(
        self, data: pd.DataFrame, var_ss_primary: Optional[str] = "secret_stat_primary"
    ) -> pd.DataFrame:
        """
        Control secondary statistical secrecy for a given dataset.

        This function controls the secondary statistical secrecy of a dataset based on a specified primary secrecy variable
        ('var_ss_primary'). It checks for secrecy violations in different subgroups defined by grouping variables
        ('var_individu' and 'var_entreprise'). The control process iterates over subgroups to determine the secondary secrecy
        status.

        Parameters:
            data (DataFrame): The dataset containing primary secrecy information.
            var_ss_primary (Optional[str]): The primary secrecy variable to control. Defaults to 'secret_stat_primary'.

        Returns:
            DataFrame: The dataset with secondary statistical secrecy controlled.
        """
        # Dans le cas où l'on ne groupby que sur un seul élément, on vérifie juste qu'il y a zéro ou strictement plus de une case ne respectant pas le secret statistique
        if len(self.list_var_groupby) == 1:
            # Ajout de la colonne relative au secret statistique secondaire
            data["secret_stat_secondary"] = self._control_local_ss2(
                data=data, list_var_subgroupby=[], var_secret_primary=var_ss_primary
            )
            # Ajout de la colonne relative au secret statistique
            data["secret_stat"] = data[[var_ss_primary, "secret_stat_secondary"]].all(
                axis=1
            )
        # Dans le cas où l'on groupby sur plus de variables, on vérifie qu'il y a zéro ou strictement plus de une case
        else:
            # Initialisation de la colonne de secret statistique
            data["secret_stat"] = data[var_ss_primary].copy()
            # Initialisation des combinaisons à parcourir
            subgroupby_iterable = list(
                combinations(self.list_var_groupby, len(self.list_var_groupby) - 1)
            )
            # Initialisation de la colonne d'intérêt
            var_secret_primary = "secret_stat"
            var_secret_secondary = "secret_stat_secondary"
            # Initialisation du jeu de données de travail
            if self.var_effectif is not None:
                data_source = data[[var_secret_primary, self.var_effectif]].copy()
            else:
                data_source = data[[var_secret_primary]].copy()
            data_source[var_secret_secondary] = True

            # Itération
            while len(subgroupby_iterable) > 0:
                data_source, subgroupby_iterable = self._iterate_local_ss2(
                    data=data_source,
                    subgroupby_iterable=subgroupby_iterable,
                    var_secret_primary=var_secret_primary,
                    var_secret_secondary=var_secret_secondary,
                )

            # Mise à jour du secret statistique
            data[["secret_stat_secondary", "secret_stat"]] = data_source[
                ["secret_stat_secondary", "secret_stat"]
            ].copy()

        return data


# Classe d'itération des statistiques descriptives et du contrôle des secrets statistiques primaires et secondaires
class SecretStatEstimator(
    StatDesGroupBy, PrimarySecretStatController, SecondarySecretStatController
):
    """
    A class for estimating and controlling statistical secrecy in a dataset.

    This class inherits from three other classes: StatDesGroupBy, PrimarySecretStatController, and SecondarySecretStatController.
    It provides methods to estimate and control statistical secrecy in a dataset based on specified parameters.

    Attributes:
        data_source (DataFrame): The dataset to estimate and control secrecy.
        list_var_groupby (List[str]): The list of variables to group the data by.
        list_var_of_interest (List[str]): The list of variables of interest for statistics.
        var_individu (Optional[Union[str, None]]): The variable representing individual-level data.
        var_entreprise (Optional[Union[str, None]]): The variable representing enterprise-level data.
        var_weights (Optional[Union[str, None]]): The variable for weighting.
        threshold_secret_stat_effectif_individu (Optional[Union[int, None]]): The threshold for individual secrecy control.
        threshold_secret_stat_effectif_entreprise (Optional[Union[int, None]]): The threshold for enterprise secrecy control.
        threshold_secret_stat_contrib_individu (Optional[Union[float, None]]): The threshold for secondary individual secrecy control.
        threshold_secret_stat_contrib_entreprise (Optional[Union[float, None]]): The threshold for secondary enterprise secrecy control.
        strategy (Optional[str]): The control strategy ('total' or 'min').

    Methods:
        _clean_operations: Clean and update the list of operations and variables of interest based on specific criteria.
        _get_drop_add_columns: Determine columns to be added and dropped in a descriptive statistics dataset.
        estimate_secret_stat: Estimate and control secondary statistical secrecy for a given dataset.
    """

    # Initialisation
    def __init__(
        self,
        data_source: pd.DataFrame,
        list_var_groupby: List[str],
        list_var_of_interest: List[str],
        var_individu: Optional[Union[str, None]] = None,
        var_entreprise: Optional[Union[str, None]] = None,
        var_weights: Optional[Union[str, None]] = None,
        threshold_secret_stat_effectif_individu: Optional[Union[int, None]] = None,
        threshold_secret_stat_effectif_entreprise: Optional[Union[int, None]] = None,
        threshold_secret_stat_contrib_individu: Optional[Union[float, None]] = None,
        threshold_secret_stat_contrib_entreprise: Optional[Union[float, None]] = None,
        strategy: Optional[str] = "total",
    ) -> None:
        """
        Initialize the SecretStatEstimator class.

        Parameters:
            data_source (DataFrame): The dataset to estimate and control secrecy.
            list_var_groupby (List[str]): The list of variables to group the data by.
            list_var_of_interest (List[str]): The list of variables of interest for statistics.
            var_individu (Optional[Union[str, None]]): The variable representing individual-level data.
            var_entreprise (Optional[Union[str, None]]): The variable representing enterprise-level data.
            var_weights (Optional[Union[str, None]]): The variable for weighting.
            threshold_secret_stat_effectif_individu (Optional[Union[int, None]]): The threshold for individual secrecy control.
            threshold_secret_stat_effectif_entreprise (Optional[Union[int, None]]): The threshold for enterprise secrecy control.
            threshold_secret_stat_contrib_individu (Optional[Union[float, None]]): The threshold for secondary individual secrecy control.
            threshold_secret_stat_contrib_entreprise (Optional[Union[float, None]]): The threshold for secondary enterprise secrecy control.
            strategy (Optional[str]): The control strategy ('total' or 'min').
        """
        # Initialisation de la classe de statistiques descriptives
        StatDesGroupBy.__init__(
            self,
            data_source=data_source,
            list_var_groupby=list_var_groupby,
            list_var_of_interest=list_var_of_interest,
            var_individu=var_individu,
            var_entreprise=var_entreprise,
            var_weights=var_weights,
            dropna=True,
        )
        # Initialisation des classes de contrôle du secret statistique
        PrimarySecretStatController.__init__(
            self,
            var_individu=var_individu,
            var_entreprise=var_entreprise,
            threshold_secret_stat_effectif_individu=threshold_secret_stat_effectif_individu,
            threshold_secret_stat_effectif_entreprise=threshold_secret_stat_effectif_entreprise,
            threshold_secret_stat_contrib_individu=threshold_secret_stat_contrib_individu,
            threshold_secret_stat_contrib_entreprise=threshold_secret_stat_contrib_entreprise,
        )
        SecondarySecretStatController.__init__(
            self,
            list_var_groupby=list_var_groupby,
            var_individu=var_individu,
            var_entreprise=var_entreprise,
            strategy=strategy,
        )

    # Fonction liminaire de nettoyage des opérations à effectuer sur les données
    def _clean_operations(
        self, iterable_operations: Union[dict, list]
    ) -> Tuple[Union[dict, list], List[str], List[str]]:
        """
        Clean and update the list of operations and variables of interest based on specific criteria.

        This function modifies the list of operations and variables of interest, ensuring it includes certain control operations
        for statistical secrecy and makes necessary adjustments.

        Parameters:
            iterable_operations (dict or list): The initial list of operations.

        Returns:
            Tuple[Union[dict, list], List[str], List[str]]: A tuple containing the following:
                - iterable_operations_work (list or dict): The cleaned and updated list of operations.
                - list_var_of_interest_work (list): The cleaned and updated list of variables of interest.
                - list_var_of_interest_max_sum_effectif (list): A list of variables related to secondary control operations.
        """
        # Copie indépendante des opérations
        iterable_operations_work = deepcopy(iterable_operations)
        # Copie indépendante des variables d'intérêt
        list_var_of_interest_work = deepcopy(self.list_var_of_interest)
        # Initialisation de la liste des max/sum
        list_var_of_interest_max_sum_effectif = []
        # Ajout des opérations de contrôle du secret statistique si elles ne sont pas prévues dans la liste d'opérations
        if isinstance(iterable_operations, dict):
            # Test du besoin d'ajout d'une opération de contrôle du secret statistique primaire
            if "count_effectif" not in iterable_operations.keys():
                iterable_operations_work["count_effectif"] = []
            # Suppression des éventuelles opérations dupliquées
            if "nunique" in iterable_operations.keys():
                # Comptage des individus et des entreprises
                if (self.var_individu in iterable_operations["nunique"]) & (
                    self.var_entreprise in iterable_operations["nunique"]
                ):
                    if len(iterable_operations["nunique"]) > 2:
                        iterable_operations_work["nunique"] = [
                            e
                            for e in iterable_operations["nunique"]
                            if e not in [self.var_individu, self.var_entreprise]
                        ]
                    else:
                        del iterable_operations_work["nunique"]
                # Comptage des individus
                elif self.var_individu in iterable_operations["nunique"]:
                    if len(iterable_operations["nunique"]) > 1:
                        iterable_operations_work["nunique"] = [
                            e
                            for e in iterable_operations["nunique"]
                            if e != self.var_individu
                        ]
                    else:
                        del iterable_operations_work["nunique"]
                # Comptage des entreprises
                elif self.var_entreprise in iterable_operations["nunique"]:
                    if len(iterable_operations["nunique"]) > 1:
                        iterable_operations_work["nunique"] = [
                            e
                            for e in iterable_operations["nunique"]
                            if e != self.var_entreprise
                        ]
                    else:
                        del iterable_operations_work["nunique"]

            # Test du besoin d'ajout d'une opération de contrôle du secret statistique secondaire
            if any(x in iterable_operations.keys() for x in ["sum", "mean"]):
                # Extraction des opérations concernées
                list_operation_max_sum_effectif = np.intersect1d(
                    ["sum", "mean"],
                    [
                        e[0] if isinstance(e, tuple) else e
                        for e in iterable_operations.keys()
                    ],
                ).tolist()
                # Extraction des variables concernées par ces opérations
                list_var_of_interest_max_sum_effectif = np.unique(
                    np.concatenate(
                        [
                            iterable_operations[operation]
                            for operation in list_operation_max_sum_effectif
                        ]
                    )
                ).tolist()
                # Suppression des variables éventuellement déjà comprises dans l'opérateur
                if "max_sum_effectif" in iterable_operations.keys():
                    iterable_operations_work["max_sum_effectif"] = np.unique(
                        iterable_operations["max_sum_effectif"]
                        + list_var_of_interest_max_sum_effectif
                    ).tolist()
                else:
                    iterable_operations_work["max_sum_effectif"] = (
                        list_var_of_interest_max_sum_effectif
                    )

        elif isinstance(iterable_operations, list):
            # Test du besoin d'ajout d'une opération de contrôle du secret statistique primaire
            if (
                ("nunique" in iterable_operations)
                | (self.var_individu in self.list_var_of_interest)
                | (self.var_entreprise in self.list_var_of_interest)
            ):
                # Actualisation des valeurs des variables
                if (self.var_individu is not None) & (self.var_entreprise is not None):
                    list_var_of_interest_work = np.unique(
                        self.list_var_of_interest
                        + [self.var_individu, self.var_entreprise]
                    ).tolist()
                elif self.var_individu is not None:
                    list_var_of_interest_work = np.unique(
                        self.list_var_of_interest + [self.var_individu]
                    ).tolist()
                elif self.var_entreprise is not None:
                    list_var_of_interest_work = np.unique(
                        self.list_var_of_interest + [self.var_entreprise]
                    ).tolist()
                iterable_operations_work = np.unique(
                    iterable_operations_work + ["nunique"]
                ).tolist()
            elif "count_effectif" not in iterable_operations:
                iterable_operations_work.append("count_effectif")

            # Test du besoin d'ajout d'une opération de contrôle du secret statistique secondaire
            list_var_of_interest_max_sum_effectif = deepcopy(self.list_var_of_interest)
            # Pour les effectifs
            if any(x in iterable_operations for x in ["sum", "mean"]) & (
                "max_sum_effectif" not in iterable_operations
            ):
                iterable_operations_work.append("max_sum_effectif")
        else:
            raise ValueError("Unsupported type for iterable_operations")

        return (
            iterable_operations_work,
            list_var_of_interest_work,
            list_var_of_interest_max_sum_effectif,
        )

    # Fonction liminaire des colonnes à ajouter et supprimer au jeu de données de statsitiques descriptives résultat
    def _get_drop_add_columns(
        self, data: pd.DataFrame, iterable_operations: Union[dict, list]
    ) -> Tuple[List[str], List[str]]:
        """
        Determine columns to be added and dropped in a descriptive statistics dataset.

        This function determines which columns should be added or dropped from a descriptive statistics dataset based on
        the provided parameters. The goal is to maintain statistical secrecy and ensure data integrity.

        Parameters:
            data (DataFrame): The dataset containing descriptive statistics.
            iterable_operations (dict or list): The list of statistical operations to be applied.

        Returns:
            Tuple[List[str], List[str]]: A tuple containing the following:
                - list_var_drop (list): A list of columns to be dropped from the dataset.
                - list_var_data_secret_stat_add (list): A list of columns to be added to the dataset.
        """
        if isinstance(iterable_operations, dict):
            if "nunique" not in iterable_operations.keys():
                if "count_effectif" in iterable_operations.keys():
                    list_var_drop = [
                        col
                        for col in data.columns
                        if ("_secret_stat" in col) | ("_max/sum" in col)
                    ]
                    list_var_data_secret_stat_add = []
                    if self.var_individu is not None:
                        list_var_data_secret_stat_add.append(
                            self.var_individu + "_nunique"
                        )
                    if self.var_entreprise is not None:
                        list_var_data_secret_stat_add.append(
                            self.var_entreprise + "_nunique"
                        )
                else:
                    list_var_drop = [
                        col
                        for col in data.columns
                        if ("_secret_stat" in col) | ("_max/sum" in col)
                    ]
                    if self.var_individu is not None:
                        list_var_drop.append(self.var_individu + "_nunique")
                    if self.var_entreprise is not None:
                        list_var_drop.append(self.var_entreprise + "_nunique")
                    list_var_data_secret_stat_add = []
            elif (self.var_individu in iterable_operations["nunique"]) | (
                self.var_entreprise in iterable_operations["nunique"]
            ):
                list_var_drop = [
                    col
                    for col in data.columns
                    if ("_secret_stat" in col) | ("_max/sum" in col)
                ]
                if self.var_individu not in iterable_operations["nunique"]:
                    if self.var_individu is not None:
                        list_var_drop.append(self.var_individu + "_nunique")
                    list_var_data_secret_stat_add = [self.var_entreprise + "_nunique"]
                elif self.var_entreprise not in iterable_operations["nunique"]:
                    if self.var_entreprise is not None:
                        list_var_drop.append(self.var_entreprise + "_nunique")
                    list_var_data_secret_stat_add = [self.var_individu + "_nunique"]
                else:
                    list_var_data_secret_stat_add = [
                        self.var_individu + "_nunique",
                        self.var_entreprise + "_nunique",
                    ]
            else:
                if (self.var_individu is not None) & (self.var_entreprise is not None):
                    list_var_drop = [
                        col
                        for col in data.columns
                        if ("_secret_stat" in col)
                        | ("_max/sum" in col)
                        | (self.var_individu + "_nunique" in col)
                        | (self.var_entreprise + "_nunique" in col)
                    ]
                elif self.var_individu is not None:
                    list_var_drop = [
                        col
                        for col in data.columns
                        if ("_secret_stat" in col)
                        | ("_max/sum" in col)
                        | (self.var_individu + "_nunique" in col)
                    ]
                elif self.var_entreprise is not None:
                    list_var_drop = [
                        col
                        for col in data.columns
                        if ("_secret_stat" in col)
                        | ("_max/sum" in col)
                        | (self.var_entreprise + "_nunique" in col)
                    ]
                else:
                    list_var_drop = [
                        col
                        for col in data.columns
                        if ("_secret_stat" in col) | ("_max/sum" in col)
                    ]
                list_var_data_secret_stat_add = []

        elif isinstance(iterable_operations, list):
            if ("count_effectif" in iterable_operations) | (
                ("nunique" in iterable_operations)
                & (
                    (self.var_individu in self.list_var_of_interest)
                    | (self.var_entreprise in self.list_var_of_interest)
                )
            ):
                if (
                    ("nunique" in iterable_operations)
                    & (self.var_individu in self.list_var_of_interest)
                    & (self.var_entreprise in self.list_var_of_interest)
                ):
                    list_var_drop = [
                        col
                        for col in data.columns
                        if ("_secret_stat" in col) | ("_max/sum" in col)
                    ]
                    list_var_data_secret_stat_add = [
                        self.var_individu + "_nunique",
                        self.var_entreprise + "_nunique",
                    ]
                elif ("nunique" in iterable_operations) & (
                    self.var_individu in self.list_var_of_interest
                ):
                    if self.var_entreprise is not None:
                        list_var_drop = [
                            col
                            for col in data.columns
                            if ("_secret_stat" in col)
                            | ("_max/sum" in col)
                            | (self.var_entreprise + "_nunique" in col)
                        ]
                    else:
                        list_var_drop = [
                            col
                            for col in data.columns
                            if ("_secret_stat" in col) | ("_max/sum" in col)
                        ]
                    list_var_data_secret_stat_add = [self.var_individu + "_nunique"]
                elif ("nunique" in iterable_operations) & (
                    self.var_entreprise in self.list_var_of_interest
                ):
                    if self.var_individu is not None:
                        list_var_drop = [
                            col
                            for col in data.columns
                            if ("_secret_stat" in col)
                            | ("_max/sum" in col)
                            | (self.var_individu + "_nunique" in col)
                        ]
                    else:
                        list_var_drop = [
                            col
                            for col in data.columns
                            if ("_secret_stat" in col) | ("_max/sum" in col)
                        ]
                    list_var_data_secret_stat_add = [self.var_entreprise + "_nunique"]
                # On est dans le cas où l'opération est count_effectif
                else:
                    list_var_drop = [
                        col
                        for col in data.columns
                        if ("_secret_stat" in col) | ("_max/sum" in col)
                    ]
                    if (self.var_individu is not None) & (
                        self.var_entreprise is not None
                    ):
                        list_var_data_secret_stat_add = [
                            self.var_individu + "_nunique",
                            self.var_entreprise + "_nunique",
                        ]
                    elif self.var_individu is not None:
                        list_var_data_secret_stat_add = [self.var_individu + "_nunique"]
                    elif self.var_entreprise is not None:
                        list_var_data_secret_stat_add = [
                            self.var_entreprise + "_nunique"
                        ]
                    else:
                        list_var_data_secret_stat_add = []
            else:
                list_var_data_secret_stat_add = []
                if (self.var_individu is not None) & (self.var_entreprise is not None):
                    list_var_drop = [
                        col
                        for col in data.columns
                        if ("_secret_stat" in col)
                        | ("_max/sum" in col)
                        | (self.var_individu + "_nunique" in col)
                        | (self.var_entreprise + "_nunique" in col)
                    ]
                elif self.var_individu is not None:
                    list_var_drop = [
                        col
                        for col in data.columns
                        if ("_secret_stat" in col)
                        | ("_max/sum" in col)
                        | (self.var_individu + "_nunique" in col)
                    ]
                elif self.var_entreprise is not None:
                    list_var_drop = [
                        col
                        for col in data.columns
                        if ("_secret_stat" in col)
                        | ("_max/sum" in col)
                        | (self.var_entreprise + "_nunique" in col)
                    ]
                else:
                    list_var_drop = [
                        col
                        for col in data.columns
                        if ("_secret_stat" in col) | ("_max/sum" in col)
                    ]

        return list_var_drop, list_var_data_secret_stat_add

    # Fonction d'estimation et de conttrole du secret statistique
    def estimate_secret_stat(
        self,
        iterable_operations: Union[dict, list],
        include_total: Optional[bool] = True,
        drop: Optional[bool] = True,
        fill_value: Optional[Union[int, float, str, np.nan]] = np.nan,
        nest: Optional[bool] = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Estimate and control secondary statistical secrecy for a given dataset.

        This function estimates secondary statistical secrecy for a dataset and controls it based on specified parameters.
        It calculates descriptive statistics, checks for primary and secondary secrecy violations, and separates variables
        to be added or removed in the resulting datasets.

        Parameters:
            iterable_operations (dict or list): The operations to perform on the variables of interest.
            include_total (bool, optional): Include total statistics in the result. Defaults to True.
            drop (bool, optional): Drop non-compliant rows from the result. Defaults to True.
            fill_value (float or np.nan, optional): The value to fill non-compliant rows when 'drop' is False. Defaults to np.nan.
            nest (bool, optional): Nest the index in the result. Defaults to False.

        Returns:
            Tuple[DataFrame, DataFrame]: A tuple containing the following:
                - data_stat_des (DataFrame): The dataset with secondary statistical secrecy estimated and controlled.
                - data_secret_stat (DataFrame): The dataset containing the secondary statistical secrecy information.
        """
        # Modification des opérations en fonction des secrets à estimer
        (
            iterable_operations_work,
            list_var_of_interest_work,
            list_var_of_interest_max_sum_effectif,
        ) = self._clean_operations(iterable_operations)

        # Statistiques descriptives
        # Itérations
        if include_total:
            data_stat_des = self.iterate_with_total(
                iterable_operations=iterable_operations_work
            )  # .reset_index()
        else:
            data_stat_des = self.iterate_without_total(
                iterable_operations=iterable_operations_work
            )  # .reset_index()

        # Contrôle du secret statistique primaire
        data_stat_des = self.control_primary_statistic_secret(
            data=data_stat_des,
            iterable_operations=iterable_operations,
            list_var_of_interest_max_sum_effectif=list_var_of_interest_max_sum_effectif,
        )

        # Enumération des variables contenant du secret statistique
        list_var_secret_stat = [
            col for col in data_stat_des.columns if "_secret_stat" in col
        ]

        # Séparation des variables à supprimer et ajouter aux jeux de données résultats
        list_var_drop, list_var_data_secret_stat_add = self._get_drop_add_columns(
            data=data_stat_des, iterable_operations=iterable_operations
        )

        # Création d'une colonne secret stat
        data_stat_des["secret_stat_primary"] = data_stat_des[list_var_secret_stat].all(
            axis=1
        )

        # Vérification du secret statistique secondaire
        if include_total & (len(self.list_var_groupby) >= 1):
            data_stat_des = self.control_ss2(
                data=data_stat_des, var_ss_primary="secret_stat_primary"
            )
        else:
            data_stat_des["secret_stat_secondary"] = True
            data_stat_des["secret_stat"] = data_stat_des["secret_stat_primary"].copy()

        # Initialisation du jeu de données de secret stat
        data_secret_stat = data_stat_des[
            list_var_data_secret_stat_add
            + list_var_drop
            + ["secret_stat_primary", "secret_stat_secondary", "secret_stat"]
        ].copy()

        # Restriction dans les deux jeux de données aux lignes respectant le secret statistique
        if drop is True:
            data_stat_des = data_stat_des.loc[data_stat_des["secret_stat"]]
            data_secret_stat = data_secret_stat.loc[data_secret_stat["secret_stat"]]
        else:
            data_stat_des.loc[~data_stat_des["secret_stat"]] = fill_value

        # Suppression des colonnes de secret stat du jeu de données principal
        data_stat_des.drop(
            list_var_drop
            + ["secret_stat_primary", "secret_stat_secondary", "secret_stat"],
            axis=1,
            inplace=True,
        )

        # Nesting des index
        if nest:
            data_stat_des = nest_groupby(
                data_stat_des=data_stat_des.reset_index().rename(
                    {"index": self.list_var_groupby[0]}, axis=1
                ),
                list_var_groupby=self.list_var_groupby,
                modality="Total",
            ).set_index(self.list_var_groupby)
            data_secret_stat = nest_groupby(
                data_stat_des=data_secret_stat.reset_index().rename(
                    {"index": self.list_var_groupby[0]}, axis=1
                ),
                list_var_groupby=self.list_var_groupby,
                modality="Total",
            ).set_index(self.list_var_groupby)

        return data_stat_des, data_secret_stat
