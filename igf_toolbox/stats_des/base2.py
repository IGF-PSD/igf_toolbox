# Importation des modules
# Modules de base
from itertools import combinations
from typing import List, Optional, Union, Dict, Any, Tuple

import numpy as np
import pandas as pd

# Utilitaire du package
from ..utils._auxiliary import _sort_index_with_total, create_dict_suffix
from .weighted import create_pond_data, weighted_quantile


# Classe d'itération de statistiques descriptives avec totaux
class StatDesGroupBy(object):
    """
    A class to compute descriptive statistics grouped by specified variables.
    
    Compared to the pandas.DataFrame.groupby method it handles weighted operations, 
    and can add totals.

    Args:
        data_source (pd.DataFrame):
            Source data for computing statistics.
        list_var_groupby (List[str]):
            List of variables to group data by.
        list_var_of_interest (List[str]):
            List of variables for which descriptive statistics will be computed.
        var_count (Union[str, List[str], None], optional):
            Variable(s) representing entities to count unique occurrences.
            Can be a single column name or a list of column names.
        var_weights (Union[str, None], optional):
            Variable representing the weights for each data entry.
        dropna (bool, optional):
            Whether to drop NaN values. Defaults to False.
    """

    # Initialisation
    def __init__(
        self,
        data_source: pd.DataFrame,
        list_var_groupby: List[str],
        list_var_of_interest: List[str],
        var_count: Optional[Union[str, List[str]]] = None,
        var_weights: Optional[str] = None,
        dropna: bool = False,
    ) -> None:
        """
        Initialize the StatDesGroupBy class with data and parameters.
        """
        # Initialisation des paramètres
        self.list_var_groupby = list_var_groupby
        self.list_var_of_interest = list_var_of_interest
        self.var_count = var_count if isinstance(var_count, list) or var_count is None else [var_count]
        self.var_weights = var_weights

        # Garder une copie du DataFrame original pour accéder aux colonnes supplémentaires si nécessaire
        if dropna:
            self._data_source_full = data_source.copy().dropna(how="any")
        else:
            self._data_source_full = data_source.copy()

        # Initialisation de la liste des variables à conserver
        list_var_keep = list_var_groupby + list_var_of_interest
        if var_weights is not None:
            list_var_keep.append(var_weights)
        if self.var_count is not None:
            list_var_keep.extend([v for v in self.var_count if v not in list_var_keep])

        # Suppression des doublons
        list_var_keep = list(dict.fromkeys(list_var_keep))

        # Copie filtrée du jeu de données (pour les opérations optimisées)
        self.data_source = self._data_source_full[list_var_keep].copy()

    # Méthode de calcul de statistiques descriptives avec totaux
    def iterate_with_total(
        self, iterable_operations: Union[Dict[str, List[str]], List[str]]
    ) -> pd.DataFrame:
        """
        Computes descriptive statistics and returns results with subtotals and a grand total.

        Args:
            iterable_operations (Union[Dict[str, List[str]], List[str]]):
                Operations or functions to apply to the grouped data.

        Returns:
            pd.DataFrame:
                Descriptive statistics with subtotals and a grand total.
        """
        # Disjonction de cas suivant la longueur de la liste de groupby
        if len(self.list_var_groupby) == 0:
            data_res = self._compute_total(iterable_operations=iterable_operations)
        elif len(self.list_var_groupby) == 1:
            # Itération des statistiques descriptives
            data_stat_des = self._compute_operations(
                data=self.data_source,
                list_var_groupby=self.list_var_groupby,
                iterable_operations=iterable_operations,
            )
            # Itération du total
            data_total = self._compute_total(iterable_operations=iterable_operations)
            # Concaténation des jeux de données
            data_res = pd.concat([data_stat_des, data_total], axis=0, join="outer")
        else:
            # Itération des statistiques descriptives
            data_stat_des = self._compute_operations(
                data=self.data_source,
                list_var_groupby=self.list_var_groupby,
                iterable_operations=iterable_operations,
            )
            # Itération des sous-totaux
            data_sub_total = self._compute_under_total(
                iterable_operations=iterable_operations
            )
            # Itération du total
            data_total = self._compute_total(iterable_operations=iterable_operations)
            # Concaténation des jeux de données
            data_res = pd.concat(
                [data_stat_des, data_sub_total, data_total], axis=0, join="outer"
            )

        # Tri de l'indice
        data_res = _sort_index_with_total(data_source=data_res)

        return data_res

    # Méthode de calcul de statistiques descriptives sans totaux
    def iterate_without_total(
        self, iterable_operations: Union[Dict[str, List[str]], List[str]]
    ) -> pd.DataFrame:
        """
        Computes descriptive statistics without subtotals or a grand total and returns results.

        Args:
            iterable_operations (Union[Dict[str, List[str]], List[str]]):
                Operations or functions to apply to the grouped data.

        Returns:
            pd.DataFrame:
                Descriptive statistics without subtotals or a grand total.
        """
        return self._compute_operations(
            data=self.data_source,
            list_var_groupby=self.list_var_groupby,
            iterable_operations=iterable_operations,
        ).sort_index()

    # Fonction auxiliaire de transformation de l'iterable d'opérations en un dictionnaire supporté par "agg"
    def _prepare_agg_dict(
        self, iterable_operations: Union[Dict[str, List[str]], List[str]]
    ) -> Tuple[Dict[str, List[Tuple[str, str]]], Dict[str, Any], Dict[str, Any]]:
        """
        Transform iterable_operations into dictionaries for different aggregation types.

        Args:
            iterable_operations (Union[Dict[str, List[str]], List[str]]):
                Operations to be applied.

        Returns:
            Tuple[Dict[str, List[Tuple[str, str]]], Dict[str, Any], Dict[str, Any]]:
                - agg_dict: Dictionary for pandas.agg()
                - weighted_ops: Operations requiring weighted computation
                - special_ops: Operations requiring special handling
        """
        # Initialisation des dictionnaires d'opérations
        agg_dict = {}
        weighted_ops = {}
        special_ops = {}
        
        # Opérations supportées directement par pandas.agg()
        pandas_agg_ops = ['sum', 'mean', 'count', 'nunique', 'any', 'all', 'min', 'max', 'std', 'var']
        
        # Traitement selon le type d'iterable_operations
        if isinstance(iterable_operations, dict):
            # Parcours des opérations dans le dictionnaire
            for operation, vars_list in iterable_operations.items():
                # Filtre des variables qui sont dans list_var_of_interest
                # SAUF pour les opérations spéciales qui peuvent s'appliquer à n'importe quelle colonne
                if isinstance(operation, str) and operation in ['majority']:
                    # Pour 'majority', ne pas filtrer - accepter toutes les variables demandées
                    vars_to_process = vars_list
                else:
                    vars_to_process = [v for v in vars_list if v in self.list_var_of_interest]

                # Distinction suivant le type d'opération
                if isinstance(operation, str):
                    if operation in pandas_agg_ops:
                        # Opération standard pandas
                        for var in vars_to_process:
                            if var not in agg_dict:
                                agg_dict[var] = []
                            agg_dict[var].append((f"{var}_{operation}", operation))

                    elif operation == 'median':
                        # Médiane = quantile 0.5
                        if self.var_weights:
                            weighted_ops['median'] = vars_to_process
                        else:
                            for var in vars_to_process:
                                if var not in agg_dict:
                                    agg_dict[var] = []
                                agg_dict[var].append((f"{var}_q0.5", 'median'))

                    elif operation == 'count_effectif':
                        # Comptage des entités uniques
                        if self.var_count:
                            for var in self.var_count:
                                if var not in agg_dict:
                                    agg_dict[var] = []
                                agg_dict[var].append((f"{var}_nunique", 'nunique'))

                    elif operation in ['majority', 'max_sum_effectif']:
                        special_ops[operation] = vars_to_process
                        
                elif isinstance(operation, tuple):
                    # Opérations avec paramètres
                    op_name = operation[0]
                    op_params = operation[1]

                    # Convertir op_params en tuple de tuples si c'est un dict
                    if isinstance(op_params, dict):
                        op_params_tuple = tuple(sorted(op_params.items()))
                    else:
                        # Déjà un tuple de tuples
                        op_params_tuple = op_params

                    if op_name == 'quantile':
                        # Quantiles
                        if self.var_weights:
                            # Créer une version hashable du tuple
                            hashable_op = (op_name, op_params_tuple)
                            weighted_ops[hashable_op] = vars_to_process
                        else:
                            # Récupérer q depuis le dict ou le tuple
                            q = op_params['q'] if isinstance(op_params, dict) else dict(op_params)['q']
                            for var in vars_to_process:
                                if var not in agg_dict:
                                    agg_dict[var] = []
                                agg_dict[var].append((f"{var}_q{q}", lambda x: x.quantile(q)))

                    elif op_name in ['prop', 'inf_threshold']:
                        # Opérations spéciales
                        # Créer une version hashable du tuple
                        hashable_op = (op_name, op_params_tuple)
                        special_ops[hashable_op] = vars_to_process
        
        else:  # Liste d'opérations
            # Parcours des opérations
            for operation in iterable_operations:
                if isinstance(operation, str):
                    if operation in pandas_agg_ops:
                        # Opération standard en pandas
                        for var in self.list_var_of_interest:
                            if var not in agg_dict:
                                agg_dict[var] = []
                            agg_dict[var].append((f"{var}_{operation}", operation))
                    
                    elif operation == 'median':
                        # Médiane
                        if self.var_weights:
                            weighted_ops['median'] = self.list_var_of_interest
                        else:
                            for var in self.list_var_of_interest:
                                if var not in agg_dict:
                                    agg_dict[var] = []
                                agg_dict[var].append((f"{var}_q0.5", 'median'))
                    
                    elif operation == 'count_effectif':
                        # Comptage des effectifs
                        if self.var_count:
                            for var in self.var_count:
                                if var not in agg_dict:
                                    agg_dict[var] = []
                                agg_dict[var].append((f"{var}_nunique", 'nunique'))
                    
                    elif operation in ['majority', 'max_sum_effectif']:
                        # Opération spéciales
                        special_ops[operation] = self.list_var_of_interest
                
                elif isinstance(operation, tuple):
                    # Cas des opérations à paramètres
                    op_name = operation[0]
                    op_params = operation[1]

                    # Convertir op_params en tuple de tuples si c'est un dict
                    if isinstance(op_params, dict):
                        op_params_tuple = tuple(sorted(op_params.items()))
                    else:
                        # Déjà un tuple de tuples
                        op_params_tuple = op_params

                    if op_name == 'quantile':
                        # Quantiles
                        if self.var_weights:
                            # Créer une version hashable du tuple
                            hashable_op = (op_name, op_params_tuple)
                            weighted_ops[hashable_op] = self.list_var_of_interest
                        else:
                            # Récupérer q depuis le dict ou le tuple
                            q = op_params['q'] if isinstance(op_params, dict) else dict(op_params)['q']
                            for var in self.list_var_of_interest:
                                if var not in agg_dict:
                                    agg_dict[var] = []
                                agg_dict[var].append((f"{var}_q{q}", lambda x: x.quantile(q)))

                    elif op_name in ['prop', 'inf_threshold']:
                        # Opérations spéciales
                        # Créer une version hashable du tuple
                        hashable_op = (op_name, op_params_tuple)
                        special_ops[hashable_op] = self.list_var_of_interest
        
        return agg_dict, weighted_ops, special_ops

    # Méthode auxiliaire de calcul des aggrégats avec pandas.agg
    def _compute_aggregations(
        self, 
        data: pd.DataFrame, 
        list_var_groupby: Optional[List[str]], 
        agg_dict: Dict[str, List[Tuple[str, str]]]
    ) -> pd.DataFrame:
        """
        Perform standard aggregations using pandas.agg().

        Args:
            data (pd.DataFrame):
                Data to aggregate.
            list_var_groupby (Optional[List[str]]):
                Grouping variables.
            agg_dict (Dict[str, List[Tuple[str, str]]]):
                Aggregation dictionary.

        Returns:
            pd.DataFrame:
                Aggregated results.
        """
        # Cas où le dictionnaire en argument est vide
        if not agg_dict:
            return pd.DataFrame()
        
        # Distinction où une liste de colonnes selon lesquelles agréger est spécifiée
        if list_var_groupby:
            # /!\ Il est plus optimisé d'effectuer un groupby sur une colonne d'entiers catégoriels que sur plusieurs colonnes de strings qui consomment plus de RAM

            # Calcul du groupby
            result = data.groupby(list_var_groupby, as_index=True, observed=True).agg(agg_dict)
            # Aplatir les colonnes multi-niveaux
            result.columns = [col[0] if col[1] == '' else col[1] for col in result.columns]
        else:
            # Pour le cas sans groupby, on doit calculer chaque opération et reformater le résultat
            # pour avoir une seule ligne avec des colonnes nommées 'var_operation'
            results_dict = {}

            for var, operations in agg_dict.items():
                for col_name, func in operations:
                    # Calculer l'agrégation pour cette variable et cette opération
                    if isinstance(func, str):
                        # Fonction standard pandas
                        value = getattr(data[var], func)()
                    else:
                        # Fonction lambda ou callable
                        value = func(data[var])
                    results_dict[col_name] = value

            # Créer un DataFrame avec une seule ligne
            result = pd.DataFrame([results_dict])

        return result

    # Méthode auxiliaire de calcul des aggrégats pondérés avec pandas.agg si possible
    def _compute_weighted_aggregations(
        self, 
        data: pd.DataFrame, 
        list_var_groupby: Optional[List[str]], 
        weighted_ops: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Compute weighted aggregations using pandas.agg() when possible.

        Args:
            data (pd.DataFrame):
                Data to aggregate.
            list_var_groupby (Optional[List[str]]):
                Grouping variables.
            weighted_ops (Dict[str, Any]):
                Weighted operations to perform.

        Returns:
            pd.DataFrame:
                Weighted aggregation results.
        """
        # Cas où le dictionnaire est vide ou la variable de poids non spécifiée
        if not weighted_ops or not self.var_weights:
            return pd.DataFrame()
        
        # Initialisation de la liste résultats
        results = []
        
        # Traitement des opérations pondérées qui peuvent utiliser agg()
        for operation, vars_list in weighted_ops.items():
            # Somme
            if operation == 'sum' and self.var_weights not in vars_list:
                
                if list_var_groupby:
                    # Somme pondérée via agg()
                    weighted_data = pd.concat([data[list_var_groupby], data[vars_list].multiply(data[self.var_weights], axis=0)], axis=1)
                    agg_dict = {var: [(f"{var}_sum", 'sum')] for var in vars_list}
                    # /!\ Il est plus optimisé d'effectuer un groupby sur une colonne d'entiers catégoriels que sur plusieurs colonnes de strings qui consomment plus de RAM

                    result = weighted_data.groupby(list_var_groupby, as_index=True).agg(agg_dict)
                else:
                    # Somme pondérée via agg()
                    weighted_data = data[vars_list].multiply(data[self.var_weights], axis=0)
                    agg_dict = {var: [(f"{var}_sum", 'sum')] for var in vars_list}
                    result = weighted_data.agg(agg_dict).T
                # Renommage des colonnes
                result.columns = [f"{var}_sum" for var in vars_list]
                # Ajout du résultat
                results.append(result)
            # Moyenne
            elif operation == 'mean':
                # Moyenne pondérée via agg()

                if list_var_groupby:
                    weighted_data = pd.concat([data[list_var_groupby], data[vars_list].multiply(data[self.var_weights], axis=0)], axis=1)
                    sum_weighted = weighted_data.groupby(list_var_groupby, as_index=True).sum()
                    # /!\ Il est plus optimisé d'effectuer un groupby sur une colonne d'entiers catégoriels que sur plusieurs colonnes de strings qui consomment plus de RAM
                    
                    sum_weights = data.groupby(list_var_groupby, as_index=True, observed=True)[self.var_weights].sum()
                    result = sum_weighted.divide(sum_weights, axis=0)
                else:
                    weighted_data = data[vars_list].multiply(data[self.var_weights], axis=0)
                    sum_weighted = weighted_data.sum()
                    sum_weights = data[self.var_weights].sum()
                    result = (sum_weighted / sum_weights).to_frame().T
                
                result.columns = [f"{var}_mean" for var in vars_list]
                results.append(result)
            # Quantiles
            elif operation == 'median' or (isinstance(operation, tuple) and operation[0] == 'quantile'):
                # Quantiles pondérés - nécessite la fonction spéciale
                if operation == 'median':
                    q = 0.5
                    suffix = 'q0.5'
                else:
                    # Reconvertir le tuple de tuples en dict
                    op_params = dict(operation[1])
                    q = op_params['q']
                    suffix = f'q{q}'
                
                if list_var_groupby:
                    result = data.groupby(list_var_groupby, as_index=True, observed=True).apply(
                        lambda x: weighted_quantile(
                            data=x,
                            vars_of_interest=vars_list,
                            var_weights=self.var_weights,
                            q=q
                        )
                    )
                else:
                    result = weighted_quantile(
                        data=data,
                        vars_of_interest=vars_list,
                        var_weights=self.var_weights,
                        q=q
                    ).to_frame().T
                
                result.columns = [f"{var}_{suffix}" for var in vars_list]
                results.append(result)
        
        # Concaténation de tous les résultats
        if results:
            return pd.concat(results, axis=1)
        else:
            return pd.DataFrame()

    # Méthode auxiliaire de calcul des aggrégats spéciaux
    def _compute_special_aggregations(
        self, 
        data: pd.DataFrame, 
        list_var_groupby: Optional[List[str]], 
        special_ops: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Compute special aggregations not supported by pandas.agg().

        Args:
            data (pd.DataFrame):
                Data to aggregate.
            list_var_groupby (Optional[List[str]]):
                Grouping variables.
            special_ops (Dict[str, Any]):
                Special operations to perform.

        Returns:
            pd.DataFrame:
                Special aggregation results.
        """
        # Cas où le dictionnaire est vide
        if not special_ops:
            return pd.DataFrame()
        
        # Initialisation de la liste des résultats
        results = []
        
        # Parcours des opérations
        for operation, vars_list in special_ops.items():
            if isinstance(operation, str):
                # Cas où l'opération est une chaine de caractères
                if operation == 'majority':
                    result = self._compute_majority(data, list_var_groupby, vars_list)
                    results.append(result)
                
                elif operation == 'max_sum_effectif':
                    if self.var_count:
                        for var_id in self.var_count:
                            result = self._compute_max_sum_effectif(
                                data, list_var_groupby, vars_list, var_id
                            )
                            results.append(result)
            
            elif isinstance(operation, tuple):
                # Cas des opérations à paramètres
                op_name = operation[0]
                # Reconvertir le tuple de tuples en dict
                op_params = dict(operation[1])

                if op_name == 'prop':
                    result = self._compute_prop(
                        data, list_var_groupby, vars_list,
                        op_params['var_ref']
                    )
                    results.append(result)

                elif op_name == 'inf_threshold':
                    result = self._compute_inf_threshold(
                        data, list_var_groupby, vars_list,
                        op_params['var_threshold'], op_params['threshold']
                    )
                    results.append(result)
        
        # Fusionner tous les résultats
        if results:
            return pd.concat(results, axis=1)
        else:
            return pd.DataFrame()

    # Méthode auxiliaire de calcul des opérations sur les données
    def _compute_operations(
        self,
        data: pd.DataFrame,
        list_var_groupby: Optional[List[str]],
        iterable_operations: Union[Dict[str, List[str]], List[str]]
    ) -> pd.DataFrame:
        """
        Compute all operations by delegating to appropriate methods.

        Args:
            data (pd.DataFrame):
                Data to process.
            list_var_groupby (Optional[List[str]]):
                Grouping variables.
            iterable_operations (Union[Dict[str, List[str]], List[str]]):
                Operations to perform.

        Returns:
            pd.DataFrame:
                Combined results from all operations.
        """
        # Préparation des dictionnaires d'opérations
        agg_dict, weighted_ops, special_ops = self._prepare_agg_dict(iterable_operations)
        
        # Gestiion du cas particulier de la somme des poids
        if self.var_weights and isinstance(iterable_operations, dict):
            if 'sum' in iterable_operations and self.var_weights in iterable_operations['sum']:
                # Ajouter la somme des poids au agg_dict
                if self.var_weights not in agg_dict:
                    agg_dict[self.var_weights] = []
                agg_dict[self.var_weights].append((f"{self.var_weights}_sum", 'sum'))
                
                # Modifier weighted_ops pour exclure var_weights
                if 'sum' in weighted_ops:
                    weighted_ops['sum'] = [v for v in weighted_ops['sum'] if v != self.var_weights]
                    if not weighted_ops['sum']:
                        del weighted_ops['sum']
        
        # Calcul chaque type d'agrégation
        # Initialisation de la liste résultat
        results = []
        
        # Agrégations standard
        std_results = self._compute_aggregations(data, list_var_groupby, agg_dict)
        if not std_results.empty:
            results.append(std_results)
        
        # Agrégations pondérées
        weighted_results = self._compute_weighted_aggregations(data, list_var_groupby, weighted_ops)
        if not weighted_results.empty:
            results.append(weighted_results)
        
        # Agrégations spéciales
        special_results = self._compute_special_aggregations(data, list_var_groupby, special_ops)
        if not special_results.empty:
            results.append(special_results)
        
        # Concaténation de tous les résultats
        if results:
            return pd.concat(results, axis=1)
        else:
            return pd.DataFrame()

    # Métholde auxiliaire de calcul du total
    def _compute_total(
        self, iterable_operations: Union[Dict[str, List[str]], List[str]]
    ) -> pd.DataFrame:
        """
        Compute total aggregations without groupby.

        Args:
            iterable_operations (Union[Dict[str, List[str]], List[str]]):
                Operations to perform.

        Returns:
            pd.DataFrame:
                Total aggregations with appropriate index.
        """
        # Calcul du total
        data_total = self._compute_operations(
            data=self.data_source,
            list_var_groupby=None,
            iterable_operations=iterable_operations
        )
        
        # Gestion de l'index
        if len(self.list_var_groupby) > 1:
            data_total.index = pd.MultiIndex.from_tuples(
                [["Total"] * len(self.list_var_groupby)], names=self.list_var_groupby
            )
        else:
            data_total.index = ["Total"]
        
        return data_total

    # Méthode auxiliaire de calcul des sous-totaux
    def _compute_under_total(
        self, iterable_operations: Union[Dict[str, List[str]], List[str]]
    ) -> pd.DataFrame:
        """
        Generate subtotals for combinations of grouping variables.

        Args:
            iterable_operations (Union[Dict[str, List[str]], List[str]]):
                Operations to perform.

        Returns:
            pd.DataFrame:
                Subtotals for variable combinations.
        """
        # Parcours de toutes les combinaisons possibles de niveaux
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
                data_sub_total = self._compute_operations(
                    data=self.data_source,
                    list_var_groupby=list_var_sub_groupby,
                    iterable_operations=iterable_operations
                )
                
                # Modification de l'index pour ajouter "Total" aux positions appropriées
                sub_total_index_res = []
                for sub_total_index in data_sub_total.index:
                    new_index = list(sub_total_index) if isinstance(sub_total_index, tuple) else [sub_total_index]
                    
                    for i_remove in sorted(combination):
                        new_index.insert(i_remove, "Total")
                    
                    sub_total_index_res.append(tuple(new_index))
                
                data_sub_total.index = pd.MultiIndex.from_tuples(
                    sub_total_index_res, names=self.list_var_groupby
                )
                
                # Ajout à la liste résultat
                list_sub_total.append(data_sub_total)
        
        # Concaténation des jeux de données résultats
        if list_sub_total:
            return pd.concat(list_sub_total, axis=0, join="outer")
        else:
            return pd.DataFrame()

    # Méthodes auxiliaires pour les opérations spéciales
    # Méthode de calcul de la modalité majoritaire
    def _compute_majority(
        self,
        data: pd.DataFrame,
        list_var_groupby: Optional[List[str]],
        vars_list: List[str]
    ) -> pd.DataFrame:
        """
        Compute majority value for specified variables.
        """
        # Initialisation de la liste résultat
        results = []

        # Parcours des variables
        for var in vars_list:
            # Utiliser le DataFrame complet si la variable n'est pas dans data
            if var not in data.columns:
                # Créer un DataFrame combiné avec les colonnes nécessaires
                if list_var_groupby:
                    cols_needed = list_var_groupby + [var]
                else:
                    cols_needed = [var]
                if self.var_weights:
                    cols_needed.append(self.var_weights)
                data_for_var = self._data_source_full[cols_needed].copy()
            else:
                data_for_var = data

            if list_var_groupby:
                if self.var_weights:
                    result = data_for_var.groupby(list_var_groupby, as_index=True, observed=True).apply(
                        lambda x: x.groupby(var)[self.var_weights].sum().idxmax()
                    ).to_frame(f"{var}_majority")
                else:
                    result = data_for_var.groupby(list_var_groupby, as_index=True, observed=True)[var].apply(
                        lambda x: x.value_counts().idxmax()
                    ).to_frame(f"{var}_majority")
            else:
                if self.var_weights:
                    result = pd.DataFrame(
                        [data_for_var.groupby(var)[self.var_weights].sum().idxmax()],
                        columns=[f"{var}_majority"]
                    )
                else:
                    result = pd.DataFrame(
                        [data_for_var[var].value_counts().idxmax()],
                        columns=[f"{var}_majority"]
                    )

            results.append(result)

        return pd.concat(results, axis=1)

    # Méthode de calcul du max/sum
    def _compute_max_sum_effectif(
        self,
        data: pd.DataFrame,
        list_var_groupby: Optional[List[str]],
        vars_list: List[str],
        var_id: str
    ) -> pd.DataFrame:
        """
        Compute max/sum ratio for specified variables.
        """
        # Création des données pondérées si nécessaire
        if self.var_weights:
            weighted_data = data[vars_list].multiply(data[self.var_weights], axis=0)
        else:
            weighted_data = data[vars_list]

        if list_var_groupby:
            # Ajouter les colonnes de groupby aux données pondérées
            weighted_data_with_groups = pd.concat([data[list_var_groupby], weighted_data], axis=1)
            max_vals = weighted_data_with_groups.groupby(list_var_groupby, as_index=True, observed=True).max()
            sum_vals = weighted_data_with_groups.groupby(list_var_groupby, as_index=True, observed=True).sum()
            result = max_vals / sum_vals
        else:
            max_vals = weighted_data.max()
            sum_vals = weighted_data.sum()
            result = (max_vals / sum_vals).to_frame().T

        result.columns = [f"{var}_{var_id}_max/sum" for var in vars_list]
        return result

    # Méthode de calcul de la proportion
    def _compute_prop(
        self,
        data: pd.DataFrame,
        list_var_groupby: Optional[List[str]],
        vars_list: List[str],
        var_ref: str
    ) -> pd.DataFrame:
        """
        Compute proportions relative to a reference variable.
        """
        # Création des données pondérées si nécessaire
        if self.var_weights:
            weighted_data = data[vars_list].multiply(data[self.var_weights], axis=0)
            weighted_ref = data[var_ref] * data[self.var_weights]
        else:
            weighted_data = data[vars_list]
            weighted_ref = data[var_ref]

        # Calcul des proportions
        if list_var_groupby:
            # Ajouter les colonnes de groupby aux données pondérées
            weighted_data_with_groups = pd.concat([data[list_var_groupby], weighted_data], axis=1)
            sum_vals = weighted_data_with_groups.groupby(list_var_groupby, as_index=True, observed=True).sum()

            # Pour weighted_ref (qui est une Series), créer un DataFrame temporaire
            weighted_ref_df = pd.concat([data[list_var_groupby], weighted_ref.to_frame()], axis=1)
            sum_ref = weighted_ref_df.groupby(list_var_groupby, as_index=True, observed=True).sum().iloc[:, 0]

            result = sum_vals.divide(sum_ref, axis=0)
        else:
            sum_vals = weighted_data.sum()
            sum_ref = weighted_ref.sum()
            result = (sum_vals / sum_ref).to_frame().T

        result.columns = [f"{var}_{var_ref}_prop" for var in vars_list]
        return result

    # Méthode de calcul de la quantité inférieure à un seuil
    def _compute_inf_threshold(
        self,
        data: pd.DataFrame,
        list_var_groupby: Optional[List[str]],
        vars_list: List[str],
        var_threshold: str,
        threshold: Union[int, float]
    ) -> pd.DataFrame:
        """
        Compute proportion of unique values below threshold.
        """
        # Utiliser le DataFrame complet si var_threshold n'est pas dans data
        if var_threshold not in data.columns:
            # Créer un DataFrame combiné avec les colonnes nécessaires
            cols_needed = list(set(list_var_groupby + vars_list + [var_threshold]))
            data_for_threshold = self._data_source_full[cols_needed].copy()
        else:
            data_for_threshold = data

        # Filtre des données sous le seuil
        data_below = data_for_threshold[data_for_threshold[var_threshold] < threshold]

        # Comptage des modalités
        if list_var_groupby:
            nunique_below = data_below.groupby(list_var_groupby, as_index=True, observed=True)[vars_list].nunique()
            nunique_total = data_for_threshold.groupby(list_var_groupby, as_index=True, observed=True)[vars_list].nunique()
            result = nunique_below / nunique_total
        else:
            nunique_below = data_below[vars_list].nunique()
            nunique_total = data_for_threshold[vars_list].nunique()
            result = (nunique_below / nunique_total).to_frame().T

        result.columns = [f"{var}_inf_{threshold}" for var in vars_list]
        return result