""""
TO DO :
- Ajouter un dropna qui s'applique aux X et aux y
"""

from typing import Dict, List, Optional

# Importation des modules
# Modules de bases
import numpy as np
import pandas as pd
# Modules sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest


# Exclusion des valeurs extrêmes à partir des forêts d'isolement
class ForestOutliersIsolation(TransformerMixin, BaseEstimator):
    """
    A transformer that uses Isolation Forest to identify and exclude outliers from a dataset.

    The Isolation Forest is an unsupervised machine learning algorithm that works by
    isolating observations by randomly selecting a feature and then randomly selecting
    a split value between the maximum and minimum values of that selected feature. It's
    primarily used for anomaly detection.

    Parameters:
    - n_estimators (int, optional): The number of base estimators in the ensemble. Default is 100.
    - random_state (int, optional): Seed used by the random number generator. Default is 42.

    Attributes:
    - model (IsolationForest): The trained Isolation Forest model.

    Methods:
    - fit(X, y=None): Fit the Isolation Forest model.
    - transform(X, y=None): Transform the data by excluding detected anomalies.

    Example:
    >>> fo = ForestOutliersIsolation()
    >>> X = pd.DataFrame({'A': [1, 2, 3, 100], 'B': [1, 2, 3, 100]})
    >>> fo.fit(X)
    >>> print(fo.transform(X))
       A  B
    0  1  1
    1  2  2
    2  3  3
    """

    def __init__(self, n_estimators: int = 100, random_state: int = 42) -> None:
        # Initialisation des paramètres
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y=None):
        """
        Fit the Isolation Forest model to the data.

        Parameters:
        - X (pd.DataFrame): The input data.
        - y (ignored): This parameter is ignored as Isolation Forest is an unsupervised method.

        Returns:
        - self: The fitted transformer.
        """
        self.model = IsolationForest(
            n_estimators=self.n_estimators, random_state=self.random_state
        ).fit(X=X, y=y)
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Remove detected anomalies from the data using the trained Isolation Forest model.

        Parameters:
        - X (pd.DataFrame): The input data.
        - y (ignored): This parameter is ignored.

        Returns:
        - pd.DataFrame: The transformed data with detected anomalies excluded.
        """
        # Copie indépendante du jeu de données
        X_res = X.copy()

        # Ajout de la prédiction des anomalies
        X_res["anomalies"] = self.model.predict(X_res)

        # Suppression des lignes concernées
        list_drop = X_res.loc[X_res["anomalies"] == -1].index.tolist()
        X_res.drop(list_drop, axis=0, inplace=True)

        # Suppression de la colonne 'anomalies'
        X_res.drop("anomalies", axis=1, inplace=True)

        return X_res


# Exclut les observations en fonction de leur valeur par rapport à un seuil
class ThresholdExcluder(TransformerMixin, BaseEstimator):
    """
    Exclude observations from a DataFrame based on threshold criteria.

    This transformer filters rows of a DataFrame based on specified column threshold values.
    Multiple criteria can be applied simultaneously, and rows which don't satisfy all the conditions
    will be excluded from the result.

    Attributes:
    - list_dict_params (List[Dict]): A list of dictionaries specifying the exclusion criteria.
        Each dictionary should contain:
        - 'variable': The column on which to apply the criterion.
        - 'operator': The comparison operator, which can be one of the following: '>', '>=', '<', '<=', '!='.
        - 'threshold': The value to compare against.
    - drop (bool) : A boolean indicating whether to drop or fill with np.nan excluded observations

    Methods:
    - fit(X, y=None): Returns self.
    - transform(X, y=None): Exclude observations based on the criteria.

    Example:
    >>> excluder = ThresholdExcluder([{'variable': 'A', 'operator': '>', 'threshold': 5},
    >>>                               {'variable': 'B', 'operator': '<=', 'threshold': 10}])
    >>> df = pd.DataFrame({'A': [1, 6, 3, 7], 'B': [5, 10, 20, 8]})
    >>> df_transformed = excluder.transform(df)
    """

    def __init__(self, list_dict_params: List[Dict], drop: Optional[bool] = True):
        """
        Initialize the ThresholdExcluder.

        Parameters:
        - list_dict_params (List[Dict]): A list of dictionaries specifying the exclusion criteria.
        """
        # Initialisation de la liste du dictionnaire de paramètres
        self.list_dict_params = list_dict_params

    def fit(self, X, y=None) -> None:
        """
        Return self.

        The fit method is implemented for compatibility with sklearn's TransformerMixin,
        but doesn't perform any actual computation.

        Parameters:
        - X (pd.DataFrame): The input data. Not used, only needed for compatibility.
        - y (ignored): This parameter is ignored.

        Returns:
        - self: The instance itself.
        """
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Exclude observations based on the specified criteria.

        Parameters:
        - X (pd.DataFrame): The input data to transform.
        - y (ignored): This parameter is ignored.

        Returns:
        - pd.DataFrame: The transformed data with observations not meeting the criteria excluded.
        """
        # Disjonction suivant la suppression
        if self.drop:
            # Initialisation de la série de booléens
            list_data_boolean = []

            # Parcours des dictionnaires de paramètres
            for dict_params in self.list_dict_params:
                if dict_params["operator"] == ">":
                    list_data_boolean.append(
                        (
                            X[dict_params["variable"]] > dict_params["threshold"]
                        ).to_frame()
                    )
                elif dict_params["operator"] == ">=":
                    list_data_boolean.append(
                        (
                            X[dict_params["variable"]] >= dict_params["threshold"]
                        ).to_frame()
                    )
                elif dict_params["operator"] == "<":
                    list_data_boolean.append(
                        (
                            X[dict_params["variable"]] < dict_params["threshold"]
                        ).to_frame()
                    )
                elif dict_params["operator"] == "<=":
                    list_data_boolean.append(
                        (
                            X[dict_params["variable"]] <= dict_params["threshold"]
                        ).to_frame()
                    )
                elif dict_params["operator"] == "!=":
                    list_data_boolean.append(
                        (
                            X[dict_params["variable"]] != dict_params["threshold"]
                        ).to_frame()
                    )
            # Concaténation des séries booléennes
            data_boolean = pd.concat(list_data_boolean, axis=1, join="outer")

            # Restriction aux observations respectant tous les seuils
            X_transformed = X.loc[data_boolean.all(axis=1)]
        else:
            # Copie indépendante du jeu de données
            X_transformed = X.copy()
            # Parcours des dictionnaires de paramètres
            for dict_params in self.list_dict_params:
                if dict_params["operator"] == ">":
                    X_transformed.loc[
                        (X[dict_params["variable"]] > dict_params["threshold"]),
                        dict_params["variable"],
                    ] = np.nan
                elif dict_params["operator"] == ">=":
                    X_transformed.loc[
                        (X[dict_params["variable"]] >= dict_params["threshold"]),
                        dict_params["variable"],
                    ] = np.nan
                elif dict_params["operator"] == "<":
                    X_transformed.loc[
                        (X[dict_params["variable"]] < dict_params["threshold"]),
                        dict_params["variable"],
                    ] = np.nan
                elif dict_params["operator"] == "<=":
                    X_transformed.loc[
                        (X[dict_params["variable"]] <= dict_params["threshold"]),
                        dict_params["variable"],
                    ] = np.nan
                elif dict_params["operator"] == "!=":
                    X_transformed.loc[
                        (X[dict_params["variable"]] != dict_params["threshold"]),
                        dict_params["variable"],
                    ] = np.nan

        return X_transformed


# Exclut les observations suivant leur position dans la distribution
class QuantileExcluder(TransformerMixin, BaseEstimator):
    """
    Exclude observations from a DataFrame based on quantile criteria.

    This transformer filters rows of a DataFrame based on specified column quantiles.
    Multiple criteria can be applied simultaneously, and rows which don't satisfy all the conditions
    will be excluded from the result.

    Attributes:
    - list_dict_params (List[Dict]): A list of dictionaries specifying the exclusion criteria.
        Each dictionary should contain:
        - 'variable': The column on which to apply the criterion.
        - 'operator': The comparison operator, which can be one of the following: 'left', 'right', 'both'.
        - 'threshold': The quantile value to compare against.
    - drop (bool) : A boolean indicating whether to drop or fill with np.nan excluded observations

    Methods:
    - fit(X, y=None): Returns self.
    - transform(X, y=None): Exclude observations based on the criteria.

    Example:
    >>> excluder = QuantileExcluder([list_dict_params={'variable': 'A', 'operator': 'left', 'threshold': 0.25},
    >>>                              {'variable': 'B', 'operator': 'right', 'threshold': 0.1}], drop=True)
    >>> df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8]})
    >>> df_transformed = excluder.transform(df)
    """

    def __init__(self, list_dict_params: List[Dict], drop: bool = True) -> None:
        """
        Initialize the QuantileExcluder.

        Parameters:
        - list_dict_params (List[Dict]): A list of dictionaries specifying the exclusion criteria.
        """
        # Initialisation de la liste du dictionnaire de paramètres
        self.list_dict_params = list_dict_params
        # Initialisation du booléen indiquant s'il faut supprimer les colonnes ou remplacer par un Nan les observations exclues
        self.drop = drop

    def fit(self, X, y=None) -> None:
        """
        Return self.

        The fit method is implemented for compatibility with sklearn's TransformerMixin,
        but doesn't perform any actual computation.

        Parameters:
        - X (pd.DataFrame): The input data. Not used, only needed for compatibility.
        - y (ignored): This parameter is ignored.

        Returns:
        - self: The instance itself.
        """
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Exclude observations based on the specified quantile criteria.

        Parameters:
        - X (pd.DataFrame): The input data to transform.
        - y (ignored): This parameter is ignored.

        Returns:
        - pd.DataFrame: The transformed data with observations not meeting the criteria excluded.
        """
        # Disjonction suivant la suppression
        if self.drop:
            # Initialisation de la série de booléens
            list_data_boolean = []

            # Parcours des dictionnaires de paramètres
            for dict_params in self.list_dict_params:
                if dict_params["operator"] == "left":
                    q = X[dict_params["variable"]].quantile(q=dict_params["threshold"])
                    list_data_boolean.append(
                        (X[dict_params["variable"]] > q).to_frame()
                    )
                elif dict_params["operator"] == "right":
                    q = X[dict_params["variable"]].quantile(
                        q=1 - dict_params["threshold"]
                    )
                    list_data_boolean.append(
                        (X[dict_params["variable"]] < q).to_frame()
                    )
                elif dict_params["operator"] == "both":
                    q = X[dict_params["variable"]].quantile(
                        q=[dict_params["threshold"], 1 - dict_params["threshold"]]
                    )
                    list_data_boolean.append(
                        (
                            (
                                X[dict_params["variable"]]
                                > q.loc[dict_params["threshold"]]
                            )
                            & (
                                X[dict_params["variable"]]
                                < q.loc[1 - dict_params["threshold"]]
                            )
                        ).to_frame()
                    )

            # Concaténation des séries booléennes
            data_boolean = pd.concat(list_data_boolean, axis=1, join="outer")

            # Restriction aux observations respectant tous les seuils
            X_transformed = X.loc[data_boolean.all(axis=1)].copy()
        else:
            # Copie indépendante du jeu de données
            X_transformed = X.copy()
            # Parcours des dictionnaires de paramètres
            for dict_params in self.list_dict_params:
                if dict_params["operator"] == "left":
                    q = X[dict_params["variable"]].quantile(q=dict_params["threshold"])
                    X_transformed.loc[
                        (X_transformed[dict_params["variable"]] < q),
                        dict_params["variable"],
                    ] = np.nan
                elif dict_params["operator"] == "right":
                    q = X[dict_params["variable"]].quantile(
                        q=1 - dict_params["threshold"]
                    )
                    X_transformed.loc[
                        (X_transformed[dict_params["variable"]] > q),
                        dict_params["variable"],
                    ] = np.nan
                elif dict_params["operator"] == "both":
                    q = X[dict_params["variable"]].quantile(
                        q=[dict_params["threshold"], 1 - dict_params["threshold"]]
                    )
                    X_transformed.loc[
                        (
                            (
                                X_transformed[dict_params["variable"]]
                                < q.loc[dict_params["threshold"]]
                            )
                            | (
                                X_transformed[dict_params["variable"]]
                                > q.loc[1 - dict_params["threshold"]]
                            )
                        ),
                        dict_params["variable"],
                    ] = np.nan

        return X_transformed


# Transformer permettant de supprimer une ou plusieurs colonnes
class ColumnExcluder(TransformerMixin, BaseEstimator):
    """
    Transformer that allows for the exclusion of one or multiple columns from a DataFrame.

    Provides methods for fitting to data and transforming data by dropping specified columns.

    Attributes:
    -----------
    list_col_drop : list
        List of column names to be dropped.

    Methods:
    --------
    fit(X, y=None) :
        Fits the transformer to the data. For this transformer, it's essentially a no-op but maintains consistency.

    transform(X, y=None) :
        Transforms the data by dropping the specified columns.

    fit_transform(X, y=None) :
        Fits and then transforms the data.

    Examples:
    ---------
    >>> col_excluder = ColumnExcluder(list_col_drop=['col1', 'col2'])
    >>> reduced_data = col_excluder.fit_transform(data)

    Notes:
    ------
    It's important to ensure that the columns specified in `list_col_drop` exist in the DataFrame. Otherwise, it may raise a KeyError.
    """

    def __init__(self, list_col_drop: List[str]) -> None:
        """
        Initializes the ColumnExcluder class.

        Parameters:
        -----------
        list_col_drop : list
            List of column names to be dropped from the DataFrame.
        """
        # Initialisation des paramètres
        self.list_col_drop = list_col_drop

    def fit(self, X, y=None) -> None:
        """
        Fits the transformer to the data. For this transformer, it's a no-op but is included for consistency.

        Parameters:
        -----------
        X : pd.DataFrame
            The data to be transformed.

        y : Ignored
            This parameter exists only for compatibility with scikit-learn pipeline and is not used.

        Returns:
        --------
        self
        """
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Transforms the data by dropping the specified columns.

        Parameters:
        -----------
        X : pd.DataFrame
            The data to be transformed.

        y : Ignored
            This parameter exists only for compatibility with scikit-learn pipeline and is not used.

        Returns:
        --------
        pd.DataFrame
            Transformed data with specified columns dropped.
        """
        return X.drop(self.list_col_drop, axis=1)

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Fits and then transforms the data.

        Parameters:
        -----------
        X : pd.DataFrame
            The data to be transformed.

        y : Ignored
            This parameter exists only for compatibility with scikit-learn pipeline and is not used.

        Returns:
        --------
        pd.DataFrame
            Transformed data with specified columns dropped.
        """
        self.fit(X=X, y=y)
        return self.transform(X=X, y=y)
