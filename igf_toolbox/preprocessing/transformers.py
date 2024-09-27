""" 
TO DO :
- un module permettant d'ajouter de l'auto-corrélation spatiale
"""

from itertools import product
from typing import Iterable, List, Optional, Union

# Importation des modules
# Modules de bases
import numpy as np
import pandas as pd
# Modules sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

# Fonction auxiliaire
from ..utils._auxiliary import _separate_category_modality_from_dummy

### Partie clustering / Machine learning


# OneHotEncoder des variables catégorielles
# SKlearn wrapper qui fonctionne avec des DataFrames et avec inverse_transform
class OneHotEncoder(TransformerMixin, BaseEstimator):
    """
    Custom OneHotEncoder for pandas DataFrames with inverse_transform support.

    This class provides a wrapper around pandas' get_dummies method for one-hot encoding
    with capabilities to inverse transform the one-hot encoded data back to its original categorical form.

    Parameters:
    - columns (iterable of str): Iterable of column names in the DataFrame to be one-hot encoded.
    - dummy_na (bool) : Boolean indicating whether to encode Nan.
    - dtype (type) : Type of the encoding dummies columns (should be in [bool, float, int]).

    Attributes:
    - list_col_dummy (iterable of str): Iterable of column names after one-hot encoding.

    Methods:
    - fit(X, y=None): Fit method. Does nothing in this transformer and returns self.
    - transform(X, y=None): Perform one-hot encoding on specified columns in the DataFrame.
    - inverse_transform(X, y=None): Convert one-hot encoded DataFrame back to its original categorical form.

    Example:
    >>> encoder = OneHotEncoder(columns=['color'])
    >>> df = pd.DataFrame({'color': ['red', 'green', 'blue'], 'value': [10, 20, 30]})
    >>> encoded_df = encoder.transform(df)
    >>> original_df = encoder.inverse_transform(encoded_df)
    """

    def __init__(
        self, columns: Iterable[str], dummy_na: bool = False, dtype: type = bool
    ) -> None:
        # Initialisation des paramètres
        self.columns = columns
        self.dummy_na = dummy_na
        self.dtype = dtype

    def fit(self, X, y=None) -> None:
        """
        Fit method for OneHotEncoder.

        Since the encoder is stateless (i.e., does not learn from the data), this method
        does nothing and simply returns the encoder instance.

        Parameters:
        - X (pd.DataFrame): The input data.
        - y (ignored): This parameter is ignored.

        Returns:
        - self: The encoder instance.
        """
        # Construction du jeu de données avec les dummies
        list_data_dummies = [
            pd.get_dummies(
                X[col],
                prefix=col,
                prefix_sep=" - ",
                dummy_na=self.dummy_na,
                dtype=self.dtype,
            )
            for col in self.columns
        ]

        if len(list_data_dummies) > 0:
            # Concaténation des variables binaires
            data_dummies = pd.concat(list_data_dummies, axis=1, join="outer")

            # Sauvegarde des colonnes pour pouvoir inverser la transformation
            self.list_col_dummy = data_dummies.columns.to_list()
        else:
            self.list_col_dummy = []

        return self

    # / ! \ A revoir car la syntaxe de get_dummies a évolué et devrait permettre de rendre la fonction plus concise, en particulier en lui donnant directement le DataFrame en argument
    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Perform one-hot encoding on the specified columns in the DataFrame.

        This method takes a DataFrame and one-hot encodes the columns specified during
        the encoder's instantiation. Other columns remain unchanged.

        Parameters:
        - X (pd.DataFrame): The input data containing columns to be one-hot encoded.
        - y (ignored): This parameter is ignored.

        Returns:
        - pd.DataFrame: A DataFrame with the specified columns one-hot encoded.
        """
        # Construction du jeu de données avec les dummies
        list_data_dummies = [
            pd.get_dummies(
                X[col],
                prefix=col,
                prefix_sep=" - ",
                dummy_na=self.dummy_na,
                dtype=self.dtype,
            )
            for col in self.columns
        ]

        if len(list_data_dummies) > 0:
            # Concaténation des variables binaires
            data_dummies = pd.concat(list_data_dummies, axis=1, join="outer").reindex(
                columns=pd.Index(data=self.list_col_dummy), method=None
            )
            # Complétion des Nan
            if self.dtype == bool:
                data_dummies.fillna(value=bool, inplace=True)
            elif self.dtype == float:
                data_dummies.fillna(value=0.0, inplace=True)
            elif self.dtype == int:
                data_dummies.fillna(value=0, inplace=True)
            else:
                raise ValueError(
                    f"Unknown dtype : {self.dtype}, should be in [bool, float, int]"
                )

            # Construction du jeu de données résultat
            X_transformed = pd.concat(
                [X.drop(self.columns, axis=1), data_dummies], axis=1, join="outer"
            )
        else:
            # Sauvegarde des colonnes pour pouvoir inverser la transformation
            self.list_col_dummy = []

            # Construction du jeu de données résultat
            X_transformed = X.copy()

        return X_transformed

    def inverse_transform(self, X: pd.DataFrame, y=None) -> None:
        """
        Convert a one-hot encoded DataFrame back to its original categorical form.

        This method takes a DataFrame with one-hot encoded columns and transforms it
        back to its original categorical form using the columns specified during
        the encoder's instantiation.

        Parameters:
        - X (pd.DataFrame): The one-hot encoded data.
        - y (ignored): This parameter is ignored.

        Returns:
        - pd.DataFrame: A DataFrame with the original categorical data.
        """
        if len(self.list_col_dummy) > 0:
            # Création d'un dataframe d'appariement
            data_dummies_categories = pd.DataFrame(
                data=self.list_col_dummy,
                index=np.arange(len(self.list_col_dummy)),
                columns=["dummies"],
            )
            data_dummies_categories = data_dummies_categories.apply(
                func=lambda x: _separate_category_modality_from_dummy(value=x), axis=1
            )

            # Initialisation de la liste résultat
            list_data_categories = []

            # Recherche de l'index du maximum
            for category in data_dummies_categories["categories"].unique():
                # Liste des dummies crées appartenant à la catégorie
                list_col_dummies_category = data_dummies_categories.loc[
                    data_dummies_categories["categories"] == category, "dummies"
                ].tolist()
                # Détermination de la dummy majoritaire pour chaque classe
                data_major_dummy_category = (
                    X[list_col_dummies_category]
                    .idxmax(axis="columns")
                    .to_frame()
                    .rename({0: "dummies_category"}, axis=1)
                )
                # Modification du nom de l'index pour s'assurer de la transformation
                data_major_dummy_category.index.name = "index"
                # Appariement avec les modalités
                data_category = (
                    pd.merge(
                        left=data_major_dummy_category.reset_index(),
                        right=data_dummies_categories[["dummies", "modalities"]],
                        left_on="dummies_category",
                        right_on="dummies",
                        how="left",
                        validate="many_to_one",
                    )
                    .set_index("index")[["modalities"]]
                    .rename({"modalities": category}, axis=1)
                )
                # Ajout à la liste résultat
                list_data_categories.append(data_category)

            # Concaténation des variables catégorielles
            data_categories = pd.concat(list_data_categories, axis=1, join="outer")

            # Construction du jeu de données résultat
            X_inverse_transformed = pd.concat(
                [X.drop(self.list_col_dummy, axis=1), data_categories],
                axis=1,
                join="outer",
            )
        else:
            X_inverse_transformed = X.copy()

        return X_inverse_transformed


# Adaptation pour les pandas.DataFrame de la classe sklearn de standardisation
class StandardScalerTransformer(TransformerMixin, BaseEstimator):
    """
    DataFrame Wrapper for the StandardScaler from sklearn.preprocessing.

    This class provides a simple wrapper around the StandardScaler from scikit-learn to work with
    pandas DataFrames. It standardizes features by removing the mean and scaling to unit variance,
    while retaining DataFrame structure, column names, and index.

    Attributes:
    - StandardScaler (StandardScaler): Instance of scikit-learn's StandardScaler.

    Methods:
    - fit(X, y=None): Compute the mean and standard deviation to be used for later scaling.
    - transform(X, y=None): Perform standard scaling (mean centering and variance normalization) on the data.
    - inverse_transform(X, y=None): Scale the data back to its original state.

    Example:
    >>> df_scaler = StandardScalerTransformer()
    >>> df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]})
    >>> df_scaled = df_scaler.fit_transform(df)
    >>> df_original = df_scaler.inverse_transform(df_scaled)
    """

    def __init__(self) -> None:
        """
        Initialize the StandardScalerTransformer.

        This method creates an instance of the StandardScaler from scikit-learn.
        """
        # Initialisation du StandardScaler
        self.StandardScaler = StandardScaler()

    def fit(self, X, y=None) -> None:
        """
        Compute the mean and standard deviation for standard scaling.

        This method fits the internal StandardScaler to the provided data.

        Parameters:
        - X (pd.DataFrame): The input data to fit the scaler.
        - y (ignored): This parameter is ignored.

        Returns:
        - self: The fitted transformer instance.
        """
        # Entrainement du StandardScaler
        self.StandardScaler.fit(X=X, y=y)
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Perform standard scaling on the provided data.

        This method uses the internal StandardScaler to perform the scaling and then
        returns the result as a pandas DataFrame with the same column names and index.

        Parameters:
        - X (pd.DataFrame): The input data to scale.
        - y (ignored): This parameter is ignored.

        Returns:
        - pd.DataFrame: The scaled data.
        """
        # Transformation du jeu de données en concervant les index
        X_transformed = pd.DataFrame(
            data=self.StandardScaler.transform(X=X), index=X.index, columns=X.columns
        )
        return X_transformed

    def inverse_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Scale the data back to its original state.

        This method uses the internal StandardScaler to perform the inverse scaling and
        then returns the result as a pandas DataFrame with the same column names and index.

        Parameters:
        - X (pd.DataFrame): The scaled input data to inverse scale.
        - y (ignored): This parameter is ignored.

        Returns:
        - pd.DataFrame: The original unscaled data.
        """
        # Inversion de la transformation en conservant les index
        X_inverse_transformed = pd.DataFrame(
            data=self.StandardScaler.inverse_transform(X=X),
            index=X.index,
            columns=X.columns,
        )
        return X_inverse_transformed


# Ajout les labels d'appartenance aux différents clusters au jeu de données
class ClusteringTransformer(TransformerMixin, BaseEstimator):
    """
    A transformer for adding cluster labels to a dataset using a clustering estimator.

    Parameters
    ----------
    estimator : estimator object
        The clustering estimator (e.g., KMeans, DBSCAN) to use for labeling.
    sample_weight : array-like or None, default=None
        Sample weights to apply during the clustering. If None, no sample weights will be used.

    Attributes
    ----------
    labels_ : pandas.Series
        The cluster labels assigned to each sample in the fitted dataset.

    Methods
    -------
    fit(X, y=None)
        Fit the clustering estimator to the input data and assign cluster labels.
    predict(X)
        Predict cluster labels for the input data.
    transform(X, y=None)
        Transform the input data by adding cluster labels as a new column.

    Notes
    -----
    This transformer is designed to work with clustering estimators from scikit-learn.

    Examples
    --------
    >>> from sklearn.cluster import KMeans
    >>> from sklearn.datasets import make_blobs
    >>> data, _ = make_blobs(n_samples=100, centers=3, random_state=42)
    >>> estimator = KMeans(n_clusters=3, random_state=42)
    >>> transformer = ClusteringTransformer(estimator)
    >>> transformed_data = transformer.fit_transform(data)

    The resulting `transformed_data` DataFrame will include a 'labels' column with cluster assignments.
    """

    def __init__(self, estimator, sample_weight: Union[pd.Series, None] = None) -> None:

        # Initialisation de l'estimateur
        self.estimator = estimator
        # Initialisation des poids d'entrainement
        self.sample_weight = sample_weight

    def fit(self, X: pd.DataFrame, y=None) -> None:
        """
        Fit the clustering estimator to the input data and assign cluster labels.

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features)
            The input data.
        y : array-like, shape (n_samples,), default=None
            The target labels (ignored in unsupervised clustering).

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Restriction aux observations qui n'ont pas été supprimées jusqu'alors
        if self.sample_weight is not None:
            sample_weight_reduced = self.sample_weight.loc[
                self.sample_weight.index.isin(X.index)
            ]
            # Entrainement du modèle
            try:
                self.estimator.fit(X=X, y=y, sample_weight=sample_weight_reduced)
            except:
                self.estimator.fit(X=X, y=y)
                pass
        else:
            # Entrainement du modèle
            self.estimator.fit(X=X, y=y)

        # Extraction des labels d'assignation à chaque cluster
        self.labels = pd.DataFrame(
            data=self.estimator.labels_, index=X.index, columns=["labels"]
        )

        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict cluster labels for the input data.

        Parameters
        ----------
        X : array-like or pd.DataFrame, shape (n_samples, n_features)
            The input data for which to predict cluster labels.

        Returns
        -------
        data_pred : pd.DataFrame, shape (n_samples, 1)
            A DataFrame with cluster labels as a 'labels' column.
        """
        # Prédiction de l'array résultat
        array_pred = self.estimator.predict(X=X)
        # Conversion en DataFrame
        data_pred = pd.DataFrame(data=array_pred, index=X.index, columns=["labels"])

        return data_pred

    def transform(self, X: pd.DataFrame, y=None) -> None:
        """
        Transform the input data by adding cluster labels as a new column.

        Parameters
        ----------
        X : array-like or pd.DataFrame, shape (n_samples, n_features)
            The input data to transform.

        Returns
        -------
        X_transformed : pd.DataFrame, shape (n_samples, n_features + 1)
            The input data with an additional 'labels' column containing cluster assignments.
        """
        # Une manière plus générale serait de prédire les labels dans un premier temps sur les X
        X_transformed = pd.concat([self.labels, X], axis=1, join="outer")

        return X_transformed


### Partie économétrie


class LogTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to apply logarithmic transformation to specified columns.

    This transformer applies a natural logarithm transformation to specified
    columns in the feature matrix. It's useful when working with skewed data
    or when the relationship between the variables is multiplicative in nature.

    Parameters:
    -----------
    list_col : list of str
        A list of column names in the DataFrame which will undergo the logarithm transformation.
    replace_inf : bool or numeric
        The value to replace the infinity values with.

    Attributes:
    -----------
    list_col : list of str
        Stored list of column names for transformation.

    Examples:
    ---------
    >>> import pandas as pd
    >>> data = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [6, 7, 8, 9, 10]})
    >>> transformer = LogTransformer(list_col=["A", "B"])
    >>> transformed_data = transformer.fit_transform(data)
    """

    def __init__(self, list_col: List[str], replace_inf: Optional[int] = 0) -> None:
        # Initialisation des paramètres
        self.list_col = list_col
        self.replace_inf = replace_inf

    def fit(self, X, y=None) -> None:
        """Fit the transformer.

        As this transformer does not require any fitting, the method simply returns self.

        Parameters:
        -----------
        X : DataFrame
            Feature matrix.

        y : Series or array-like, default=None
            Target variable. Not used in this transformer.

        Returns:
        --------
        self : LogTransformer
            The transformer instance.
        """
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Apply logarithmic transformation to the specified columns in X.

        Parameters:
        -----------
        X : DataFrame
            Feature matrix.

        y : Series or array-like, default=None
            Target variable. Not used in this transformer.

        Returns:
        --------
        X : DataFrame
            Transformed feature matrix.
        """
        # Transformation logarithmique des colonnes d'intérêt
        X[self.list_col] = np.log(X[self.list_col])

        # Remplacement des valeurs inf par la valeur choisie
        # Le .replace([-np.inf, np.inf], to_replace) ne fonctionne pas
        if not isinstance(self.replace_inf, bool):
            for var_log in self.list_col:
                X.loc[X[var_log].isin([np.inf, -np.inf]), var_log] = self.replace_inf

        return X

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Fit the transformer and apply the transformation.

        Parameters:
        -----------
        X : DataFrame
            Feature matrix.

        y : Series or array-like, default=None
            Target variable. Not used in this transformer.

        Returns:
        --------
        X : DataFrame
            Transformed feature matrix.
        """
        self.fit(X=X, y=y)
        return self.transform(X=X, y=y)


# Ajout d'une constante au jeu de données
class AddConstante(TransformerMixin, BaseEstimator):
    """
    Add a constant column to a DataFrame.

    This transformer appends a constant column named 'Constante' with value 1 to a DataFrame.
    It can be useful in certain statistical models where an intercept term is needed.

    Methods:
    - fit(X, y=None): Returns self.
    - transform(X, y=None): Adds the constant column to the data.
    - fit_transform(X, y=None): Fit to data and then transform it.

    Example:
    >>> adder = AddConstante()
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> df_transformed = adder.transform(df)
    >>> print(df_transformed)
       A  B  Constante
    0  1  4          1
    1  2  5          1
    2  3  6          1
    """

    def __init__(self) -> None:
        """
        Initialize the AddConstante transformer.
        """
        pass

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
        Add a constant column to the input DataFrame.

        Parameters:
        - X (pd.DataFrame): The input data to transform.
        - y (ignored): This parameter is ignored.

        Returns:
        - pd.DataFrame: The transformed data with an additional constant column.
        """
        data_res = X.copy()
        data_res["Constante"] = 1

        return data_res

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Fit to data and then transform it.

        Parameters:
        - X (pd.DataFrame): The input data to transform.
        - y (ignored): This parameter is ignored.

        Returns:
        - pd.DataFrame: The transformed data.
        """
        self.fit(X=X, y=y)
        return self.transform(X=X, y=y)


# Ajout d'un effet fixe au jeu de données
class AddFixedEffect(TransformerMixin, BaseEstimator):
    """
    Add fixed effect dummy variables to a DataFrame.

    This transformer appends dummy variables derived from a categorical column specified by `x_categorie`
    to the DataFrame. If the `no_trap` parameter is provided, the specified columns will be dropped
    to avoid the dummy variable trap.

    Attributes:
    - x_categorie (str): Name of the column from which to generate the dummy variables.
    - no_trap (list or str, optional): Name(s) of the dummy column(s) to drop for avoiding dummy variable trap.

    Methods:
    - fit(X, y=None): Returns self.
    - transform(X, y=None): Adds the dummy columns derived from `x_categorie` to the data.
    - fit_transform(X, y=None): Fit to data and then transform it.

    Example:
    >>> transformer = AddFixedEffect(x_categorie='Color', no_trap='Red')
    >>> df = pd.DataFrame({'Color': ['Red', 'Green', 'Blue'], 'Value': [1, 2, 3]})
    >>> df_transformed = transformer.transform(df)
    >>> print(df_transformed)
       Color  Value  Green  Blue
    0    Red      1      0     0
    1  Green      2      1     0
    2   Blue      3      0     1
    """

    def __init__(
        self, x_categorie: str, no_trap: Optional[Union[str, List[str], None]] = None
    ) -> None:
        """
        Initialize the AddFixedEffect transformer.

        Parameters:
        - x_categorie (str): The categorical column to convert into dummy variables.
        - no_trap (list or str, optional): Dummy column(s) to drop to avoid the dummy variable trap.
        """
        # Initialisation des valeurs des paramètres
        self.x_categorie = x_categorie
        self.no_trap = no_trap

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
        Add dummy variables to the input DataFrame.

        This method converts the specified `x_categorie` column to dummy variables and appends them to
        the input DataFrame. If the `no_trap` attribute is set, the specified dummy column(s) will be dropped.

        Parameters:
        - X (pd.DataFrame): The input data to transform.
        - y (ignored): This parameter is ignored.

        Returns:
        - pd.DataFrame: The transformed data with added dummy variables.
        """
        # Création et ajout du jeu de données d'indicatrices
        data_res = pd.concat(
            [X, pd.get_dummies(data=X[self.x_categorie])], axis=1, join="outer"
        )  # .drop(self.x_categorie, axis=1)

        # Supression des colonnes superflues
        if self.no_trap is not None:
            data_res.drop(self.no_trap, axis=1, inplace=True)

        return data_res

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Fit to data and then transform it.

        Parameters:
        - X (pd.DataFrame): The input data to transform.
        - y (ignored): This parameter is ignored.

        Returns:
        - pd.DataFrame: The transformed data.
        """
        self.fit(X=X, y=y)
        return self.transform(X=X, y=y)


# Ajout d'une variable d'intéraction au jeu de données
class AddInteraction(TransformerMixin, BaseEstimator):
    """
    Add interaction variables to a DataFrame based on the specified categorical columns.

    This transformer creates interaction dummy variables based on the given list of categorical columns.
    If the `list_no_trap` parameter is provided, the specified columns will be dropped
    to avoid the dummy variable trap.

    Attributes:
    - list_x_categorie (list): List of columns from which to generate the interaction dummy variables.
    - list_no_trap (list, optional): List of names of the dummy columns to drop for avoiding dummy variable trap.

    Methods:
    - fit(X, y=None): Returns self.
    - transform(X, y=None): Adds interaction dummy variables derived from `list_x_categorie` to the data.
    - fit_transform(X, y=None): Fit to data and then transform it.

    Example:
    >>> transformer = AddInteraction(list_x_categorie=['Color', 'Shape'], list_no_trap=['Red'])
    >>> df = pd.DataFrame({'Color': ['Red', 'Green'], 'Shape': ['Circle', 'Square'], 'Value': [1, 2]})
    >>> df_transformed = transformer.transform(df)
    >>> print(df_transformed)
       Color   Shape  Value  Green - Circle  Green - Square
    0    Red  Circle      1              0               0
    1  Green  Square      2              0               1
    """

    def __init__(
        self,
        list_x_categorie: List[str],
        list_no_trap: Optional[Union[List[str], None]] = None,
    ) -> None:
        """
        Initialize the AddInteraction transformer.

        Parameters:
        - list_x_categorie (list): The list of categorical columns to convert into interaction dummy variables.
        - list_no_trap (list, optional): Dummy columns to drop to avoid the dummy variable trap.
        """
        # Initialisation des valeurs des paramètres
        # Variables concernées
        self.list_x_categorie = list_x_categorie
        # Liste énumérant les modalités à retirer
        self.list_no_trap = list_no_trap

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
        Add interaction dummy variables to the input DataFrame.

        This method converts the specified columns in `list_x_categorie` into interaction dummy variables
        and appends them to the input DataFrame. If the `list_no_trap` attribute is set,
        the specified dummy column(s) will be dropped.

        Parameters:
        - X (pd.DataFrame): The input data to transform.
        - y (ignored): This parameter is ignored.

        Returns:
        - pd.DataFrame: The transformed data with added interaction dummy variables.
        """
        # Création et ajout du jeu de données d'indicatrices
        data_res = pd.concat(
            [X, pd.get_dummies(data=X[self.list_x_categorie], prefix="interact")],
            axis=1,
            join="outer",
        )  # .drop(self.list_x_categorie, axis=1)

        # Supression des colonnes superflues
        if self.list_no_trap is not None:
            data_res.drop(
                ["interact_" + no_trap for no_trap in self.list_no_trap],
                axis=1,
                inplace=True,
            )

        # Combinaison des indicatrices et création des intéractions

        # Création des combinaisons possibles
        list_modalite = [
            np.setdiff1d(X[x_categorie].unique(), self.list_no_trap).tolist()
            for x_categorie in self.list_x_categorie
        ]
        list_combinaison = list(product(*list_modalite))

        # Ajout des interactions
        for combinaison in list_combinaison:
            # Nom de la colonne d'intéraction
            nom_col = " - ".join(combinaison)
            # Initialisation de la colonne d'interaction
            data_res[nom_col] = 1
            for modalite in combinaison:
                # Mise à jour de la variable d'interaction
                data_res[nom_col] *= data_res["interact_" + modalite]

        # Suppression des indicatrices non interagies
        list_col_drop = [
            "interact_" + col
            for col in np.unique(
                np.setdiff1d(
                    X[self.list_x_categorie].values.reshape(-1), self.list_no_trap
                )
            )
        ]
        data_res.drop(list_col_drop, axis=1, inplace=True)

        return data_res

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Fit to data and then transform it.

        Parameters:
        - X (pd.DataFrame): The input data to transform.
        - y (ignored): This parameter is ignored.

        Returns:
        - pd.DataFrame: The transformed data.
        """
        self.fit(X=X, y=y)
        return self.transform(X=X, y=y)
