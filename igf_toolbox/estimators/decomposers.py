# Importation des modules
# Modules de base
from typing import Optional

import pandas as pd
# Scikit-Learn
from sklearn.base import BaseEstimator

# Fonction de régression des moindres carrés ordinaires
from .regressors import LeastSquaresEstimator

# from igf_toolbox_python.estimators.regression import LeastSquaresEstimator


# Classe de décomposition de Oaxaca-Blinder
class OaxacaBlinder(BaseEstimator):
    """
    Oaxaca-Blinder Decomposition Estimator.

    This estimator implements the Oaxaca-Blinder decomposition to decompose the difference
    in means of a dependent variable between two groups into an explained and unexplained
    component.

    Parameters:
    -----------
    bifurcate : Series
        A binary series indicating the group of each observation.
        Observations belonging to the reference group are labeled as 1,
        and the other group as 0.

    method : {'pooled', 'other'}
        The method used for coefficient estimation:
        - 'pooled': Coefficients estimated from the pooled data.
        - 'other': Coefficients estimated from the reference group.

    Attributes:
    -----------
    bifurcate : Series
        Stored bifurcation series.

    method : str
        Stored method string.

    X_group_ref : DataFrame
        Features of the reference group.

    y_group_ref : Series
        Target variable of the reference group.

    X_other_group : DataFrame
        Features of the other group.

    y_other_group : Series
        Target variable of the other group.

    model : LeastSquaresEstimator
        Estimated linear regression model.

    _coef : Series or array-like
        Coefficients from the linear regression model.

    Examples:
    ---------
    >>> import pandas as pd
    >>> data = pd.DataFrame({"X1": [1, 2, 3, 4, 5], "y": [6, 7, 8, 9, 10], "group": [1, 1, 0, 0, 0]})
    >>> decomposer = OaxacaBlinder(bifurcate=data["group"], method="pooled")
    >>> decomposer.fit(data[["X1"]], data["y"])
    >>> result = decomposer.decompose()

    """

    def __init__(self, bifurcate: pd.Series, method: str) -> None:
        # Initialisation des paramètres
        # bifurcate est une série d'indicatrices du jeu de données parmi les X utilisée pour séparer les deux groupes
        # Les observations appartenant au groupe de référence sont notée 1
        self.bifurcate = bifurcate
        if method in ["pooled", "other"]:
            self.method = method
        else:
            raise ValueError(
                "Unknown method. 'method' should be in ['pooled', 'other']"
            )

    def fit(self, X, y) -> None:
        """
        Fit the OaxacaBlinder decomposer.

        Parameters:
        -----------
        X : DataFrame
            The feature matrix.

        y : Series or array-like
            The target variable.

        Returns:
        --------
        self : OaxacaBlinder
            The fitted decomposer.
        """
        # Décomposition des observations
        self.X_group_ref, self.y_group_ref = (
            X.loc[self.bifurcate == 1],
            y.loc[self.bifurcate == 1],
        )
        self.X_other_group, self.y_other_group = (
            X.loc[self.bifurcate == 0],
            y.loc[self.bifurcate == 0],
        )

        # Estimation du modèle
        self.model = LeastSquaresEstimator()
        if self.method == "pooled":
            self.model.fit(X, y)
        elif self.method == "other":
            self.model.fit(X_group_ref, y_group_ref)

        self._coef = self.model._coef

    def summary(self):
        """
        Summarize the estimated regression model.

        Returns:
        --------
        DataFrame
            Summary statistics of the estimated regression model.
        """
        # Résumé des résultats d'estimation du modèle
        return self.model.summary()

    def rsquared(self):
        """
        Compute the R-squared statistic.

        Returns:
        --------
        DataFrame
            R-squared and adjusted R-squared values.
        """
        # R2 du modèle
        return self.model.rsquared()

    def decompose(self, detailed: Optional[bool] = False) -> pd.DataFrame:
        """
        Apply the Oaxaca-Blinder decomposition.

        Parameters:
        -----------
        detailed : bool, default=False
            If True, the result includes the detailed effect for each feature.
            If False, only the aggregated effect is returned.

        Returns:
        --------
        DataFrame
            Decomposed effects, either detailed or aggregated, based on the 'detailed' parameter.
        """
        # Application de la décomposition de Oaxaca-Blinder
        if detailed:
            # Estimation détaillée de l'effet pour chacun des X
            # Valorisé au niveau du groupe de référence
            two_fold_decomposition = (
                (
                    (self.X_group_ref.mean(axis=0) - self.X_other_group.mean(axis=0))
                    * self._coef
                )
                .to_frame()
                .rename({0: "explained effect"}, axis=1)
            )
        else:
            two_fold_decomposition = [
                (self.y_group_ref.mean() - self.y_other_group.mean())
                - (
                    (self.X_group_ref.mean(axis=0) - self.X_other_group.mean(axis=0))
                    * self._coef
                ).sum(),
                (
                    (self.X_group_ref.mean(axis=0) - self.X_other_group.mean(axis=0))
                    * self._coef
                ).sum(),
                self.y_group_ref.mean() - self.y_other_group.mean(),
            ]
            # Conversion en DataFrame
            two_fold_decomposition = pd.Series(
                two_fold_decomposition,
                index=["Unexplained Effect", "Explained Effect", "Gap"],
                name="decomposition",
            ).to_frame()

        return two_fold_decomposition
