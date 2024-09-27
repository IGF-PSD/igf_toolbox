""" 
TO DO :
- une classe permettant d'estimer un modèle à effets aléatoire (éventuellement en choisissant entre modèle à effets aléatoires et effets fixes selon un test d'Hausman)
- une classe permettant d'estimer un modèle avec des retards de la variable endogène
- ajouter davantage d'attributs aux classes existantes pour pouvoir accéder plus facilement aux différents éléments estimés (par exemple les effets fixes dans un modèle de panel)
"""

from typing import Optional, Union

# Importation des modules
# modules de base
import numpy as np
import pandas as pd
# Linearmodels
from linearmodels.panel.model import PanelOLS
# Scikit-Learn
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
# Stat Models
from statsmodels.regression.linear_model import OLS, WLS


# Estimateur des moindres carrés ordinaires
class LeastSquaresEstimator(BaseEstimator):
    """
    A simple wrapper around the statsmodels OLS (Ordinary Least Squares) and WLS (Weighted Least Squares) regression models.

    Provides methods for fitting a linear regression model, making predictions, and summarizing the results.

    Attributes:
    -----------
    model : statsmodels regression model
        The internal regression model, either OLS or WLS.
    _coef : pd.Series
        The estimated coefficients from the regression model.

    Methods:
    --------
    fit(X, y, sample_weight=None) :
        Fits a regression model to the data.

    predict(X) :
        Uses the fitted model to predict the dependent variable for a new set of independent variables.

    summary() :
        Returns a summary of the estimated regression coefficients, their p-values, and 95% confidence intervals.

    rsquared() :
        Returns the R-squared and adjusted R-squared values for the fitted model.

    Examples:
    ---------
    >>> estimator = LeastSquaresEstimator()
    >>> estimator.fit(X_train, y_train)
    >>> predictions = estimator.predict(X_test)
    >>> summary_results = estimator.summary()
    >>> r2_values = estimator.rsquared()
    """

    def __init__(self) -> None:
        """Initializes the LeastSquaresEstimator class."""

    def fit(self, X, y, sample_weight: Optional[Union[pd.Series, None]] = None) -> None:
        """
        Fits a regression model to the data.

        Parameters:
        -----------
        X : pd.DataFrame
            The independent variables (explanatory variables).

        y : pd.Series
            The dependent variable (response variable).

        sample_weight : pd.Series, optional
            Optional weights for each observation. If provided, WLS is used; otherwise, OLS is used.

        Returns:
        --------
        None
        """
        # Estimation du modèle
        if sample_weight is None:
            self.model = OLS(endog=y, exog=X).fit()
        else:
            self.model = WLS(endog=y, exog=X, weights=sample_weight).fit()

        self._coef = self.model.params

        return self

    def predict(self, X):
        """
        Uses the fitted model to predict the dependent variable for a new set of independent variables.

        Parameters:
        -----------
        X : pd.DataFrame
            The independent variables for which to predict the dependent variable.

        Returns:
        --------
        pd.Series
            Predicted values of the dependent variable.
        """
        return self.model.predict(X)

    def summary(self) -> pd.DataFrame:
        """
        Returns a summary of the estimated regression coefficients, their p-values, and 95% confidence intervals.

        Returns:
        --------
        pd.DataFrame
            A dataframe with the estimated regression coefficients, their p-values, and 95% confidence intervals.
        """

        # Description des résultats du modèle
        coefs = self.model.params.to_frame().rename({0: "Coefficients"}, axis=1)
        p_values = self.model.pvalues.to_frame().rename({0: "P-valeurs"}, axis=1)
        ic_95 = self.model.conf_int(alpha=0.05).rename({0: "ICg", 1: "ICd"}, axis=1)

        # Concaténation des résultats
        data_res = pd.concat([coefs, p_values, ic_95], axis=1, join="outer")

        return data_res

    def rsquared(self) -> pd.DataFrame:
        """
        Returns the R-squared and adjusted R-squared values for the fitted model.

        Returns:
        --------
        pd.DataFrame
            A dataframe with the R-squared and adjusted R-squared values.
        """
        # Description de la pertinence du modèle
        data_r2 = pd.DataFrame(
            [[self.model.rsquared, self.model.rsquared_adj]],
            index=[0],
            columns=["R2", "R2Adj"],
        )
        return data_r2


# Estimateurs des moindres carrés ordinaires sur des données de panel
class PanelLeastSquaresEstimator(BaseEstimator):
    """
    An estimator for panel data using Ordinary Least Squares (OLS).

    Parameters
    ----------
    entity_effects : bool
        Whether to include entity (fixed) effects in the model.
    time_effects : bool
        Whether to include time effects in the model.
    drop_absorbed : bool, optional (default=False)
        If true, drops variables that are fully absorbed by the entity or time effects.
    cov_type : str, optional (default='unadjusted')
        Type of covariance matrix estimator to use.

    Attributes
    ----------
    model : PanelOLS
        The fitted model.

    Methods
    -------
    fit(X, y, sample_weight=None):
        Fits the model using the provided data.
    predict(X):
        Predicts the response variable using the provided data.
    summary():
        Returns a summary of the regression results.
    rsquared():
        Returns various R^2 measures for the model.
    estimated_effects():
        Returns the estimated entity and time effects.
    """

    def __init__(
        self,
        entity_effects: bool,
        time_effects: bool,
        drop_absorbed: Optional[bool] = False,
        check_rank: Optional[bool] = True,
        cov_type: Optional[str] = "unadjusted",
    ) -> None:
        # Initialisation des valeurs des booléens indiquant si les effets fixes individuels et temporels doivent être traités
        self.entity_effects = entity_effects
        self.time_effects = time_effects
        self.drop_absorbed = drop_absorbed
        # Vérification du rang
        self.check_rank = check_rank
        # Type d'estimation de la convariance
        self.cov_type = cov_type

        # return self

    def fit(self, X, y, sample_weight: Optional[Union[pd.Series, None]] = None) -> None:
        """
        Fit the model using the provided data.

        Parameters
        ----------
        X : DataFrame
            Feature matrix.
        y : Series
            Response variable.
        sample_weight : Series, optional
            Weights for each observation.

        Returns
        -------
        self : PanelLeastSquaresEstimator
            The instance itself.
        """
        # Par convention les Entity X Year index sont les deux premières colonnes si X est un ndarray
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X).set_index([0, 1])

        # Entrainement du modèle
        self.model = PanelOLS(
            dependent=y,
            exog=X,
            weights=sample_weight,
            entity_effects=self.entity_effects,
            time_effects=self.time_effects,
            drop_absorbed=self.drop_absorbed,
        ).fit(cov_type=self.cov_type)
        return self

    def predict(self, X):
        """
        Predict the response variable using the provided data.

        Parameters
        ----------
        X : DataFrame
            Feature matrix to predict response for.

        Returns
        -------
        Series
            Predicted values.
        """
        # Par convention les Entity X Year index sont les deux premières colonnes si X est un ndarray
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X).set_index([0, 1])

        return self.model.predict(X)

    def summary(self) -> pd.DataFrame:
        """
        Returns a summary of the regression results.

        Returns
        -------
        DataFrame
            A DataFrame with coefficients, p-values, and 95% confidence intervals.
        """
        # Description des résultats du modèle
        coefs = self.model.params.to_frame().rename(
            {"parameter": "Coefficients"}, axis=1
        )
        p_values = self.model.pvalues.to_frame().rename({"pvalue": "P-valeurs"}, axis=1)
        ic_95 = self.model.conf_int(level=0.95).rename(
            {"lower": "ICg", "upper": "ICd"}, axis=1
        )

        # Concaténation des résultats
        data_res = pd.concat([coefs, p_values, ic_95], axis=1, join="outer")

        return data_res

    def rsquared(self) -> pd.DataFrame:
        """
        Returns various R^2 measures for the model.

        Returns
        -------
        DataFrame
            A DataFrame containing various R^2 measures and the number of observations.
        """
        return pd.DataFrame(
            data=[
                [
                    self.model.nobs,
                    self.model.rsquared,
                    self.model.rsquared_inclusive,
                    self.model.rsquared_overall,
                    self.model.rsquared_between,
                    self.model.rsquared_within,
                ]
            ],
            index=[0],
            columns=[
                "n_observations",
                "R2",
                "R2 - Effets fixes inclus",
                "R2 - Effets fixes exclus",
                "R2 - Purgé des effets temporels",
                "R2 - Purgé des effets individuels",
            ],
        )

    def estimated_effects(self):
        """
        Returns the estimated entity and time effects.

        Returns
        -------
        DataFrame
            A DataFrame containing the estimated entity and time effects.
        """
        return self.model.estimated_effects
