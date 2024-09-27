# Importation des modules
# modules de base
from typing import Optional, Union

import numpy as np
import pandas as pd
# Scikit-Learn
from sklearn.base import BaseEstimator
# Stat Models
from statsmodels.discrete.discrete_model import Logit, Probit


# Modèle Probit
class ProbitClassifier(BaseEstimator):
    """
    Probit model wrapped to a sklearn class
    Parameters
    ----------
    proba_threshold : float, default=0.5
        Threshold to distinguish forecast corresponding to the zero class and to the one class

    scoring : string, default='roc'
        Score to calculate via the score method
    regularized : bool, default=True
        Whether to add a regularisation constrain to select variables in the minimization process and ensure convergence

    method : string, default='l1'
        Type of regularization to apply

    alpha : float, default=0.01
        Regularization parameter

    Attributes
    ----------
    probit : Probit
        Probit fitted regressor
    """

    def __init__(
        self,
        proba_threshold: Optional[float] = 0.5,
        scoring: Optional[str] = "roc",
        regularized: Optional[bool] = False,
        method: Optional[str] = "l1",
        alpha: Optional[float] = 0.01,
    ) -> None:
        # Initialisation des paramètres du modèle
        self.scoring = scoring
        self.regularized = regularized
        self.method = method
        self.alpha = alpha
        self.trim_mode = "size"
        self.estimator_type = "classifier"
        self._estimator_type = "classifier"
        self.proba_threshold = proba_threshold

    def set_params(self, **parameters) -> None:
        """
        Change the parameters of the model
        Returns
        -------
        self : returns an instance of self
        """
        # Ajout des paramètres
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=True) -> dict:
        """
        Return a dictionnary containing the current parameters of the model
        Parameters
        ----------

        deep : bool, default=True
            Whether to return an independent copy of the parameters
        Returns
        -------

        DictParams : dictionnary
            Dictionnary containing the name of the parameter and its corresponding value
        """
        # Retourne les paramètres
        return {
            "alpha": self.alpha,
            "scoring": self.scoring,
            "regularized": self.regularized,
            "method": self.method,
            "trim_mode": self.trim_mode,
            "proba_threshold": self.proba_threshold,
        }

    def fit(self, X, y) -> None:
        """
        Fit the probit model.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary
        Returns
        -------
        self : returns an instance of self
        """
        # Entrainement du modèle
        if self.regularized:
            self.probit = Probit(y, X).fit_regularized(
                method=self.method,
                alpha=self.alpha,
                trim_mode=self.trim_mode,
                maxiter=100000,
            )
        else:
            self.probit = Probit(y, X).fit()
        # Extraction des coefficients et des classes
        self._coef = self.probit.params
        self.classes_ = np.unique(y)

        return self

    def predict(self, X) -> np.ndarray:
        """Predict class for X.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        Returns
        -------
        Prediction : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The predicted classes.
        """
        # Prédiction
        if self.proba_threshold:
            return (self.probit.predict(X) >= self.proba_threshold) * 1
        return self.probit.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        """Predict class associated with the class labelized as one for X.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        Returns
        -------
        Prediction : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The predicted probabilities.
        """
        # Prédiction de la probabilité d'appartenance à chaque classe
        proba = []
        XArray = np.array(X)
        for i in range(len(XArray)):
            proba.append(
                [
                    1 - self.probit.predict(XArray[i, :])[0],
                    self.probit.predict(XArray[i, :])[0],
                ]
            )

        return np.array(proba).reshape(-1, 2)

    def score(self, X_test, y_test):
        """Calculate the score for the label predicted by the model for X_test et les vrais labels y_test.
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        y_test : array-like of shape (n_samples, 1).
            The test samples
        Returns
        -------
        score : float or array-like
            The computed score.
        """
        # Calcul du ROC-AUC ou de l'Accuracy
        self.y_pred = ((self.predict(X_test) > 0.25) * 1) * True
        self.y_true = y_test
        if self.scoring == "accuracy":
            score = np.count_nonzero(
                np.add(
                    self.y_pred.tolist(),
                    np.multiply(self.y_true.iloc[:, 0].tolist(), -1),
                )
                == 0
            ) / len(self.y_pred)
        if self.scoring == "roc":
            from sklearn.metrics import roc_auc_score

            score = roc_auc_score(self.y_true, self.predict(X_test))
        return score

    def summary(self):
        """
        Give information about the estimation of the probit model
        Returns
        -------
        Summary : Information about the coefficients estimated by the model
        """
        # Résumé des résultats de l'estimation
        return self.probit.summary()

    def cov_matrix(self):
        """
        Give the covariance matrix of the model
        Returns
        -------
        Cov : Covariance matrix of the model
        """
        # Matrice de variance-covariance
        return self.probit.cov_params()


# Modèle Logit
class LogitClassifier(BaseEstimator):
    """
    Logit model wrapped to a sklearn class
    Parameters
    ----------
    proba_threshold : float, default=0.5
        Threshold to distinguish forecast corresponding to the zero class and to the one class

    scoring : string, default='roc'
        Score to calculate via the score method
    regularized : bool, default=True
        Whether to add a regularisation constrain to select variables in the minimization process and ensure convergence

    method : string, default='l1'
        Type of regularization to apply

    alpha : float, default=0.01
        Regularization parameter

    Attributes
    ----------
    logit : Logit
        Logit fitted regressor
    """

    def __init__(
        self,
        proba_threshold: Optional[float] = 0.5,
        scoring: Optional[str] = "roc",
        regularized: Optional[bool] = False,
        method: Optional[str] = "l1",
        alpha: Optional[float] = 0.01,
    ) -> None:
        # Initialisation du modèle
        self.scoring = scoring
        self.regularized = regularized
        self.method = method
        self.alpha = alpha
        self.trim_mode = "size"
        self.estimator_type = "classifier"
        self._estimator_type = "classifier"
        self.proba_threshold = proba_threshold

    def set_params(self, **parameters) -> None:
        """
        Change the parameters of the model
        Returns
        -------
        self : returns an instance of self
        """
        # Ajout des paramètres
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep: Optional[bool] = True) -> dict:
        """
        Return a dictionnary containing the current parameters of the model
        Parameters
        ----------

        deep : bool, default=True
            Whether to return an independent copy of the parameters
        Returns
        -------

        DictParams : dictionnary
            Dictionnary containing the name of the parameter and its corresponding value
        """
        # Retourne les paramètres
        return {
            "alpha": self.alpha,
            "scoring": self.scoring,
            "regularized": self.regularized,
            "method": self.method,
            "trim_mode": self.trim_mode,
            "proba_threshold": self.proba_threshold,
        }

    def fit(self, X, y):
        """
        Fit the probit model.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary
        Returns
        -------
        self : returns an instance of self
        """
        # Entrainement du modèle
        if self.regularized:
            self.logit = Logit(y, X).fit_regularized(
                method=self.method,
                alpha=self.alpha,
                trim_mode=self.trim_mode,
                maxiter=100000,
            )
        else:
            self.logit = Logit(y, X).fit()
        # Extraction du coefficient et des classes
        self._coef = self.logit.params
        self.classes_ = np.unique(y)

        return self

    def predict(self, X):
        """Predict class for X.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        Returns
        -------
        Prediction : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The predicted classes.
        """
        # Prédiction
        if self.proba_threshold:
            return (self.logit.predict(X) >= self.proba_threshold) * 1
        return self.logit.predict(X)

    def predict_proba(self, X):
        """Predict class associated with the class labelized as one for X.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        Returns
        -------
        Prediction : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The predicted probabilities.
        """
        # Prédiction de la probabilité d'appartenance à chacune des classes
        proba = []
        XArray = np.array(X)
        for i in range(len(XArray)):
            proba.append(
                [
                    1 - self.logit.predict(XArray[i, :])[0],
                    self.logit.predict(XArray[i, :])[0],
                ]
            )
        return np.array(proba).reshape(-1, 2)

    def score(self, X_test, y_test):
        """Calculate the score for the label predicted by the model for X_test et les vrais labels y_test.
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        y_test : array-like of shape (n_samples, 1).
            The test samples
        Returns
        -------
        score : float or array-like
            The computed score.
        """
        # Calcul du ROC-AUC ou de l'Accuracy
        self.y_pred = ((self.predict(X_test) > 0.25) * 1) * True
        self.y_true = y_test
        if self.scoring == "accuracy":
            score = np.count_nonzero(
                np.add(
                    self.y_pred.tolist(),
                    np.multiply(self.y_true.iloc[:, 0].tolist(), -1),
                )
                == 0
            ) / len(self.y_pred)
        if self.scoring == "roc":
            from sklearn.metrics import roc_auc_score

            score = roc_auc_score(self.y_true, self.predict(X_test))
        return score

    def summary(self):
        """
        Give information about the estimation of the logit model
        Returns
        -------
        Summary : Information about the coefficients estimated by the model
        """
        # Résumé des résultats de l'estimation
        return self.logit.summary()

    def cov_matrix(self):
        """
        Give the covariance matrix of the model
        Returns
        -------
        Cov : Covariance matrix of the model
        """
        # Matrice de variance-covariance
        return self.logit.cov_params()

    def margeff(self):
        """
        Give the marignal effects of the exogenous variables in the model
        Returns
        -------
        MargEff : Marginal effects of the model
        """
        # Effets marginaux
        return self.logit.get_margeff().summary_frame()
