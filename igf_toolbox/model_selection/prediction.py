""" 
TO DO :
- donner un nom crédible aux colonnes du DataFrame retourné par la crossval_predict, y compris lorsque des intervalles de confiance sont simulés
- inclure l'estimation d'intervalles de prédiction dans un cas général
"""

from typing import Optional, Union

# Modules de base
import numpy as np
import pandas as pd
# Scikit-Learn
from sklearn.base import clone, is_classifier
from sklearn.model_selection._split import check_cv
from sklearn.model_selection._validation import _enforce_prediction_order
from sklearn.utils import indexable, resample
from sklearn.utils._joblib import Parallel, delayed
from sklearn.utils.metaestimators import _safe_split
# from sklearn.utils.validation import _check_method_params


# La crossVal n'est pas faite pour le Online Learning car on clone à chaque fois l'estimateur initial
# Adaptation de la fonction sklearn pour retourner un pandas.DataFrame
def crossval_predict(
    estimator,
    X,
    y=None,
    groups=None,
    cv="warn",
    n_jobs: Optional[int] = 2,
    verbose: Optional[int] = 0,
    compute_confidence_interval: Optional[bool] = False,
    bootstrap_size: Optional[Union[int, None]] = None,
    n_iterations_boostrap: Optional[Union[int, None]] = None,
    alpha: Optional[Union[float, None]] = None,
    fit_params: Optional[Union[dict, None]] = None,
    pre_dispatch: Optional[Union[str, int]] = "2*n_jobs",
    method: Optional[str] = "predict",
):
    """
    Perform cross-validation on an estimator and obtain prediction results.
    This version is adapted for online learning scenarios and returns a pandas DataFrame.

    Parameters
    ----------
    estimator : estimator object
        The object to use to fit the data.
    X : array-like or pd.DataFrame
        The data to fit.
    y : array-like, optional, default: None
        The target variable to try to predict.
    groups : array-like, optional, default: None
        Group labels for the samples.
    cv : int, cross-validation generator or 'warn', optional (default='warn')
        Determines the cross-validation splitting strategy.
    n_jobs : int, optional (default=2)
        Number of jobs to run in parallel.
    verbose : int, optional (default=0)
        Verbosity level.
    compute_confidence_interval : bool, optional (default=False)
        Whether to compute confidence interval for the prediction.
    bootstrap_size : int, optional (default=None)
        Size of bootstrap sample.
    n_iterations_boostrap : int, optional (default=None)
        Number of bootstrap iterations.
    alpha : float, optional (default=None)
        Alpha value for intervals.
    fit_params : dict, optional (default=None)
        Parameters to pass to the fit method.
    pre_dispatch : int, or string, optional (default='2*n_jobs')
        Controls the number of jobs that get dispatched during parallel execution.
    method : string, optional (default='predict')
        Invokes the passed method name of the passed estimator.

    Returns
    -------
    predictions : pd.DataFrame
        The predictions obtained from cross-validation.

    Notes
    -----
    This function is adapted for online learning scenarios and is designed to prevent cloning the initial estimator during cross-validation.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.linear_model import LogisticRegression
    >>> X, y = load_iris(return_X_y=True)
    >>> crossval_predict(LogisticRegression(), X, y, cv=3)

    """
    # Le score rmse est ici équivalent à la valeur absolue de l'erreur
    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)
    prediction_blocks = parallel(
        delayed(_fit_and_predict)(
            clone(estimator),
            X,
            y,
            train,
            test,
            verbose,
            compute_confidence_interval,
            bootstrap_size,
            n_iterations_boostrap,
            alpha,
            fit_params,
            method,
        )
        for train, test in cv.split(X, y, groups)
    )

    # Concatenate the predictions
    predictions = [pred_block_i for pred_block_i, _ in prediction_blocks]
    predictions = np.hstack(predictions)

    # Wrap results in pd.DataFrame
    test_indices = np.concatenate([indices_i for _, indices_i in prediction_blocks])
    test_index = [y.index[_] for _ in test_indices]

    return pd.DataFrame(np.transpose(predictions), index=test_index)
    # if predictions.ndim == 1 :
    #     return pd.DataFrame(np.transpose(predictions), index = test_index)
    #     #return pd.Series(predictions, index = test_index)
    # else :
    #     return pd.DataFrame(np.transpose(predictions), index = test_index)


# Fonction auxiliaire d'entrainement et de prédiction du modèle
def _fit_and_predict(
    estimator,
    X,
    y,
    train,
    test,
    verbose: int,
    compute_confidence_interval: bool,
    bootstrap_size: Union[int, None],
    n_iterations_boostrap: Union[int, None],
    alpha: Union[float, None],
    fit_params: Union[dict, None],
    method: str,
):
    """
    Internal auxiliary function to fit the model and predict on a given train-test split.

    Parameters
    ----------
    estimator : object
        An estimator instance.
    X : array-like or pd.DataFrame
        The data to fit.
    y : array-like or None
        The target variable to try to predict.
    train : array-like
        The indices of the training samples.
    test : array-like
        The indices of the testing samples.
    verbose : int
        The verbosity level.
    compute_confidence_interval : bool
        Whether to compute confidence intervals for the predictions.
    bootstrap_size : int or None
        The number of bootstrap samples to use.
    n_iterations_boostrap : int or None
        The number of bootstrap iterations.
    alpha : float or None
        Alpha level for the confidence interval.
    fit_params : dict or None
        Parameters to pass to the fit method of the estimator.
    method : str
        The prediction method to use ('predict', 'predict_proba', etc.).

    Returns
    -------
    predictions : ndarray
        Predicted values for each test sample. If `compute_confidence_interval` is True,
        it also returns the lower and upper bounds for each prediction.

    Notes
    -----
    This is an internal function, users should use `crossval_predict` instead.
    """
    # Ajuste la longeur des vecteurs de poids sur les observations
    fit_params = fit_params if fit_params is not None else {}
    # fit_params = _check_method_params(X, fit_params, train)

    # Extraction des observations correspondant au Train
    X_train, y_train = _safe_split(estimator, X, y, train)

    # Extraction des observations correspondant au Test
    X_test, _ = _safe_split(estimator, X, y, test, train)

    # Entrainement de l'Estimateur
    if y_train is None:
        estimator.fit(X_train, **fit_params)
    else:
        estimator.fit(X_train, y_train, **fit_params)
    func = getattr(estimator, method)
    predictions = func(X_test)

    encode = (
        method in ["decision_function", "predict_proba", "predict_log_proba"]
        and y is not None
    )

    if encode:
        if isinstance(predictions, list):
            predictions = [
                _enforce_prediction_order(
                    estimator.classes_[i_label],
                    predictions[i_label],
                    n_classes=len(set(y[:, i_label])),
                    method=method,
                )
                for i_label in range(len(predictions))
            ]
        else:
            # A 2D y array should be a binary label indicator matrix
            n_classes = len(set(y)) if y.ndim == 1 else y.shape[1]
            predictions = _enforce_prediction_order(
                estimator.classes_, predictions, n_classes, method
            )

    # Computation de l'intervalle de confiance
    # Normalement pour prédiction on pourrait prendre la médiane plutot que la valeur prédite sur l'ensemble
    # L'algorithme ne marche sans doute pas dans le cas d'un y de dimension supérieure à 1, faire des np.array et chercher les std en séprarant composante par composante
    # Pour le Bootstrap cela vaudrait peut être vraiment le coup d'implémenter avec numba ou en C++
    if (
        bootstrap_size is not None
        and n_iterations_boostrap is not None
        and alpha is not None
    ):
        # ScoreBootstrap = []
        list_predictions_boostrap = []
        for _ in range(n_iterations_boostrap):
            if y_train is None:
                X_trainBootstrap = resample(
                    X_train, replace=True, n_samples=len(X_train) * bootstrap_size
                )
                estimator.fit(X_trainBootstrap, **fit_params)
            else:
                X_trainBootstrap, y_trainBootstrap = resample(
                    X_train,
                    y_train,
                    replace=True,
                    n_samples=int(len(X_train) * bootstrap_size),
                )
                estimator.fit(X_trainBootstrap, y_trainBootstrap, **fit_params)

            func = getattr(estimator, method)
            predictionsBootstrap = func(X_test)

            encode = (
                method in ["decision_function", "predict_proba", "predict_log_proba"]
                and y is not None
            )

            if encode:
                if isinstance(predictionsBootstrap, list):
                    predictionsBootstrap = [
                        _enforce_prediction_order(
                            estimator.classes_[i_label],
                            predictionsBootstrap[i_label],
                            n_classes=len(set(y[:, i_label])),
                            method=method,
                        )
                        for i_label in range(len(predictionsBootstrap))
                    ]
                else:
                    # A 2D y array should be a binary label indicator matrix
                    n_classes = len(set(y)) if y.ndim == 1 else y.shape[1]
                    predictionsBootstrap = _enforce_prediction_order(
                        estimator.classes_, predictionsBootstrap, n_classes, method
                    )

            # Ajout de la prédiction
            list_predictions_boostrap.append(predictionsBootstrap)

        if compute_confidence_interval:
            # Computation des limites de l'intervalle de confiance
            lower_bound_conf = np.quantile(
                a=np.vstack(list_predictions_boostrap), q=(1 - alpha) / 2, axis=0
            )
            upper_bound_conf = np.quantile(
                a=np.vstack(list_predictions_boostrap), q=(1 + alpha) / 2, axis=0
            )

            predictions = np.vstack(
                [predictions, lower_bound_conf, upper_bound_conf]
            )  # Peut être faut-il le changer en hstack, essayer dans le crossval prédict

    return np.transpose(predictions)
