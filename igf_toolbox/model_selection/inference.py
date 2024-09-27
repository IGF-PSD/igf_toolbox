## Importation des modules
# Modules de base
from typing import Optional, Union

import pandas as pd
# Modules sklearn
from sklearn.base import TransformerMixin


# Fonction d'estimation d'un modèle de régression
def estimate_summarize(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    transformer: Optional[Union[TransformerMixin, None]] = None,
    excluder: Optional[Union[TransformerMixin, None]] = None,
    sample_weight: Optional[Union[pd.Series, None]] = None,
):
    """
    Estimate a regression model and return its summary and R-squared value.

    This function allows for optional transformation and exclusion of data
    before fitting the model. It also allows for optional weighting of observations.

    Parameters:
    -----------
    estimator : object
        A regression estimator with fit, summary, and rsquared methods.

    X : DataFrame
        The feature matrix.
        Missing values in the DataFrame should be removed or imputed before passing to this function.

    y : Series
        The target variable.

    transformer : TransformerMixin, default=None
        An optional transformer object that has a fit_transform method.
        If provided, it will be used to transform X.

    excluder : TransformerMixin, default=None
        An optional column excluder object with a fit_transform method.
        If provided, it will be used to exclude certain columns from the data.

    sample_weight : Series, default=None
        Optional series of sample weights. If provided, these will be used to weight
        the observations during model fitting.

    Returns:
    --------
    tuple
        A tuple containing:
        - model summary (as returned by estimator.summary())
        - R-squared value (as returned by estimator.rsquared())

    Examples:
    ---------
    >>> est = OLS(...)
    >>> X = ...
    >>> y = ...
    >>> transformer = ...
    >>> summary, r2 = estimate_summarize(est, X, y, transformer=transformer)
    """

    # Transformation des données
    if transformer is not None:
        X = transformer.fit_transform(X.dropna())

    # Appariement des données et suppression des Nan
    data_work = pd.concat([X, y.to_frame()], axis=1, join="inner").dropna()

    # Exclusion de certaines données
    if excluder is not None:
        data_work = excluder.fit_transform(data_work)

    if sample_weight is not None:
        # Appariement avec les poids et suppression des Nan
        data_work = pd.concat(
            [data_work, sample_weight.loc[sample_weight > 0].to_frame()],
            axis=1,
            join="inner",
        ).dropna()
        # Initialisation et estimation du modèle
        model = estimator.fit(
            X=data_work[X.columns],
            y=data_work[y.name],
            sample_weight=data_work[sample_weight.name],
        )
    else:
        # Initialisation et estimation du modèle
        model = estimator.fit(X=data_work[X.columns], y=data_work[y.name])

    # Description des résultats
    return model.summary(), model.rsquared()
