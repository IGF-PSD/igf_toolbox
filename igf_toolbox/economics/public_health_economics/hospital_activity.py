import holidays
import pandas as pd

def count_number_wroking_days(year:int)->int:
    """

    """

    french_holidays=holidays.France(years=year)
    dates=pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq="D")

def effet_cjo(year:int, activity:str="volume")->float:
    """
    Returns the holiday adjustments to apply to yearly evolutions.

    Args:

    Returns:

    Raises:
    
    """

    if activity=="volume":
        coeff=.49
    elif activity=="séjours":
        coeff=.34
    else:
        raise ValueError("""
                        `activity` parameter should be in [`volume`, `séjours`] for holiday adjustments.
                        For `volume` a 49% correction is applied, for `séjours` a 34% correction is applied.
                        
                        """)

    jours_activite_n=
    jours_activite_n_1=

    return jours_activite_n/jours_activite_n_1-1
    