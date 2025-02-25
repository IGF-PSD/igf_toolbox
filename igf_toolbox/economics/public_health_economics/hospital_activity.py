import holidays
import pandas as pd

def count_number_working_days(year:int)->int:
    """
    """

    # Define a list of dates of French holidays
    french_holidays_dates=holidays.France(years=year)

    # Count the number of working days and non working days (i.e weekends and holidays)
    dates=pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq="D")
    number_working_days=sum(1 for date in dates 
                            if (date.weekday() < 5) and (date not in french_holidays_dates))
    number_non_working_days=len(dates)-number_working_days

    return number_working_days, number_non_working_days

def effet_cjo(year:int, activity:str="volume")->float:
    """
    Returns the holiday adjustments to apply to yearly evolutions.

    Args:

    Returns:

    Raises:
    """

    # Define the weight of the non working days
    if activity=="volume":
        coeff=.49
    elif activity=="séjours":
        coeff=.34
    else:
        raise ValueError("""
                        `activity` parameter should be in [`volume`, `séjours`] for holiday adjustments.
                        For `volume` a 49% correction is applied, for `séjours` a 34% correction is applied.
                        `séjours` is applied for both stays and `équivalents-journées`.
                        """)

    # Compute the number of `jours d'activite` and return their evolution
    # as CJO adjustment
    jours_activite_n=count_number_working_days(year)[0]+coeff*count_number_working_days(year)[1]
    jours_activite_n_1=count_number_working_days(year-1)[0]+coeff*count_number_working_days(year-1)[1]

    return jours_activite_n/jours_activite_n_1-1
    