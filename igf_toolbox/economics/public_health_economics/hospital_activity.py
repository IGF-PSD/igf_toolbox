import holidays
import pandas as pd



def effet_cjo(year:int, activity:str="volume")->float:
    """
    Returns the holiday adjustments to apply to yearly evolutions.

    Args:

    Returns:

    Raises:
    
    """

    french_holidays=holidays.France(years=year)
    