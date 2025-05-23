import gc
import re

import holidays
import pandas as pd


class HospitalActivity:
    def __init__(self, file_path):
        """ """

        self.data_prix = pd.read_excel(file_path,
                                      sheetname="prix",
                                      header=4)

    def _count_number_working_days(self, year: int) -> tuple[int]:
        """
        Returns the number of working and non working days in a given year.

        Args:
            year (int): The year for which numbers of days should be returned.

        Returns:
            tuple[int]: A tuple with a first element being number of working days and
                        second element being number of non working days.
        """

        # Define a list of dates of French holidays
        french_holidays_dates = holidays.France(years=year)

        # Count the number of working days and non working days (i.e weekends and holidays)
        dates = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq="D")
        number_working_days = sum(
            1
            for date in dates
            if (date.weekday() < 5) and (date not in french_holidays_dates)
        )
        number_non_working_days = len(dates) - number_working_days

        return number_working_days, number_non_working_days

    def _effet_cjo(self, year: int, activity: str = "volume") -> float:
        """
        Returns the holiday adjustments to apply to yearly evolutions
        between year and year - 1.

        Args:
            year (int): The reference year for computing the effect in regards to the previous year.
            activity (str, optional): Type of weight to apply to the adjustment. 49% for `volume`
                                      and 34% for `sejours`. Values in [`volume`,`sejours`].
                                      Defaults to `volume`.

        Returns:
            float: CJO adjustment between year and year - 1.

        Raises:
            ValueError: Raises an error if `activity` parameter is not in [`volume`,`sejours`].
        """

        # Define the weight of the non working days
        if activity == "volume":
            coeff = 0.49
        elif activity == "sejours":
            coeff = 0.34
        else:
            raise ValueError("""
                            `activity` parameter should be in [`volume`, `séjours`] for holiday adjustments.
                            For `volume` a 49% correction is applied, for `séjours` a 34% correction is applied.
                            `séjours` is applied for both stays and `équivalents-journées`.
                            """)

        # Compute the number of `jours d'activite` and return their evolution
        # as CJO adjustment
        jours_activite_n = (
            self._count_number_working_days(year)[0]
            + coeff * self._count_number_working_days(year)[1]
        )
        jours_activite_n_1 = (
            self._count_number_working_days(year - 1)[0]
            + coeff * self._count_number_working_days(year - 1)[1]
        )

        return jours_activite_n / jours_activite_n_1 - 1

    def _preprocess_ghm(self, x: str) -> str:
        """ """
        x = x.split(" - ")[0]
        return x[0:6]

    def _preprocess_severite(self, x: str) -> str:
        """ """
        dict_severite = {"A": "1", "B": "2", "C": "3", "D": "4"}
        return dict_severite.get(x, x)

    def effet_volume(
        self,

    ) -> pd.DataFrame:
        """
        Returns effet volume, its calendar adjustment and all its components.

        Args:

        Returns:

        Raises:
        """

