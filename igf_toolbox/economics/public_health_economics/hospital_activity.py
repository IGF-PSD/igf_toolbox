import holidays
import re
import pandas as pd


class HospitalActivity:
    def __init__(self, data, ghm, ghs, price, prefix_stays):
        """ """

        self.data = data
        self.ghm = ghm
        self.ghs = ghs
        self.price = price
        self.prefix_stays = prefix_stays
        self.years = sorted(
            [
                int(re.search(r"\d+", col).group())
                for col in self.data.columns
                if col.startswith(self.prefix_stays)
            ]
        )

    def count_number_working_days(self, year: int) -> tuple[int]:
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

    def effet_cjo(self, year: int, activity: str = "volume") -> float:
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
            self.count_number_working_days(year)[0]
            + coeff * self.count_number_working_days(year)[1]
        )
        jours_activite_n_1 = (
            self.count_number_working_days(year - 1)[0]
            + coeff * self.count_number_working_days(year - 1)[1]
        )

        return jours_activite_n / jours_activite_n_1 - 1

    def effet_volume(self) -> pd.DataFrame:
        """ """

        data = self.data.groupby(by=[self.ghm, self.ghs]).sum()  # TO CHANGE

        dict_volume_economique = {
            year: (data["prix"] * data[f"{self.stays_prefix}_{year}"]).sum()
            for year in self.years
        }

        dict_effet_volume = {
            year: dict_volume_economique[year] / dict_volume_economique[year - 1] - 1
            for year in self.years[1:]
        }

        dict_effet_volume_cjo = {
            year: dict_effet_volume[year] - self.effet_cjo(year)
            for year in self.years[1:]
        }

        data_activity = pd.DataFrame(
            {
                "Volume économique": dict_volume_economique,
                "Effet volume": dict_effet_volume,
                "Effet volume CJO": dict_effet_volume_cjo,
            }
        )

        return data_activity
