import re

import holidays
import pandas as pd


class HospitalActivity:
    def __init__(self, data, ghm, prefix_stays):
        """ 
        
        """

        self.data = data
        self.ghm = ghm
        self.prefix_stays = prefix_stays

        self.years = sorted(
            [
                int(re.search(r"\d+", col).group())
                for col in self.data.columns
                if col.startswith(self.prefix_stays)
            ]
        )

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

    def _prix_apparent(self, ghs, nber_stays_eps, amount_eps, rate_eps)->pd.DataFrame:
        """ 
        Returns the average price of a GHMxGHS.

        Args:
        """

        data_price=self.data.groupby(by=[self.ghm, ghs],
                                    as_index=False)[[nber_stays_eps,
                                                    amount_eps,
                                                    rate_eps]+
                                                    [col for col in self.data.columns
                                                    if col.startswith(self.prefix_stays)]].sum()
        
        data_price["prix_apparent"]=data_price[[nber_stays_eps, 
                                              amount_eps, 
                                              rate_eps]].apply(lambda x: 0 if x[0]==0 or x[2]==0 else (1/x[0])*(x[1]/x[2]), 
                                                               axis=1)

        return data_price

    def _prix_apparent_breakdown(by, nber_stays_eps, amount_eps, rate_eps)->pd.DataFrame:
        """
        Returns a dataframe with an average price for a given breakdown of effet volume.

        Args:

        Returns:
        
        """

        data_price=self.data.copy(deep=True)
        
        data_price=data_price.groupby(by=by,
                                     as_index=False).agg(
            sum_stays_eps=(nber_stays_eps, "sum"),
            sum_amount_eps=(amount_eps,lambda x: ((x/data_price.loc[x.index, rate_eps]).where(data_price.loc[x.index,rate_eps]!=0, 0)).sum())
                                     )
        
        data_price["prix_apparent"]=data_price[[sum_stays_eps,
                                               sum_amount_eps]].apply(lambda x: 0 if x[0]==0 else (1/x[0])*x[1])
        return data_price

    def _effet_structure_components():
        pass

    def effet_volume(self, ghs, nber_stays_eps, amount_eps, rate_eps) -> pd.DataFrame:
        """ 
        Returns effet volume, its calendar adjustment and all its components.

        Args:

        Returns:

        Raises:
        """

        data = self._prix_apparent(ghs, nber_stays_eps, amount_eps, rate_eps)

        data = data.groupby(
            by=[self.ghm, ghs], as_index=False
        ).agg({"prix_apparent":"mean", 
               **dict.fromkeys([col for col in data.columns
                                if col.startswith(self.prefix_stays)], "sum")})

        dict_volume_economique = {
            year: (data["prix_apparent"] * data[f"{self.prefix_stays}{year}"]).sum()
            for year in self.years
        }

        dict_effet_volume = {
            year: dict_volume_economique[year] / dict_volume_economique[year - 1] - 1
            for year in self.years[1:]
        }

        dict_effet_volume_cjo = {
            year: dict_effet_volume[year] - self._effet_cjo(year)
            for year in self.years[1:]
        }

        dict_effet_nombre_de_sejours = {
            year: data[f"{self.prefix_stays}{year}"].sum()
            / data[f"{self.prefix_stays}{year - 1}"].sum()
            - 1
            for year in self.years[1:]
        }



        # We compute the effet racine
        data_racine = self.data.copy(deep=True)
        data_racine["racine"] = data_racine[self.ghm].str[:5]
        data_racine = self._prix_apparent_breakdown("racine",
                                                   nber_stays_eps,
                                                   amount_eps,
                                                   rate_eps)

        # We compute the effet bascule vers l'ambulatoire

        # We compute the effet sévérité

        data_activity = pd.DataFrame(
            {
                "Volume économique": dict_volume_economique,
                "Effet volume": dict_effet_volume,
                "Effet volume CJO": dict_effet_volume_cjo,
                "Effet nombre de séjours": dict_effet_nombre_de_sejours,
                "Effet racine":dict_effet_racine,
                "Effet bascule vers l'ambulatoire":dict_effet_bascule_ambulatoire,
                "Effet sévérité":dict_effet_severite
            }
        )

        # We compute the effet structure
        data_activity["Effet structure"] = (
            data_activity["Effet volume"] - data_activity["Effet nombre de séjours"]
        )

        # We compute the effet résiduel
        data_activity["Effet résiduel"] = (
            data_activity["Effet structure"] - (data_activity["Effet racine"] + 
                                                data_activity["Effet bascule vers l'ambulatoire"] + 
                                                data_activity["Effet sévérité"])
            
        )

        return data_activity

    # def evolution_equivalents_journees(self, prefix_dms):
    #     """ """

    #     pass
