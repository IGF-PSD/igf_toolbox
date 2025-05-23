import gc
import re

import holidays
import pandas as pd


class HospitalActivity:
    def __init__(self, file_path):
        """ 
        
        """

        self.data_valorisations = pd.read_excel(
            file_path,
            sheet_name="Valorisations",
            header=4

        )
        self.data_casemix = pd.read_excel(
            file_path,
            sheet_name="Volume économique",
            header=1
        )

        self.ghm = "PMSI MCO - GHM"
        self.ghs = "PMSI MCO - GHS"
        self.sejours_prix = "ACTIVITE - Nb Séjours Hors Séances"
        self.montants_am = "Montant AM GHS"
        self.taux_am = "Taux de remboursement AM"
        self.type_hosp = "PMSI MCO - Activité - Type Hospitalisation"
        self.tranche_age = "PMSI - Tranche Age INSEE"

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
        """ 
        """
        x = x.split(" - ")[0]
        return x[0:6]

    def _preprocess_severite(self, x: str) -> str:
        """ 
        """
        dict_severite = {"A": "1", "B": "2", "C": "3", "D": "4"}
        return dict_severite.get(x, x)

    def _compute_prix_apparents(
        self
    ) -> pd.DataFrame:
        """

        """

        data_prix_apparents = self.data_valorisations.copy()

        # Fill NaNs in GHM and preprocess to only keep 6-character code
        data_prix_apparents[self.ghm] = (
            data_prix_apparents[self.ghm].ffill().apply(lambda x: self._preprocess_ghm(x))
        )

        # Impute data under statistical secret
        data_prix_apparents[self.sejours_prix] = (
            data_prix_apparents[self.sejours_prix].replace("1 à 5", 2.5)
        )

        data_prix_apparents[[self.sejours_prix,
                             self.taux_am,
                             self.montants_am]] = (
            data_prix_apparents[[self.sejours_prix,
                                     self.taux_am,
                                     self.montants_am]].astype(float)
                                 )

        # Only keep data with reimbursement rates superior to 0 and stays superior to 0
        data_prix_apparents = (
            data_prix_apparents[
                (data_prix_apparents[self.taux_am] > 0)
                & (data_prix_apparents[self.sejours_prix] > 0)
                ]
        )
        
        data_prix_apparents[self.montants_am] = (
            data_prix_apparents[self.montants_am].fillna(0)
        )

        # Compute the average price for a GHMxGHS
        # given by: (1/sejour) * (Montant AM / Taux AM)
        data_prix_apparents["prix_apparent"] = (
            (1/data_prix_apparents[self.sejours_prix])
            * (data_prix_apparents[self.montants_am]/data_prix_apparents[self.taux_am])
        )

        return data_prix_apparents[[self.ghm, self.ghs, "prix_apparent"]]

    def effet_volume(self) -> pd.DataFrame:
        """

        """

        # We compute first the effet volume and effet volume CJO
        data_volume_eco = self.data_casemix.copy()

        data_volume_eco[self.ghm] = (
            data_volume_eco[self.ghm].ffill().apply(lambda x: self._preprocess_ghm(x))
        )

        data_volume_eco[self.ghs] = (
            data_volume_eco[self.ghs].ffill()
        )

        data_volume_eco = data_volume_eco.drop(columns = [
            self.type_hosp,
            self.tranche_age
        ])

        data_volume_eco = data_volume_eco.groupby(
            by = [self.ghm, self.ghs],
            as_index = False
        ).sum()

        data_volume_eco = (
            data_volume_eco.merge(self._compute_prix_apparents(),
                                 on = [self.ghm, self.ghs],
                                 how = "right")
        )

        list_years = [
            col for col in data_volume_eco.columns
            if col.isdigit()
        ]
        list_sorted_years = sorted(
            map(
                int, list_years
            )
        )

        dict_nb_sejours = {}
        
        for year in list_sorted_years:
            data_volume_eco[year] = (
                data_volume_eco[str(year)]*data_volume_eco["prix_apparent"]
            )
            dict_nb_sejours[year] = data_volume_eco[str(year)].sum()

        data_volume_eco = (
            data_volume_eco[
                list_sorted_years
            ].sum().to_frame("volume_economique")
        )

        data_volume_eco["effet_volume"] = (
            data_volume_eco["volume_economique"].pct_change()
        )

        data_volume_eco["effet_cjo"] = (
            data_volume_eco.index.map(lambda x: self._effet_cjo(x))
        )

        data_volume_eco["effet_volume_cjo"] = (
            data_volume_eco["effet_volume"] - data_volume_eco["effet_cjo"]
        )

        # We can add now the effet nombre de séjours and effet structure
        data_volume_eco["nombre_sejours"] = pd.DataFrame.from_dict(dict_nb_sejours, 
                                                                   orient="index", 
                                                                   columns=["nombre_sejours"])
        data_volume_eco["effet_nombre_sejours"] = (
            data_volume_eco["nombre_sejours"].pct_change()
        )

        data_volume_eco["effet_structure"] = (
            data_volume_eco["effet_volume"] - data_volume_eco["effet_nombre_sejours"]
        )

        return data_volume_eco

