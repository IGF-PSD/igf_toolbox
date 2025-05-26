import holidays
import pandas as pd


class HospitalActivityDiamant:
    """ """

    def __init__(self, file_path):
        """ """

        self.data_valorisations = pd.read_excel(
            file_path, sheet_name="Valorisations", header=4
        )
        self.data_valorisations_racine = pd.read_excel(
            file_path, sheet_name="Valorisations - racine", header=4
        )
        self.data_valorisations_severite = pd.read_excel(
            file_path, sheet_name="Valorisations - sévérité", header=5
        )
        self.data_valorisations_type_hosp = pd.read_excel(
            file_path, sheet_name="Valorisations - type hosp", header=4
        )
        self.data_casemix = pd.read_excel(
            file_path, sheet_name="Volume économique", header=1
        )

        self.ghm = "PMSI MCO - GHM"
        self.ghs = "PMSI MCO - GHS"
        self.sejours_prix = "ACTIVITE - Nb Séjours Hors Séances"
        self.montants_am = "Montant AM GHS"
        self.taux_am = "Taux de remboursement AM"
        self.type_hosp = "PMSI MCO - Activité - Type Hospitalisation"
        self.tranche_age = "PMSI - Tranche Age INSEE"
        self.hosp_hp = "Ambulatoire"
        self.hosp_hc = "HC"
        self.racine = "PMSI MCO - GHM - Code Racine"
        self.severite = "PMSI MCO - GHM - Niveau de sévérité"
        self.list_severite = ["1", "2", "3", "4", "A", "B", "C", "D"]

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

    @staticmethod
    def _preprocess_ghm(x: str) -> str:
        """ """
        x = x.split(" - ")[0]
        return x[0:6]

    @staticmethod
    def _preprocess_racine(x: str) -> str:
        """ """
        x = x.split(" - ")[0]
        return x[0:5]

    @staticmethod
    def _preprocess_severite(x: str) -> str:
        """ """
        x = x.split(" - ")[0]
        return x[-1]

    def _compute_prix_apparents(self) -> pd.DataFrame:
        """ """

        data_prix_apparents = self.data_valorisations.copy()

        # Fill NaNs in GHM and preprocess to only keep 6-character code
        data_prix_apparents[self.ghm] = (
            data_prix_apparents[self.ghm]
            .ffill()
            .apply(lambda x: self._preprocess_ghm(x))
        )

        # Impute data under statistical secret
        data_prix_apparents[self.sejours_prix] = data_prix_apparents[
            self.sejours_prix
        ].replace("1 à 5", 2.5)

        data_prix_apparents[[self.sejours_prix, self.taux_am, self.montants_am]] = (
            data_prix_apparents[
                [self.sejours_prix, self.taux_am, self.montants_am]
            ].astype(float)
        )

        # Only keep data with reimbursement rates superior to 0 and stays superior to 0
        data_prix_apparents = data_prix_apparents[
            (data_prix_apparents[self.taux_am] > 0)
            & (data_prix_apparents[self.sejours_prix] > 0)
        ]

        data_prix_apparents[self.montants_am] = data_prix_apparents[
            self.montants_am
        ].fillna(0)

        # Compute the average price for a GHMxGHS
        # given by: (1/sejour) * (Montant AM / Taux AM)
        data_prix_apparents["prix_apparent"] = (
            1 / data_prix_apparents[self.sejours_prix]
        ) * (data_prix_apparents[self.montants_am] / data_prix_apparents[self.taux_am])

        return data_prix_apparents[[self.ghm, self.ghs, "prix_apparent"]]

    def _compute_prix_apparents_breakdown(
        self, breakdown: str = "racine"
    ) -> pd.DataFrame:
        """ """

        if breakdown == "racine":
            data_prix_apparents = self.data_valorisations_racine.copy()
            data_prix_apparents = data_prix_apparents[
                data_prix_apparents[self.taux_am] > 0
            ]
            data_prix_apparents["prix_apparent"] = (
                1 / data_prix_apparents[self.sejours_prix]
            ) * (
                data_prix_apparents[self.montants_am]
                / data_prix_apparents[self.taux_am]
            )
            return data_prix_apparents[[self.racine, "prix_apparent"]]

        elif breakdown == "type_hosp":
            data_prix_apparents = self.data_valorisations_type_hosp.copy()
            data_prix_apparents = data_prix_apparents[
                data_prix_apparents[self.type_hosp].isin([self.hosp_hp, self.hosp_hc])
            ]
            data_prix_apparents = data_prix_apparents[
                data_prix_apparents[self.taux_am] > 0
            ]
            data_prix_apparents["prix_apparent"] = (
                1 / data_prix_apparents[self.sejours_prix]
            ) * (
                data_prix_apparents[self.montants_am]
                / data_prix_apparents[self.taux_am]
            )
            return data_prix_apparents[[self.type_hosp, "prix_apparent"]]

        elif breakdown == "severite":
            data_prix_apparents = self.data_valorisations_severite.copy()
            data_prix_apparents = data_prix_apparents[
                data_prix_apparents[self.severite].isin(self.list_severite)
            ]
            data_prix_apparents = data_prix_apparents[
                data_prix_apparents[self.taux_am] > 0
            ]
            data_prix_apparents["prix_apparent"] = (
                1 / data_prix_apparents[self.sejours_prix]
            ) * (
                data_prix_apparents[self.montants_am]
                / data_prix_apparents[self.taux_am]
            )
            return data_prix_apparents[[self.severite, "prix_apparent"]]

    def effet_volume(self) -> pd.DataFrame:
        """ """

        # We compute first the effet volume and effet volume CJO
        data_volume_eco = self.data_casemix.copy()

        data_volume_eco[self.ghm] = (
            data_volume_eco[self.ghm].ffill().apply(lambda x: self._preprocess_ghm(x))
        )

        data_volume_eco[self.ghs] = data_volume_eco[self.ghs].ffill()

        data_volume_eco = data_volume_eco.drop(
            columns=[self.type_hosp, self.tranche_age]
        )

        data_volume_eco = data_volume_eco.groupby(
            by=[self.ghm, self.ghs], as_index=False
        ).sum()

        data_volume_eco = data_volume_eco.merge(
            self._compute_prix_apparents(), on=[self.ghm, self.ghs], how="right"
        )

        list_years = [col for col in data_volume_eco.columns if col.isdigit()]
        list_sorted_years = sorted(map(int, list_years))

        dict_nb_sejours = {}

        for year in list_sorted_years:
            data_volume_eco[year] = (
                data_volume_eco[str(year)] * data_volume_eco["prix_apparent"]
            )
            dict_nb_sejours[year] = data_volume_eco[str(year)].sum()

        data_volume_eco = (
            data_volume_eco[list_sorted_years].sum().to_frame("volume_economique")
        )

        data_volume_eco["effet_volume"] = data_volume_eco[
            "volume_economique"
        ].pct_change()

        data_volume_eco["effet_cjo_volume_economique"] = data_volume_eco.index.map(
            lambda x: self._effet_cjo(x)
        )

        data_volume_eco["effet_volume_cjo"] = (
            data_volume_eco["effet_volume"]
            - data_volume_eco["effet_cjo_volume_economique"]
        )

        # We can add now the effet nombre de séjours and effet structure
        data_volume_eco["nombre_sejours"] = pd.DataFrame.from_dict(
            dict_nb_sejours, orient="index", columns=["nombre_sejours"]
        )
        data_volume_eco["effet_nombre_sejours"] = data_volume_eco[
            "nombre_sejours"
        ].pct_change()

        data_volume_eco["effet_structure"] = (
            data_volume_eco["effet_volume"] - data_volume_eco["effet_nombre_sejours"]
        )

        # We also add effet nombre de séjours CJO
        data_volume_eco["effet_cjo_nombre_sejours"] = data_volume_eco.index.map(
            lambda x: self._effet_cjo(x, "sejours")
        )

        data_volume_eco["effet_nombre_sejours_cjo"] = (
            data_volume_eco["effet_nombre_sejours"]
            - data_volume_eco["effet_cjo_nombre_sejours"]
        )

        # We breakdown effet structure: effet type de prise en charge
        data_effet_prise_charge = self.data_casemix[
            [self.type_hosp] + list_years
        ].copy()
        data_effet_prise_charge[self.type_hosp] = data_effet_prise_charge[
            self.type_hosp
        ].ffill()
        data_effet_prise_charge = data_effet_prise_charge[
            data_effet_prise_charge[self.type_hosp].isin([self.hosp_hp, self.hosp_hc])
        ]
        data_effet_prise_charge = data_effet_prise_charge.groupby(
            self.type_hosp, as_index=False
        ).sum()
        data_effet_prise_charge = data_effet_prise_charge.merge(
            self._compute_prix_apparents_breakdown("type_hosp"),
            on=self.type_hosp,
            how="right",
        )

        for year in data_volume_eco.index[1:]:
            numerator = (
                data_effet_prise_charge["prix_apparent"]
                * data_effet_prise_charge[str(year)]
                / data_effet_prise_charge[str(year)].sum()
            ).sum()
            denominator = (
                data_effet_prise_charge["prix_apparent"]
                * data_effet_prise_charge[str(year - 1)]
                / data_effet_prise_charge[str(year - 1)].sum()
            ).sum()
            data_volume_eco.loc[year, "effet_type_prise_en_charge"] = (
                numerator / denominator - 1
            )

        # We breakdown effet structure : effet racine
        data_effet_racine = self.data_casemix[[self.ghm] + list_years].copy()
        data_effet_racine[self.ghm] = data_effet_racine[self.ghm].ffill()
        data_effet_racine[self.racine] = data_effet_racine[self.ghm].apply(
            lambda x: self._preprocess_racine(x)
        )
        data_effet_racine = data_effet_racine[[self.racine] + list_years]
        data_effet_racine = data_effet_racine.groupby(self.racine, as_index=False).sum()
        data_effet_racine = data_effet_racine.merge(
            self._compute_prix_apparents_breakdown("racine"),
            on=self.racine,
            how="right",
        )

        for year in data_volume_eco.index[1:]:
            numerator = (
                data_effet_racine["prix_apparent"]
                * data_effet_racine[str(year)]
                / data_effet_racine[str(year)].sum()
            ).sum()
            denominator = (
                data_effet_racine["prix_apparent"]
                * data_effet_racine[str(year - 1)]
                / data_effet_racine[str(year - 1)].sum()
            ).sum()
            data_volume_eco.loc[year, "effet_racine"] = numerator / denominator - 1

        # We breakdown effet structure : effet sévérité
        data_effet_severite = self.data_casemix[
            [self.ghm, self.type_hosp] + list_years
        ].copy()
        data_effet_severite[self.ghm] = data_effet_severite[self.ghm].ffill()
        data_effet_severite[self.type_hosp] = data_effet_severite[
            self.type_hosp
        ].ffill()
        data_effet_severite = data_effet_severite[
            data_effet_severite[self.type_hosp] == self.hosp_hc
        ]
        data_effet_severite[self.severite] = data_effet_severite[self.ghm].apply(
            lambda x: self._preprocess_severite(x)
        )
        data_effet_severite = data_effet_severite[[self.severite] + list_years]
        data_effet_severite = data_effet_severite[
            data_effet_severite[self.severite].isin(self.list_severite)
        ]
        data_effet_severite = data_effet_severite.groupby(
            self.severite, as_index=False
        ).sum()
        data_effet_severite = data_effet_severite.merge(
            self._compute_prix_apparents_breakdown("severite"),
            on=self.severite,
            how="right",
        )

        for year in data_volume_eco.index[1:]:
            numerator = (
                data_effet_severite["prix_apparent"]
                * data_effet_severite[str(year)]
                / data_effet_severite[str(year)].sum()
            ).sum()
            denominator = (
                data_effet_severite["prix_apparent"]
                * data_effet_severite[str(year - 1)]
                / data_effet_severite[str(year - 1)].sum()
            ).sum()
            data_volume_eco.loc[year, "effet_severite"] = numerator / denominator - 1

        # We breakdown effet structure : effet résiduel
        data_volume_eco["effet_residuel"] = data_volume_eco["effet_structure"] - (
            data_volume_eco["effet_type_prise_en_charge"]
            + data_volume_eco["effet_racine"]
            + data_volume_eco["effet_severite"]
        )

        # We breakdown effet nombre de séjours: effet augmentation de la population 

        # We breakdown effet nombre de séjours: effet pyramide des âges

        # We breakdown effet nombre de séjours: effet démographie
        data_volume_eco["effet_demographie"] = (
            data_volume_eco["effet_augmentation_population"]
            + data_volume_eco["effet_pyramide_ages"]
        )
        
        # We breakdown effet nombre de séjours: effet modification du recours à l'hospitalisation
        data_volume_eco["effet_modification_recours"] = (
            data_volume_eco["effet_nombre_sejours"] - data_volume_eco["effet_demographie"]
        )

        return data_volume_eco[
            [
                "volume_economique",
                "nombre_sejours",
                "effet_volume",
                "effet_volume_cjo",
                "effet_nombre_sejours",
                "effet_nombre_sejours_cjo",
                "effet_structure",
                "effet_type_prise_en_charge",
                "effet_racine",
                "effet_severite",
                "effet_residuel",
                "effet_demographie",
                "effet_modification_recours",
                "effet_augmentation_population",
                "effet_pyramide_ages"
            ]
        ]
