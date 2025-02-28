import gc
import re

import holidays
import pandas as pd


class HospitalActivity:
    def __init__(self, data, ghm, prefix_stays):
        """ """

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

    def _prix_apparent(
        self, data, ghs, nber_stays_eps, amount_eps, rate_eps
    ) -> pd.DataFrame:
        """
        Returns the average price of a GHMxGHS.

        Args:
        """

        data_price = data.groupby(by=[self.ghm, ghs], as_index=False)[
            [nber_stays_eps, amount_eps, rate_eps]
            + [col for col in data.columns if col.startswith(self.prefix_stays)]
        ].sum()

        data_price["prix_apparent"] = data_price[
            [nber_stays_eps, amount_eps, rate_eps]
        ].apply(
            lambda x: 0
            if x.iloc[0] == 0 or x.iloc[2] == 0
            else (1 / x.iloc[0]) * (x.iloc[1] / x.iloc[2]),
            axis=1,
        )

        return data_price

    def _prix_apparent_breakdown(
        self, data, by, nber_stays_eps, amount_eps, rate_eps
    ) -> pd.DataFrame:
        """
        Returns a dataframe with an average price for a given breakdown of effet volume.

        Args:

        Returns:

        """

        data_price = data.copy(deep=True)

        data_price = data_price.groupby(by=by, as_index=False).agg(
            sum_stays_eps=(nber_stays_eps, "sum"),
            sum_amount_eps=(
                amount_eps,
                lambda x: (
                    (x / data_price.loc[x.index, rate_eps]).where(
                        data_price.loc[x.index, rate_eps] != 0, 0
                    )
                ).sum(),
            ),
        )

        data_price["prix_apparent"] = data_price[
            ["sum_stays_eps", "sum_amount_eps"]
        ].apply(lambda x: 0 if x.iloc[0] == 0 else (1 / x.iloc[0]) * x.iloc[1], axis=1)

        return data_price

    def _preprocess_ghm(self, x: str) -> str:
        """ """
        x = x.split(" - ")[0]
        return x[0:6]

    def _preprocess_severite(self, x: str) -> str:
        """ """
        dict_severite = {"A": "1", "B": "2", "C": "3", "D": "4"}
        return dict_severite.get(x, x)

    def _preprocess_effet_volume(
        self, data, ghs, nber_stays_eps, amount_eps, rate_eps, type_hosp
    ) -> pd.DataFrame:
        """ """

        # Fill with 0 missing values in stays and amounts
        for col in [col for col in data.columns if col.startswith(self.prefix_stays)]:
            data[col] = data[col].fillna(0)

        for col in [nber_stays_eps, amount_eps, rate_eps]:
            data[col] = data[col].fillna(0)

        # Fill missing GHM, GHS, type_hosp
        for col in [self.ghm, ghs, type_hosp]:
            data[col] = data[col].ffill()

        # Make the GHM in the right format to compute racine
        data[self.ghm] = data[self.ghm].apply(self._preprocess_ghm)

        return data

    def effet_volume(
        self,
        ghs,
        nber_stays_eps,
        amount_eps,
        rate_eps,
        type_hosp,
        prefix_pop,
        age_class,
        hc_value="HC",
    ) -> pd.DataFrame:
        """
        Returns effet volume, its calendar adjustment and all its components.

        Args:

        Returns:

        Raises:
        """

        data = self.data.copy(deep=True)
        data = self._preprocess_effet_volume(
            data, ghs, nber_stays_eps, amount_eps, rate_eps, type_hosp
        )
        data = self._prix_apparent(data, ghs, nber_stays_eps, amount_eps, rate_eps)

        data = data.groupby(by=[self.ghm, ghs], as_index=False).agg(
            {
                "prix_apparent": "mean",
                **dict.fromkeys(
                    [col for col in data.columns if col.startswith(self.prefix_stays)],
                    "sum",
                ),
            }
        )

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
        data_price = self._prix_apparent_breakdown(
            data_racine, "racine", nber_stays_eps, amount_eps, rate_eps
        )[["racine", "prix_apparent"]]
        data_racine = data_racine.groupby("racine", as_index=False)[
            [col for col in data_racine.columns if col.startswith(self.prefix_stays)]
        ].sum()
        data_racine = data_racine.merge(
            data_price, how="inner", left_on="racine", right_on="racine"
        )

        dict_effet_racine = {
            year: (
                (
                    data_racine["prix_apparent"]
                    * data_racine[f"{self.prefix_stays}{year}"]
                    / data_racine[f"{self.prefix_stays}{year}"].sum()
                ).sum()
                / (
                    data_racine["prix_apparent"]
                    * data_racine[f"{self.prefix_stays}{year - 1}"]
                    / data_racine[f"{self.prefix_stays}{year - 1}"].sum()
                ).sum()
            )
            - 1
            for year in self.years[1:]
        }

        del data_racine
        _ = gc.collect()

        # We compute the effet bascule vers l'ambulatoire
        data_type_hosp = self.data.copy(deep=True)
        data_price = self._prix_apparent_breakdown(
            data_type_hosp, type_hosp, nber_stays_eps, amount_eps, rate_eps
        )[[type_hosp, "prix_apparent"]]
        data_type_hosp = data_type_hosp.groupby(type_hosp, as_index=False)[
            [col for col in data_type_hosp.columns if col.startswith(self.prefix_stays)]
        ].sum()
        data_type_hosp = data_type_hosp.merge(
            data_price, how="inner", left_on=type_hosp, right_on=type_hosp
        )

        dict_effet_bascule_ambulatoire = {
            year: (
                (
                    data_type_hosp["prix_apparent"]
                    * data_type_hosp[f"{self.prefix_stays}{year}"]
                    / data_type_hosp[f"{self.prefix_stays}{year}"].sum()
                ).sum()
                / (
                    data_type_hosp["prix_apparent"]
                    * data_type_hosp[f"{self.prefix_stays}{year - 1}"]
                    / data_type_hosp[f"{self.prefix_stays}{year - 1}"].sum()
                ).sum()
            )
            - 1
            for year in self.years[1:]
        }

        del data_type_hosp
        _ = gc.collect()

        # We compute the effet sévérité
        data_severite = self.data.copy(deep=True)

        # For correlation purposes with effet bascule vers l'ambulatoire
        # effet sévérité is computed on the scope HC stays only, to reduce effet résiduel
        data_severite = data_severite[data_severite[type_hosp] == hc_value]
        data_severite["severite"] = data_severite[self.ghm].apply(
            lambda x: self._preprocess_severite(x[-1])
        )
        data_price = self._prix_apparent_breakdown(
            data_severite, "severite", nber_stays_eps, amount_eps, rate_eps
        )[["severite", "prix_apparent"]]
        data_severite = data_severite.groupby("severite", as_index=False)[
            [col for col in data_severite.columns if col.startswith(self.prefix_stays)]
        ].sum()
        data_severite = data_severite.merge(
            data_price, how="inner", left_on="severite", right_on="severite"
        )

        dict_effet_severite = {
            year: (
                (
                    data_severite["prix_apparent"]
                    * data_severite[f"{self.prefix_stays}{year}"]
                    / (data_severite[f"{self.prefix_stays}{year}"].sum())
                ).sum()
                / (
                    data_severite["prix_apparent"]
                    * data_severite[f"{self.prefix_stays}{year - 1}"]
                    / (data_severite[f"{self.prefix_stays}{year - 1}"].sum())
                ).sum()
            )
            - 1
            for year in self.years[1:]
        }

        del data_severite
        _ = gc.collect()

        data_demo = self.data.copy(deep=True)

        # We add effet augmentation de la population
        years_pop = sorted(
            [
                int(re.search(r"\d+", col).group())
                for col in data_demo.columns
                if col.startswith(prefix_pop)
            ]
        )

        if not set(self.years).issubset(set(years_pop)):
            raise ValueError("""Stays and population time series are not given for the 
            same years""")

        dict_effet_augmentation_population = {
            year: (
                (
                    data_demo.drop_duplicates(subset=age_class)[
                        f"{prefix_pop}{year}"
                    ].sum()
                    / data_demo.drop_duplicates(subset=age_class)[
                        f"{prefix_pop}{year - 1}"
                    ].sum()
                )
                - 1
            )
            for year in self.years[1:]
        }

        # We add effet pyramide des âges

        dict_effet_pyramide_ages = {
            year: (
                (
                    data_demo.groupby(age_class)[
                        [
                            f"{self.prefix_stays}{year - 1}",
                            f"{prefix_pop}{year - 1}",
                            f"{prefix_pop}{year}",
                        ]
                    ]
                    .apply(
                        lambda x: (
                            x[f"{self.prefix_stays}{year - 1}"].sum()
                            / x[f"{prefix_pop}{year - 1}"].mean()
                            * x[f"{prefix_pop}{year}"].mean()
                        )
                    )
                    .sum()
                    / data_demo.drop_duplicates(subset=age_class)[
                        f"{prefix_pop}{year}"
                    ].sum()
                )
                / (
                    data_demo.groupby(age_class)[
                        [
                            f"{self.prefix_stays}{year - 1}",
                            f"{prefix_pop}{year - 1}",
                            f"{prefix_pop}{year}",
                        ]
                    ]
                    .apply(
                        lambda x: (
                            x[f"{self.prefix_stays}{year - 1}"].sum()
                            / x[f"{prefix_pop}{year - 1}"].mean()
                            * x[f"{prefix_pop}{year - 1}"].mean()
                        )
                    )
                    .sum()
                    / data_demo.drop_duplicates(subset=age_class)[
                        f"{prefix_pop}{year - 1}"
                    ].sum()
                )
            )
            - 1
            for year in self.years[1:]
        }

        # We concatenate all the effects
        data_activity = pd.DataFrame(
            {
                "Volume économique": dict_volume_economique,
                "Effet volume": dict_effet_volume,
                "Effet volume CJO": dict_effet_volume_cjo,
                "Effet nombre de séjours": dict_effet_nombre_de_sejours,
                "Effet racine": dict_effet_racine,
                "Effet bascule vers l'ambulatoire": dict_effet_bascule_ambulatoire,
                "Effet sévérité": dict_effet_severite,
                "Effet augmentation de la population": dict_effet_augmentation_population,
                "Effet pyramide des âges": dict_effet_pyramide_ages,
            }
        )

        # We compute the effet structure
        data_activity["Effet structure"] = (
            data_activity["Effet volume"] - data_activity["Effet nombre de séjours"]
        )

        # We compute the effet résiduel
        data_activity["Effet résiduel"] = data_activity["Effet structure"] - (
            data_activity["Effet racine"]
            + data_activity["Effet bascule vers l'ambulatoire"]
            + data_activity["Effet sévérité"]
        )

        # We compute the effet démographie and the effet modification du taux de recours
        data_activity["Effet démographie"] = (
            data_activity["Effet augmentation de la population"]
            + data_activity["Effet pyramide des âges"]
        )
        data_activity["Effet modification du recours à l'hospitalisation"] = (
            data_activity["Effet nombre de séjours"]
            - data_activity["Effet démographie"]
        )

        # Return the data in a proper order
        return data_activity[
            [
                "Volume économique",
                "Effet volume",
                "Effet volume CJO",
                "Effet nombre de séjours",
                "Effet modification du recours à l'hospitalisation",
                "Effet démographie",
                "Effet augmentation de la population",
                "Effet pyramide des âges",
                "Effet structure",
                "Effet racine",
                "Effet bascule vers l'ambulatoire",
                "Effet sévérité",
                "Effet résiduel",
            ]
        ]

    def evolution_equivalents_journees(
        self, prefix_dms, type_hosp, hc_value="HC", hp_value="Ambulatoire"
    ) -> pd.DataFrame:
        """ """

        data = self.data.copy(deep=True)
        data = data[
            [self.ghm, type_hosp]
            + [
                col
                for col in data.columns
                if col.startswith(self.prefix_stays) or col.startswith(prefix_dms)
            ]
        ]

        # We preprocess the data
        for col in [col for col in data.columns if col.startswith(prefix_dms)]:
            data[col] = data[col].fillna(1)

        for col in [col for col in data.columns if col.startswith(self.prefix_stays)]:
            data[col] = data[col].fillna(0)

        # Fill missing GHM, type_hosp
        for col in [self.ghm, type_hosp]:
            data[col] = data[col].ffill()

        dict_equivalents_journees_hp = {
            year: data[data[type_hosp] == hp_value][f"{self.prefix_stays}{year}"].sum()
            for year in self.years
        }

        dict_equivalents_journees_hc = {
            year: data[data[type_hosp] == hc_value]
            .groupby(self.ghm)[[f"{self.prefix_stays}{year}", f"{prefix_dms}{year}"]]
            .apply(
                lambda x: (
                    x[f"{self.prefix_stays}{year}"] * x[f"{prefix_dms}{year}"]
                ).sum()
            )
            .sum()
            for year in self.years
        }

        dict_sejours_hc = {
            year: data[data[type_hosp] == hc_value][f"{self.prefix_stays}{year}"].sum()
            for year in self.years
        }

        dict_dms_moyenne = {
            year: (
                (
                    data[data[type_hosp] == hc_value][f"{self.prefix_stays}{year}"]
                    * data[data[type_hosp] == hc_value][f"{prefix_dms}{year}"]
                ).sum()
                / data[data[type_hosp] == hc_value][f"{self.prefix_stays}{year}"].sum()
            )
            for year in self.years
        }

        data_activity = pd.DataFrame(
            {
                "Equivalents-journées HP": dict_equivalents_journees_hp,
                "Equivalents-journées HC": dict_equivalents_journees_hc,
                "Séjours HC": dict_sejours_hc,
                "DMS moyenne": dict_dms_moyenne,
            }
        )

        data_activity["Equivalents-journées"] = (
            data_activity["Equivalents-journées HP"]
            + data_activity["Equivalents-journées HC"]
        )

        # We compute the evolution of équivalents-journées
        data_activity["Evolution équivalents-journées"] = data_activity[
            "Equivalents-journées"
        ].pct_change()

        # We compute the calendar adjustments for this evolution
        data_activity["Evolution équivalents-journées CJO"] = data_activity.apply(
            lambda x: x["Evolution équivalents-journées"]
            - self._effet_cjo(x.name, "sejours"),
            axis=1,
        )

        # We compute the contributions of HC and HP to this evolution
        data_activity["Contribution équivalents-journées HP"] = (
            data_activity["Equivalents-journées HP"].diff()
            / data_activity["Equivalents-journées"].shift()
        )
        data_activity["Contribution équivalents-journées HC"] = (
            data_activity["Equivalents-journées HC"].diff()
            / data_activity["Equivalents-journées"].shift()
        )

        # We compute the contribution of HC stays, HP stays and DMS to this evolution
        data_activity["Contribution séjours HP"] = data_activity[
            "Contribution équivalents-journées HP"
        ]
        data_activity["Contribution séjours HC"] = (
            data_activity["Séjours HC"].diff()
            / data_activity["Equivalents-journées"].shift()
        )
        data_activity["Contribution DMS"] = (
            data_activity["DMS moyenne"].diff()
            / data_activity["Equivalents-journées"].shift()
        )
        data_activity["Effet résiduel"] = data_activity[
            "Evolution équivalents-journées"
        ] - data_activity[
            ["Contribution séjours HP", "Contribution séjours HC", "Contribution DMS"]
        ].sum(axis=1)

        return data_activity[
            [
                "Equivalents-journées",
                "Equivalents-journées HP",
                "Equivalents-journées HC",
                "Evolution équivalents-journées",
                "Evolution équivalents-journées CJO",
                "Contribution équivalents-journées HP",
                "Contribution équivalents-journées HC",
                "Contribution séjours HP",
                "Contribution séjours HC",
                "Contribution DMS",
                "Effet résiduel",
            ]
        ]
