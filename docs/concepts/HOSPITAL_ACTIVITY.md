# Public Health Economics indicators for hospital activity measurement

## PMSI MCO

L'unité statistique à retenir dans le PMSI dans le cadre d'une analyse économique de l'activité hospitalière MCO est celle du croisement d'un GHM (Groupe homogène de maldes) et d'un GHS (Groupe homogène de séjours). 

Un GHM regroupe les prises en charge de même nature médicale et économique et constitue la catégorie élémentaire de la classification en MCO. Chaque séjour aboutit dans un GHM selon un algorithme fondé sur les informations médico-administratives contenues dans le résumé de sortie standadrdisé (RSS) de chaque patient. Un GHS correspond, dans le cadre de la T2A, au tarif du groupe homogène de malades. La très vaste majorité des GHM ne correspondent qu'à un unique GHS, c'est-à-dire à un seul tarif. Toutefois, certains GHM peuvent être affectés à deux ou plusieurs tarifs (dépendant, pour une même prise en charge - pour un même GHM -, de niveaux d'équipements différents par exemple). L'unité statistique retenue dans le cadre de la T2A MCO n'est donc pas celle du GHM, mais celle d'un GHM croisé à un GHS. Plus particulièrement:
- un GHM est codé dans le PMSI par une suite alpha-numérique constituée de 6 caractères, dont les 3 premiers et le dernier sont signifiants et apportent des informations sur le séjour
  - Les deux premiers indiquent la spécialité de prise en charge, dite catégorie majeure de diagnostic ou CMD, et regroupent 28 catégories libellées de 01, 02, 03 à 28 dans la partie [CMD](#cmd)
  - Le troisième indique la nature de la prise en charge, dite catégorie d'activité de soins ou CAS, et correspondent à une lettre parmi celles de la partie [CAS](#cas), correspondant à l'un des 9 groupes de type de prise en charge (chirurgie, médecine, techniques peu invasives) et de durée de séjour (séjour avec ou sans nuitée)
  - Le dernier caractère indique la complexité du séjour, sa gravité ou sa durée  

A chaque couple de GHM-GHS est ainsi associé un nombre de séjours.

### CMD

Les CMD correspondent le plus souvent à un système fonctionnel (affections du système nerveux, de l'oeil, de l'appareil respiratoire, etc...); elles sont subdivisées en racines de GHM, elles-mêmes subdivisées en GHM.
|CMD    |Label    |
|-------|---------|
|01|Affections du système nerveux|
|02|Affections de l'oeil|
|03|Affections des oreilles, du nez, de la gorge, de la bouche et des dents|
|04|Affections de l'appareil respiratoire|
|05|Affections de l'appareil circulatoire|
|06|Affections du tube digestif|
|07|Affections du système hépatobiliaire et du pancréas|
|08|Affections et traumatismes de l'appareil musculosquelettique et du tissu conjonctif|
|09|Affections de la peau, des tissus sous-cutanés et des seins|
|10|Affections endocriniennes, métaboliques et nutritionnelles|
|11|Affections du rein et des voies urinaires|
|12|Affections de l'appareil génital masculin|
|13|Affections de l'appareil génital féminin|
|14|Grossesses pathologiques, accouchements et affections du post-partum|
|15|Nouveau-nés, prématurés et affections de la période périnatale|
|16|Affections du sang et des organes hématopoïétiques|
|17|Affections myéloprolifératives et tumeurs de siège imprécis ou diffus et/ou CMA|
|18|Maladies infectieuses et parasitaires|
|19|Maladies et troubles mentaux|
|20|Troubles mentaux organiques liés à l'absorption de drogues ou induits par celles-ci|
|21|Traumatismes, allergies et empoisonnements|
|22|Brûlures|
|23|Facteurs influant sur l'état de santé et autres motifs de recours aux services de santé|
|24|Séjours de moins de 2 jours|
|25|Maladies dues à une infection par le VIH|
|26|Traumatismes multiples graves|
|27|Transplantations d'organes|
|28|Séances|
|90|Erreurs et autres séjours inclassables|

### CAS

La CAS d'un séjour est donnée selon la durée du séjour et le type de prise en charge selon l'un des groupes suivants:
|CAS    |Label    |
|-------|---------|
|C|Chirurgie non ambulatoire|
|C|Chirurgie ambulatoire|
|O|Obstétrique-mère|
|N|Obstétrique-enfant|
|K|Techniques peu invasives|
|S|Séances|
|X|Séjours sans acte classant sans nuitée - médecine notamment|
|X| Séjours sans acte classant d'au moins une nuit - médecine notamment|
|Z| Séjours inclassables|

### 6ème caractère du GHM (sévérité)

|6ème caractère    |Label    |
|------------------|---------|
|1 ou A||
|2 ou B||
|3 ou C||
|4 ou D||
|J||
|T||
|Z||
|E||


## MCO

### Analysis of MCO activity

Three different measures have been selected to analyze activity in the field of MCO stays:  

- **The number of stays in full hospitalization and outpatient care**: This is the simplest measure of hospital activity. A stay refers to the period during which a patient is hospitalized:  
  - Full hospitalization (HC) includes all stays of at least one night.  
  - Outpatient hospitalization (HP) includes stays without an overnight stay (except in case of death).  

- **The number of equivalent days**: This measure synthesizes the evolution of activity associated with full hospitalizations and outpatient hospitalizations (an outpatient stay is valued as one day), taking into account the length of stays. This "physical" measure of activity has the advantage of being easily interpretable. It is notably used by the Directorate for Research, Studies, Evaluation, and Statistics (DREES) in its annual overview of healthcare facilities.  

- **The economic volume**: This corresponds to activity-related revenues based on the rates associated with each category of stays, adjusted to neutralize the "price effects" caused by the annual revaluation of these rates. While the evolution of the number of equivalent days is a concrete and simple indicator to calculate, it does not account for differences in costs and, therefore, in health insurance revenues related to hospitalization days depending on the treated pathologies. The economic volume aims to factor in these structural effects: the activity volume is thus calculated as the number of stays weighted by a coefficient representing their cost for health insurance. It does not reflect the "actual" revenues of healthcare facilities, which benefit from annual price effects linked to the revaluation of these rates. An identical rate is applied between former DG and former OQN establishments to make their economic volume comparable.  

- **The average length of stay (DMS)**: The DMS is the ratio of the number of days to the number of stays. This is calculated only for full hospitalizations.

### Number of stays


### Number of equivalent days

### Economic volume

### Holiday effect correction

## HAD

## SMR

## PSY
