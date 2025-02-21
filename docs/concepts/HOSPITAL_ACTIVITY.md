# Public Health Economics indicators for hospital activity measurement

## *MCO* activity

The statistical unit to be considered in the *PMSI* for an economic analysis of *MCO* hospital activity is the intersection of a *GHM* (*Groupe homogène de malades*) and a *GHS* (*Groupe homogène de séjours*).

A **GHM** groups together cases of the same medical and economic nature and constitutes the elementary category of classification in *MCO*. Each stay is assigned to a *GHM* according to an algorithm based on the medico-administrative information contained in the *résumé de sortie standardisé* (*RSS*) of each patient.  

A *GHS* corresponds, within the framework of *T2A*, to the tariff of the *groupe homogène de malades*. The vast majority of *GHM* correspond to only a single *GHS*, meaning a single tariff. However, some *GHM* may be assigned to two or more tariffs (depending, for the same case — for the same *GHM* — on different levels of equipment, for example).  

Therefore, the statistical unit considered within the framework of *T2A MCO* is not the *GHM* itself but a *GHM* combined with a *GHS*. More particularly:  
- A *GHM* is coded in the *PMSI* by an alphanumeric sequence consisting of 6 characters, of which the first three and the last one are significant and provide information about the stay.  
  - The first two indicate the specialty of care, called *catégorie majeure de diagnostic* or *CMD*, and group 28 categories labeled from 01, 02, 03 to 28 in the section [CMD](#cmd).  
  - The third indicates the nature of the care, called *catégorie d'activité de soins* or *CAS*, and corresponds to a letter among those in the section [CAS](#cas), representing one of the 9 groups of care types (surgery, medicine, minimally invasive techniques) and length of stay (stay with or without an overnight stay).  
  - The last character indicates the complexity of the stay, its severity, or its duration, with level 1 being the lowest severity level and 4 the highest. This last character follows this [classification](#6th-character-of-ghm).

As an example, the *GHM* *08C471*, whose label is *Prothèses de hanche pour traumatismes récents, niveau 1*, indicates that it belongs to *CMD* *08* (*Affections et traumatismes de l'appareil musculosquelettique et du tissu conjonctif*), meaning orthopedics, then surgery (the third character being *C*), and finally that the severity level is low (level 1). Moreover, the middle number, *47*, is only used to sequence this *GHM* within all orthopedic surgery *GHM* in this case.  

Finally, the first five characters of the *GHM* correspond to the *racine du GHM* and group together all *GHM* of the same *CMD* and the same type of care, regardless of the severity level of the stay.  

As an example, the *GHM* *08C471* belongs to the *racine* *08C47*, along with the *GHM*: *08C472*, *08C473*, and *08C474*.

The definition of *GHS*, on the other hand, is based on an average valuation concept assessed in relation to costs and public health plans, a concept adapted in view of concrete situations such as:  

- differences in length of stay (*EXB* or *EXH* stays corresponding respectively to very short or very long durations relative to the *GHM*)  
- differences in the complexity of care due to admission to a specialized unit (intensive care, high-dependency care, continuous monitoring, neonatology), leading to the creation of additional daily supplements  
- expensive drugs from the additional list (*molécules onéreuses de la liste en sus*)
- etc.

The *GHM* and *GHS* make it possible to measure hospital activity and manage their performance and resources according to the hospital *casemix*.  

Each *GHM-GHS* pair is thus associated with:  
- a number of stays or sessions  
- a number of days  
- a type of hospitalization:  
  - full hospitalization (*HC*): when the stay required at least one overnight stay  
  - outpatient hospitalization (*HP*): when the stay did not require an overnight stay  
- an average length of stay (*DMS*), corresponding to the ratio of the number of days to the number of stays, and defined only for *HC* stays  
- an average price  

### *CMD*

The *CMD* most often correspond to a functional system (nervous system disorders, eye disorders, respiratory system disorders, etc.); they are subdivided into *racines de GHM*, which are themselves subdivided into *GHM*.  

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

### *CAS*

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

### 6th character of *GHM*

|6ème caractère    |Label    |
|------------------|---------|
|1 ou A|Niveaux de sévérité 1 ou A|
|2 ou B|Niveaux de sévérité 2 ou B|
|3 ou C|Niveaux de sévérité 3 ou C|
|4 ou D|Niveaux de sévérité 4 ou D|
|J|Séjours en ambulatoire|
|T|Séjours de très courte durée|
|Z|Séjours sans niveau de sévérité|
|E|Séjours avec décès|

### Analysis of *MCO* activity

Three different measures have been selected to analyze activity in the field of *MCO* stays:  

- **The number of stays in full hospitalization and outpatient care**: This is the simplest measure of hospital activity. A stay refers to the period during which a patient is hospitalized:  
  - Full hospitalization (HC) includes all stays of at least one night.  
  - Outpatient hospitalization (HP) includes stays without an overnight stay (except in case of death).  

- **The number of equivalent days**: This measure synthesizes the evolution of activity associated with full hospitalizations and outpatient hospitalizations (an outpatient stay is valued as one day), taking into account the length of stays. This "physical" measure of activity has the advantage of being easily interpretable. It is notably used by the Directorate for Research, Studies, Evaluation, and Statistics (DREES) in its annual overview of healthcare facilities.  

- **The economic volume**: This corresponds to activity-related revenues based on the rates associated with each category of stays, adjusted to neutralize the "price effects" caused by the annual revaluation of these rates. While the evolution of the number of equivalent days is a concrete and simple indicator to calculate, it does not account for differences in costs and, therefore, in health insurance revenues related to hospitalization days depending on the treated pathologies. The economic volume aims to factor in these structural effects: the activity volume is thus calculated as the number of stays weighted by a coefficient representing their cost for health insurance. It does not reflect the "actual" revenues of healthcare facilities, which benefit from annual price effects linked to the revaluation of these rates. An identical rate is applied between former DG and former OQN establishments to make their economic volume comparable.  

#### Number of stays


#### Number of equivalent days

#### *Economic volume*

#### Holiday effect correction

## *HAD*, *SMR* and *PSY*
