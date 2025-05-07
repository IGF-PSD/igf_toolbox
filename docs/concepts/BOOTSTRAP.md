# Bootstrap

Le bootstrap est une technique d'inférence statistique fondée sur des techniques de rééchantillonnage et dont le but est de fournir, à partir d'un échantillon, un intervalle de confiance pour l'estimation d'une statistique donnée.

A titre d'exemple, si l'on désire estimer l'espérance d'une loi, on procède comme suit:
- Boucle: pour $b$ allant de 1 à $B$:
    - on tire un échantillon bootstrap $X_{1}^{\*}, X_{2}^{\*}, ..., X_{n}^{\*}$ selon une loi connue $\hat{F}$ (bootstrap paramétrique) ou selon un tirage avec remise à partir de l'échantillon initial (bootstrap non paramétrique)
    - on calcule à partir de l'échantillon bootstrap $X_{1}^{\*}, X_{2}^{\*}, ..., X_{n}^{\*}$ la statistique d'intérêt, par exemple pour l'espérance : $\hat{\theta_{b}}=\frac{\sum_{i=1}^{n}X_{i}^{*}}{n}$
- La moyenne des ces $B$ statistiques peut être alors calculée, notée $\bar{\hat{\theta}}$, ainsi que leur variance donnée par : $\frac{1}{B}\sum_{b=1}^{B}(\hat{\theta_{b}} - \bar{\hat{\theta}})^{2}$

Le module Bootstrap permet d'implémenter à la fois des bootstraps non paramétriques, ou empiriques, et des bootstraps paramétriques pour des lois normale, de Poisson et uniforme. Les estimations peuvent être conduites pour la moyenne, la médiane, la variance, l'écart-type, des quartiles, ... Les intervalles de confiance à un niveau $\alpha$ fournis par le module, le sont selon deux méthodes différentes:
- Méthode dite "percentile bootstrap" : l'intervalle de confiance correspond à $(\theta_{b}^{(\frac{\alpha}{2})}, \theta_{b}^{(1-\frac{\alpha}{2})})$ où $\theta_{b}^{(1-\frac{\alpha}{2})}$ correspond au percentile $1-\frac{\alpha}{2}$ de la distribution des statistiques issues du bootstrap
- Méthode dite "basic bootstrap": l'intervalle de confiance est calculé à la fois à partir des percentiles de la distribution des statistiques issues du bootstrap et de l'estimation faite sur l'échantillon initial : $(2 \hat{\theta}-\theta_{b}^{(1-\frac{\alpha}{2}), 2 \hat{\theta}-\theta_{b}^{(\frac{\alpha}{2}))$