# Bootstrap

Le bootstrap est une technique d'inférence statistique fondée sur des techniques de rééchantillonnage et dont le but est de fournir, à partir d'un échantillon, un intervalle de confiance pour l'estimation d'une statistique donnée.

A titre d'exemple, si l'on désire estimer l'espérance d'une loi, on procède comme suit:
- Boucle: pour $b$ allant de 1 à $B$:
    - on tire un échantillon bootstrap $X_{1}^{\*}, X_{2}^{\*}, ..., X_{n}^{\*}$ selon une loi connue $\hat{F}$ (bootstrap paramétrique) ou selon un tirage avec remise (bootstrap non paramétrique)
