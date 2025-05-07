# Bootstrap

Bootstrap is a statistical inference technique based on resampling methods, aiming to provide, from a single sample, a confidence interval for the estimation of a given statistic.

As an example, if we want to estimate the expected value of a distribution, we proceed as follows:
- Loop: for b from 1 to B:
    - draw a bootstrap sample X_{1}^{\*}, X_{2}^{\*}, ..., X_{n}^{\*} from a known distribution \hat{F} (parametric bootstrap) or by sampling with replacement from the initial sample (non-parametric bootstrap)
    - compute the statistic of interest from the bootstrap sample $X_{1}^{\*}, X_{2}^{\*}, ..., X_{n}^{\*}$, for example for the mean: $\hat{\theta_{b}}=\frac{\sum_{i=1}^{n}X_{i}^{*}}{n}$
- The average of these $B$ statistics can then be calculated, denoted $\bar{\hat{\theta}}$, as well as their variance given by: $\frac{1}{B}\sum_{b=1}^{B}(\hat{\theta_{b}} - \bar{\hat{\theta}})^{2}$

The Bootstrap module allows implementation of both non-parametric (empirical) and parametric bootstraps for normal, Poisson, and uniform distributions. Estimations can be carried out for the mean, median, variance, standard deviation, quartiles, etc. Confidence intervals at level $\alpha$ provided by the module are calculated using two different methods:
- So-called "percentile bootstrap" method: the confidence interval corresponds to $(\theta_{b}^{(\frac{\alpha}{2})}, \theta_{b}^{(1-\frac{\alpha}{2})})$ where $\theta_{b}^{(1-\frac{\alpha}{2})}$ corresponds to the $1-\frac{\alpha}{2}$ percentile of the distribution of bootstrap statistics
- So-called "basic bootstrap" method: the confidence interval is calculated using both the percentiles of the distribution of bootstrap statistics and the estimate from the original sample: $(2 \hat{\theta}-\theta_{b}^{(1-\frac{\alpha}{2})}, 2 \hat{\theta}-\theta_{b}^{(\frac{\alpha}{2})})$
