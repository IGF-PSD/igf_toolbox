# Industrial Economics indicators for market concentration

## Gini Index

Used as a measure of inequality, the Gini Index can also be used to measure how evenly market shares are distributed across companies. A Gini Index of 0 indicates perfect equality in market shares between firms, whereas a Gini Index of 1 indicates maximum inequality in terms of market share distribution i.e. a total monopoly. 

To compute the Gini Index:
- Sort the market shares in ascending order
- Calculate the Gini Index using this formula : $\boxed{\frac{2}{n}\frac{\sum_{i=1}^{n}i \times s_{i}}{\sum_{i=1}^{n}s_{i}}-\frac{n+1}{n}}$, where $n$ is the total number of companies and $s_{i}$ is the market share of the $i$ -th company after sorting market shares in ascending order

## Lorenz curve

Graphical representation that shows for the bottom $x$% of firms in terms of market shares, the percentage of the total market share $y$% of an industry. The percentage of firms is plotted on the x-axis and the percentage of total market share on the y-axis.

A perfect competitive market is represented by a Lorenz curve where the bottom $k$% have $k$% of the total market share. This is depicted by the straight line $y=x$. By contrast, a perfectly monopolistic distribution would be one in which $y=0$% for all $x<100$% and $y=100$% when $x=100$%.

The Gini Index can be computed using the Lorenz Curve. If $A$ is the area between the $y=x$ line and the Lorenz Curve and $B$ is the area between the Lorenz Curve and the x-axis, then the Gini Index is given by $\boxed{\frac{A}{A+B}}$

## Concentration Ratio ($CR_{k}$)

A concentration ratio for a pre-specified number $k$ of firms ($CR_{k}$) is the sum of the market shares of the $k$ largest companies in a market of $n$ companies. $CR_{k}$ shows the combined market share of the $k$ largest companies in the industry. The usual values for $k$ are $\lbrace 3, 4, 5, 6, 7, 8 \rbrace$ corresponding to $CR_{3}, CR_{4}, CR_{5}, CR_{6}, CR_{7}, CR_{8}$. For example, $CR_{4}$ defines the combined market share of the four largest firms in a market. The formula of the concentration ratio is given by : $\boxed{CR_{k}=\sum_{i=1}^{k}s_{i}=s_{1}+s_{2}+ \ldots + s_{k}}$, where $s_{j}$ is the market share of the jth-largest company in the market (for example $s_{1}$ is the market share of the largest company in the industry).

The most commonly used concentration ratios are $CR_{4}$ and $CR_{8}$, and their values range from 0 (lowest concetration level) to 1 (highest concentration level), or from 0 to 100 if market shares are expressed in percentages. The different possible concentration levels are the following :
- Perfect concentration : from $CR_{k}=0$ to $CR_{k}=\frac{k}{n}$, all firms have equal market share
- Low concentration : from $CR_{k}=\frac{k}{n}$ to $CR_{k}=0,4$
- Medium concentration : from $CR_{k}=0,4$ to $CR_{k}=0,7$, the market is likely an oligopoly : the industry is dominated by a small number of firms each with significant market shares
- High concentration :  from $CR_{k}=0,7$ to $CR_{k}=1$, this concentration level ranges from oligopoly to monopoly

Concentration Ratio has however a shortfall as it doesn't capture the distribution of market shares but only the sum of the $k$ largest market shares. For instance, for an industry A and B, where the 4 largest market shares are the following :
- A : 20%-20%-20%-20%
- B: 35%-25%-10%-10%
  
The $CR_{4}$ for A and B is 80%, whereas it is evident that B is more concentrated than A.

## Herfindahl-Hirschman Index ($HHI$)

The Herfindahl-Hirschman Index is calculated by summing the squares of the market shares of each company competing in a market: $\boxed{HHI=\sum_{i=1}^{n} s_{i}^{2}}$ where $s_{i}$ is the market share of company $i$ and $n$ is the number of companies in the market. Thus, for a market held by two companies with market shares of 30% and 70%, the $HHI$ is $HHI=0,3^{2}+0,7^{2}=0,09+0,49=0,58$.

The $HHI$ ranges from $\frac{1}{n}$ to 1, and when market shares are provided as percentages, the $HHI$ can go up to 10,000. Typically, above a threshold of 0.25 (or 2,500), the market is considered concentrated.

In general, the closer the $HHI$ is to 1 (or 10,000), the more monopolistic the market is.

A normalized $HHI$ also exists, ensuring index values between 0 and 1, regardless of the input data. The formula for the normalized $HHI$ is: $\boxed{\hat{HHI}=\frac{HHI-\frac{1}{n}}{1-\frac{1}{n}}}$

## Shannon entropy

Shannon entropy is a useful measure for market concentration because it quantifies the diversity in a market, the lower the entropy, the higher is the concentration. The formula of the Shannon entropy is given by : $\boxed{-\sum_{i=1}^{n}s_{i}\ln(s_{i})}$. When the entropy is 0 the market is in a situation of monopoly, when the entropy is $\ln(n)$, the market is in a situation of perfect competition.

Shannon entropy is defined for probability densities, so at least one market share should be different from 0.

## Theil Index

The Theil Index is an entropy-based measure of concentration in a market that:
- is sensitive to distribution : unlike simpler measures such as concentration ratio, the Theil Index captures how uneven the market shares are
- decomposable into within-group and between-group components, useful for analyzing different sectors
- entropy-based : it accounts for information theory principles, making it more theoretically grounded

The Theil Index is given by: $\boxed{T=\frac{1}{n}\sum_{i=1}^{n}(\frac{s_{i}}{\overline{s}})\ln(\frac{s_{i}}{\overline{s}})}$

A Theil Index close to 0 indicates a highly competitive market, the higher the Theil Index is the more concentrated the market is. The Theil Index is an entropic measure to perfect competition in a market and can be interpreted as follow:
- $T=0$: perfect competition
- $0<T<0.2$: high competition
- $0.2\leq T < 0.5$: moderate concentration
- $0.5 \leq T < 1$: high concentration
- $1 \leq T$ : very high concentration
