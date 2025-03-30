# FreeMesh
Official code of FreeMesh: Boosting Mesh Generation with Coordinates Merging



## 1. More Generated Complex Results



## 2. Compare with BPT




## 3. Proof

Given that RAW, AMT, EDR representations and naive merge coordinates all induce a slight increase in **PTME**, we analyze **PCME** using RAW as an exemplar while assuming $ C_R $ remains constant.

Let $ N_i $ denote the frequency of substring $ i $ with total frequency $ N $. We estimate $ p_i = N_i/N $, where $ l_i $ represents the length of substring $ i $. The PCME metric is defined as:

$$
\mathcal{PCME} = \frac{\mathcal{H}}{l} = \frac{-\sum_i p_i \log p_i}{\sum_i p_i l_i}
$$

Consider merging adjacent items $ a $ and $ b $ with joint frequency $ N_{ab} $. Pre-merging probability is $ p_{ab} = N_{ab}/N $. Post-merging, the total frequency becomes $ \tilde{N} = N - N_{ab} $, yielding updated probabilities:

$$
\begin{aligned}
\tilde{p}_{ab} &= \frac{p_{ab}}{1 - p_{ab}}, \\
\tilde{p}_a &= \frac{p_a - p_{ab}}{1 - p_{ab}}, \\
\tilde{p}_b &= \frac{p_b - p_{ab}}{1 - p_{ab}}, \\
\tilde{p}_i &= \frac{p_i}{1 - p_{ab}}, \quad (i \neq a,b)
\end{aligned}
$$

The updated entropy measure becomes:

$$
\begin{aligned}
\tilde{\mathcal{H}} &= -\frac{1}{1 - p_{ab}} \left[ p_{ab} \log \frac{p_{ab}}{1 - p_{ab}} + \sum_{\substack{i=a,b}} (p_i - p_{ab}) \log \frac{p_i - p_{ab}}{1 - p_{ab}} \right. \\
&\quad + \left. \sum_{i \neq a,b} p_i \log \frac{p_i}{1 - p_{ab}} \right] \\
&= \frac{1}{1 - p_{ab}} (\mathcal{H} - \mathcal{F}_{ab})
\end{aligned}
$$

where:

$$
\mathcal{F}_{ab} = p_{ab} \log \frac{p_{ab}}{p_a p_b} - (1 - p_{ab}) \log(1 - p_{ab}) + \sum_{i=a,b} (p_i - p_{ab}) \log \left(1 - \frac{p_{ab}}{p_i} \right)
$$

The effective length transforms as:

$$
\begin{aligned}
\tilde{l} &= \frac{p_{ab}(l_a + l_b) + \sum_{i=a,b} (p_i - p_{ab})l_i + \sum_{i \neq a,b} p_i l_i}{1 - p_{ab}} \\
&= \frac{l}{1 - p_{ab}}
\end{aligned}
$$

Thus, the PCME difference becomes:

$$
\frac{\tilde{\mathcal{H}}}{\tilde{l}} - \frac{\mathcal{H}}{l} = - \frac{\mathcal{F}_{ab}}{l}
$$

For $ p_{ab} \ll p_a, p_b $, we approximate using natural logarithms:

$$
\begin{aligned}
\ln(1 - p_{ab}) &\approx -p_{ab} \\
\ln \left(1 - \frac{p_{ab}}{p_i} \right) &\approx -\frac{p_{ab}}{p_i}
\end{aligned}
$$

Substituting into $ \mathcal{F}_{ab} $ while neglecting higher-order terms yields:

$$
\mathcal{F}_{ab} \approx \mathcal{F}_{ab}^* = p_{ab} \left( \ln \frac{p_{ab}}{p_a p_b} - 1 \right)
$$

where $ \text{PMI}(a, b) = \ln \frac{p_{ab}}{p_a p_b} $ denotes Pointwise Mutual Information. To reduce $ \tilde{\mathcal{H}}/\tilde{l} $, we require $ \mathcal{F}_{ab} \geq 0 $, which necessitates:
- High co-occurrence probability $ p_{ab} $
- Strong mutual information ($ \text{PMI} \geq 1 $)

The observed PCME increase stems from **insufficient** $ p_{ab} $ values. Our rearrangement strategy enhances $ p_{ab} $ by increasing substring co-occurrence probabilities.