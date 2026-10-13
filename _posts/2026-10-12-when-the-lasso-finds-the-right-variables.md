---
title: "When the lasso finds the right variables: the irrepresentable condition"
date: 2026-10-12
topic: Statistical theory
summary: "The lasso recovers the exact signed support of a sparse regression vector only if the inactive covariates are not too well explained by the active ones. A primal–dual witness argument shows that this irrepresentable condition is essentially necessary and, with a margin, sufficient."
---

The lasso is often used to select variables: fit

$$
\hat\beta \in \argmin_{\beta \in \R^p} \Bigl\{ \frac{1}{2n} \lVert y - X\beta \rVert_2^2 + \lambda \lVert \beta \rVert_1 \Bigr\}
$$

and report the covariates with nonzero coefficients. Prediction error and $$\ell_2$$ estimation error of the lasso are controlled under mild conditions on the design, but selection is a more delicate property. Even with infinite data and the right choice of $$\lambda$$, the lasso can include a variable that does not belong in the model. Whether it does depends on a single vector computed from the design, the irrepresentable quantity $$\Gamma z_S$$ introduced below. Deriving the condition from the optimality conditions of the lasso shows where it comes from and why it cannot be avoided.

## Setting

The design $$X \in \R^{n \times p}$$ is fixed, with columns $$X_1, \dots, X_p$$, and

$$
y = X\beta^* + w, \qquad S = \{j : \beta^*_j \neq 0\}, \qquad s = \abs{S}, \qquad z_S = \sign(\beta^*_S).
$$

Write $$X_S$$ for the columns indexed by $$S$$, assume $$X_S$$ has full column rank, and put

$$
\hat\Sigma_{SS} = \frac{1}{n} X_S^\top X_S, \qquad \Gamma = X_{S^c}^\top X_S (X_S^\top X_S)^{-1}, \qquad \Pi^\perp = I_n - X_S (X_S^\top X_S)^{-1} X_S^\top .
$$

The $$j$$-th row of $$\Gamma$$ contains the least-squares coefficients from regressing the inactive column $$X_j$$ on the active columns. The **irrepresentable condition** asks that

$$
\lVert \Gamma z_S \rVert_\infty < 1 ,
$$

or, in the quantitative form used below, $$\lVert \Gamma z_S \rVert_\infty \le 1 - \gamma$$ for some $$\gamma \in (0, 1]$$. In words: no inactive variable is too well represented, in the direction of the true sign pattern, by the active ones. The lasso estimates $$\beta^*$$ with the **correct signed support** if $$\sign(\hat\beta) = \sign(\beta^*)$$ componentwise, with $$\sign(0) = 0$$.

## Optimality conditions

The lasso objective is convex, so $$\hat\beta$$ is a minimizer if and only if zero belongs to its subdifferential:

$$
\frac1n X^\top (y - X\hat\beta) = \lambda \hat z, \qquad \hat z_j = \sign(\hat\beta_j) \text{ if } \hat\beta_j \neq 0, \quad \hat z_j \in [-1, 1] \text{ if } \hat\beta_j = 0 .
$$

The vector $$\hat z$$ is the dual certificate. For an inactive variable, $$\abs{\hat z_j} \le 1$$ says that the correlation of $$X_j$$ with the residual cannot exceed $$\lambda$$. The lasso solution need not be unique when $$p > n$$, so the first step is a lemma that controls all solutions at once.

<div class="result" markdown="1">
**Lemma 1.** Fix $$\lambda > 0$$.

1. All lasso solutions have the same fitted values $$X\hat\beta$$, hence the same $$\hat z = X^\top (y - X\hat\beta)/(n\lambda)$$.
2. If $$T \subseteq \{1, \dots, p\}$$ and $$\abs{\hat z_j} < 1$$ for every $$j \notin T$$, then every lasso solution vanishes outside $$T$$.
3. If, in addition, $$X_T$$ has full column rank, the lasso solution is unique.
</div>

<div class="proof" markdown="1">
*Proof.* For part 1, write the objective as $$g(X\beta) + \lambda \lVert \beta \rVert_1$$ with $$g(u) = \lVert y - u \rVert_2^2 / (2n)$$, which is strictly convex in $$u$$. If two solutions $$\beta^{(1)}, \beta^{(2)}$$ had $$X\beta^{(1)} \neq X\beta^{(2)}$$, their midpoint would have strictly smaller objective value, by strict convexity of $$g$$ and convexity of the norm. So $$X\hat\beta$$ is common to all solutions, and so is $$\hat z$$.

For part 2, every solution $$\tilde\beta$$ satisfies the optimality condition with the same $$\hat z$$, and a nonzero coordinate $$\tilde\beta_j$$ requires $$\abs{\hat z_j} = 1$$. Hence $$\tilde\beta_j = 0$$ for $$j \notin T$$.

For part 3, all solutions are then minimizers of the objective restricted to vectors supported on $$T$$, which is strictly convex in $$\beta_T$$ because $$\beta_T \mapsto \lVert y - X_T \beta_T \rVert_2^2$$ is strictly convex when $$X_T$$ has full column rank.
</div>

## The primal–dual witness

The proof strategy, due to Wainwright, is to *guess* the solution and then check the guess. If the lasso has the correct signed support, its active part must satisfy the optimality condition on $$S$$ with $$\hat z_S = z_S$$, that is, $$\hat\Sigma_{SS} (\beta^*_S - \hat\beta_S) + X_S^\top w / n = \lambda z_S$$. Solving,

$$
\check\beta_S = \beta^*_S + \hat\Sigma_{SS}^{-1} \Bigl( \frac1n X_S^\top w - \lambda z_S \Bigr), \qquad \check\beta_{S^c} = 0 .
$$

The candidate certificate on $$S^c$$ is then forced. Since $$y - X_S \check\beta_S = X_S(\beta^*_S - \check\beta_S) + w = \Pi^\perp w + \lambda n\, X_S (X_S^\top X_S)^{-1} z_S$$,

$$
\check z_{S^c} = \frac{1}{n\lambda} X_{S^c}^\top (y - X_S \check\beta_S) = \Gamma z_S + \frac{1}{n\lambda} X_{S^c}^\top \Pi^\perp w .
\tag{1}
$$

In (1), the first term is the *shrinkage bias*: because the lasso shrinks the active coefficients by $$\lambda \hat\Sigma_{SS}^{-1} z_S$$, part of the signal remains in the residual, and inactive columns correlated with that leftover signal see a correlation of size $$\lambda (\Gamma z_S)_j$$. The second term is noise projected away from the active columns. Whether the candidate is the lasso solution comes down to two checks, a sign check on $$S$$ and a bound on $$\check z_{S^c}$$; the next lemma makes this precise.

<div class="result" markdown="1">
**Lemma 2.** If $$\sign(\check\beta_S) = z_S$$ and $$\lVert \check z_{S^c} \rVert_\infty < 1$$, then $$\check\beta$$ is the unique lasso solution, and it has the correct signed support. Conversely, if some lasso solution has the correct signed support, then it equals $$\check\beta$$ and $$\lVert \check z_{S^c} \rVert_\infty \le 1$$.
</div>

<div class="proof" markdown="1">
*Proof.* By construction $$X_S^\top (y - X\check\beta)/n = \lambda z_S$$, and by (1) $$X_{S^c}^\top (y - X\check\beta)/n = \lambda \check z_{S^c}$$. Under the two checks, $$(z_S, \check z_{S^c})$$ is a valid subgradient of $$\lVert \cdot \rVert_1$$ at $$\check\beta$$, so $$\check\beta$$ satisfies the optimality condition and is a solution. Uniqueness follows from Lemma 1 with $$T = S$$.

Conversely, a solution with the correct signed support satisfies the optimality condition on $$S$$ with $$\hat z_S = z_S$$. Since $$\hat\Sigma_{SS}$$ is invertible, this determines $$\hat\beta_S = \check\beta_S$$, and then the condition on $$S^c$$ reads $$\hat z_{S^c} = \check z_{S^c}$$ with $$\lVert \hat z_{S^c} \rVert_\infty \le 1$$.
</div>

## Noiseless data: an exact characterization

With $$w = 0$$, (1) reduces to $$\check z_{S^c} = \Gamma z_S$$ and $$\check\beta_S = \beta^*_S - \lambda \hat\Sigma_{SS}^{-1} z_S$$. Lemma 2 immediately gives:

<div class="result" markdown="1">
**Proposition 3.** Let $$w = 0$$.

1. If $$s \ge 1$$, $$\lVert \Gamma z_S \rVert_\infty < 1$$ and $$0 < \lambda < \min_{j \in S} \abs{\beta^*_j} \,/\, \lVert \hat\Sigma_{SS}^{-1} z_S \rVert_\infty$$, the lasso solution is unique and has the correct signed support.
2. If $$\lVert \Gamma z_S \rVert_\infty > 1$$, then for no $$\lambda > 0$$ does any lasso solution have the correct signed support.
</div>

By the second statement, with perfect, noise-free data and with $$\lambda$$ chosen by an oracle, the lasso cannot select the right model. Shrinking $$\lambda$$ does not help, because the shrinkage bias and the threshold $$\lambda$$ scale together.

## Noisy data

With Gaussian noise the same two checks hold with high probability, provided the irrepresentable condition holds with a margin and the nonzero coefficients are large enough to survive both the shrinkage and the noise.

<div class="result" markdown="1">
**Theorem 4.** Let $$w \sim N(0, \sigma^2 I_n)$$. Assume $$\lVert \Gamma z_S \rVert_\infty \le 1 - \gamma$$ for some $$\gamma \in (0, 1]$$, $$\lVert X_j \rVert_2 \le \sqrt n$$ for every $$j \in S^c$$, and $$c_{\min} = \lambda_{\min}(\hat\Sigma_{SS}) > 0$$. Fix $$\delta \in (0, 1)$$. If

$$
\lambda \ge \frac{\sigma}{\gamma} \sqrt{\frac{2 \log(2(p - s)/\delta)}{n}} \qquad \text{and} \qquad \min_{j \in S} \abs{\beta^*_j} > \lambda \lVert \hat\Sigma_{SS}^{-1} z_S \rVert_\infty + \sigma \sqrt{\frac{2 \log(2s/\delta)}{n\, c_{\min}}},
$$

then with probability at least $$1 - 2\delta$$ the lasso solution is unique and $$\sign(\hat\beta) = \sign(\beta^*)$$.

</div>

<div class="proof" markdown="1">
*Proof.* We verify the two checks of Lemma 2.

*Dual feasibility.* For $$j \in S^c$$ let $$V_j = X_j^\top \Pi^\perp w / n$$. It is centred Gaussian with variance $$\sigma^2 \lVert \Pi^\perp X_j \rVert_2^2 / n^2 \le \sigma^2 / n$$, since $$\Pi^\perp$$ is an orthogonal projection and $$\lVert X_j \rVert_2 \le \sqrt n$$. The Gaussian tail bound $$\P(\abs{N(0, v)} \ge t) \le 2 e^{-t^2/(2v)}$$ and a union bound give

$$
\P\Bigl(\max_{j \in S^c} \abs{V_j} \ge \gamma\lambda\Bigr) \le 2(p - s) \exp\Bigl(-\frac{n \gamma^2 \lambda^2}{2\sigma^2}\Bigr) \le \delta
$$

by the choice of $$\lambda$$. Off this event, (1) gives $$\lVert \check z_{S^c} \rVert_\infty \le (1 - \gamma) + \max_j \abs{V_j}/\lambda < 1$$.

*Sign consistency.* Let $$U = \hat\Sigma_{SS}^{-1} X_S^\top w / n \sim N(0, \sigma^2 \hat\Sigma_{SS}^{-1} / n)$$. The diagonal entries of $$\hat\Sigma_{SS}^{-1}$$ are at most $$1/c_{\min}$$, so the same two bounds give $$\max_{j \in S} \abs{U_j} < t := \sigma \sqrt{2 \log(2s/\delta) / (n c_{\min})}$$ outside an event of probability at most $$\delta$$. On the complement, for every $$j \in S$$,

$$
\abs{\check\beta_j - \beta^*_j} \le \lambda \lVert \hat\Sigma_{SS}^{-1} z_S \rVert_\infty + \abs{U_j} < \abs{\beta^*_j},
$$

so $$\check\beta_j$$ has the sign of $$\beta^*_j$$. Both checks hold outside an event of probability at most $$2\delta$$, and Lemma 2 concludes.
</div>

The margin $$\gamma$$ enters through $$\lambda$$: the noise term in (1) must be smaller than the slack $$\gamma$$ left by the shrinkage bias, which forces $$\lambda$$ to be of order $$\gamma^{-1} \sigma \sqrt{\log p / n}$$. A small margin therefore costs a large $$\lambda$$, hence more shrinkage, hence a stronger requirement on the smallest nonzero coefficient.

The condition is also necessary in the presence of noise, in the following sense.

<div class="result" markdown="1">
**Proposition 5.** Let $$w$$ be centred Gaussian, or more generally symmetric about zero, and suppose $$\abs{(\Gamma z_S)_j} > 1$$ for some $$j \in S^c$$. Then for every $$\lambda > 0$$, the probability that some lasso solution has the correct signed support is at most $$1/2$$.
</div>

<div class="proof" markdown="1">
*Proof.* Say $$(\Gamma z_S)_j > 1$$; the other case is symmetric. By Lemma 2, correct signed support implies $$\abs{\check z_j} \le 1$$, and by (1) $$\check z_j = (\Gamma z_S)_j + V_j/\lambda$$. Hence $$V_j \le -\lambda\bigl((\Gamma z_S)_j - 1\bigr) < 0$$. Since $$V_j = X_j^\top \Pi^\perp w / n$$ is symmetric about zero, $$\P(V_j < 0) \le 1/2$$.
</div>

## Three covariates

The simplest failure needs three variables. Let $$X_1, X_2$$ be independent standard normal, let $$X_3 = c X_1 + c X_2 + \sqrt{1 - 2c^2}\, \xi$$ with $$\xi$$ an independent standard normal and $$0 < c < 1/\sqrt 2$$, so that $$X_3$$ is standard normal with correlation $$c$$ with each of $$X_1$$ and $$X_2$$. Take $$\beta^* = (1, 1, 0)$$ and rows of $$X$$ drawn i.i.d. from this law. The population analogue of $$\Gamma$$ is $$\Sigma_{S^c S} \Sigma_{SS}^{-1} = (c, c)$$, so

$$
\Gamma z_S \to 2c \quad \text{almost surely as } n \to \infty .
$$

The irrepresentable condition holds for $$c < 1/2$$ and fails for $$c > 1/2$$, although all pairwise correlations stay below $$0.71$$ and $$X_3$$ is not collinear with the active columns. The mechanism is visible directly: $$\Cov(X_3, Y) = 2c$$ exceeds $$\Cov(X_1, Y) = 1$$ when $$c > 1/2$$, so $$X_3$$ is the variable most correlated with the response and enters the lasso path first. Once in, it stays in over the whole range of $$\lambda$$ at which the true coefficients are noticeably shrunk.

<figure>
  <img src="/assets/figures/when-the-lasso-finds-the-right-variables/lasso-paths.svg" alt="Lasso coefficient paths for the three-variable design with c equal to 0.4 and 0.65.">
  <figcaption>Figure 1. Lasso paths for one sample of size $n = 1000$ with $\sigma = 1$. Left: $c = 0.4$, where $2c = 0.8 < 1$; the inactive coefficient stays at zero over the whole range of useful $\lambda$. Right: $c = 0.65$, where $2c = 1.3 > 1$; the inactive variable enters first and remains active while the true variables are shrunk. The values $\lVert \Gamma z_S \rVert_\infty$ in the titles are the population limits.</figcaption>
</figure>

<figure>
  <img src="/assets/figures/when-the-lasso-finds-the-right-variables/sign-recovery.svg" alt="Probability of exact signed support recovery against sample size for the two designs.">
  <figcaption>Figure 2. Probability of exact signed support recovery, estimated from $200$ replications, for the same design augmented with $47$ independent standard normal covariates ($p = 50$), using $\lambda_n = 4\sigma\sqrt{2\log p / n}$. For $c = 0.4$ the probability tends to one; for $c = 0.65$ it is zero at every sample size, in line with Proposition 5.</figcaption>
</figure>

## Remarks

The condition depends on $$\beta^*$$ only through its sign pattern. Requiring it for all sign patterns gives the stronger incoherence condition $$\max_{j \in S^c} \lVert \Gamma_{j\cdot} \rVert_1 \le 1 - \gamma$$, the form used by Wainwright. Meinshausen and Bühlmann identified the same requirement, under the name neighbourhood stability, in their analysis of the lasso for graphical model selection.

Failure of the condition does not make the lasso useless. Bounds on prediction error and on $$\lVert \hat\beta - \beta^* \rVert_2$$ require only restricted eigenvalue or compatibility conditions, which hold in the example above for every $$c < 1/\sqrt 2$$. What fails is exact selection. Two standard remedies change the shrinkage bias that drives (1): the adaptive lasso of Zou reweights the penalty by an initial estimate, and thresholding the lasso estimate removes small spurious coefficients when the true ones are large.

References: P. Zhao and B. Yu, "On model selection consistency of Lasso", *Journal of Machine Learning Research* (2006); N. Meinshausen and P. Bühlmann, "High-dimensional graphs and variable selection with the Lasso", *Annals of Statistics* (2006); M. J. Wainwright, "Sharp thresholds for high-dimensional and noisy sparsity recovery using $$\ell_1$$-constrained quadratic programming (Lasso)", *IEEE Transactions on Information Theory* (2009); H. Zou, "The adaptive lasso and its oracle properties", *Journal of the American Statistical Association* (2006); and the book of P. Bühlmann and S. van de Geer, *Statistics for High-Dimensional Data* (2011).
