---
title: "Cover's universal portfolio: averaging over rebalancing rules and a logarithmic regret bound"
date: 2026-10-12
topic: Quantitative finance
summary: "Invest as if the wealth were spread over all constant rebalancing rules at once. The resulting portfolio is a posterior mean, its wealth is a prior average, and a two-line argument shows that it trails the best rule in hindsight by at most $$(m-1)\\log(n+1) + 1$$ in log wealth for every sequence of returns, with no probabilistic assumptions."
---

Most of portfolio theory starts from a model for returns: a mean vector and a covariance matrix, or a stochastic process. Cover's universal portfolio ("Universal portfolios", *Mathematical Finance*, 1991) starts from none. It asks for a strategy whose wealth, for *every* sequence of returns, is almost as large as that of the best strategy in a benchmark class chosen with hindsight. The benchmark class is the constant rebalanced portfolios, and the answer is a Bayesian-looking average over that class. The setting is a prototype of online learning with the logarithmic loss, and the proof of the main bound is short enough to give in full.

## Constant rebalanced portfolios

There are $$m$$ assets and $$n$$ trading periods. In period $$t$$, asset $$i$$ has price relative $$x_{t,i} \ge 0$$, the ratio of its closing to its opening price, and we assume each vector $$x_t = (x_{t,1}, \dots, x_{t,m})$$ has at least one positive entry. A portfolio is a vector $$b$$ in the simplex $$\Delta_m = \{b \in [0,1]^m : \sum_i b_i = 1\}$$, the fractions of current wealth held in each asset. Holding $$b$$ during period $$t$$ multiplies wealth by $$b^\top x_t$$.

A **constant rebalanced portfolio** (CRP) trades at the end of every period to restore the same fractions $$b$$. Starting from wealth one, its wealth after $$n$$ periods is

$$
S_n(b) = \prod_{t=1}^n b^\top x_t .
$$

Rebalancing means selling assets that have risen relative to the others and buying those that have fallen. For two assets with fraction $$w$$ in the first and per-period log returns $$r_{t,1}$$, $$r_{t,2}$$, a second-order expansion of $$\log(w e^{r_{t,1}} + (1-w) e^{r_{t,2}})$$ gives

$$
\log S_n(w) \approx w \sum_t r_{t,1} + (1 - w) \sum_t r_{t,2} + \frac{w(1-w)}{2} \sum_t (r_{t,1} - r_{t,2})^2 ,
$$

the expansion being accurate when the per-period returns are small. The last term is nonnegative: it is half the variance of the two-point distribution that puts mass $$w$$ on $$r_{t,1}$$ and $$1 - w$$ on $$r_{t,2}$$. When the relative price of the two assets oscillates, this term can make a CRP grow even if neither asset does. The best CRP in hindsight,

$$
S_n^* = \max_{b \in \Delta_m} S_n(b),
$$

which exists because $$S_n$$ is continuous on the compact set $$\Delta_m$$, is therefore a demanding benchmark: it is at least as good as the best single asset (a vertex of $$\Delta_m$$) and often much better. It cannot be used to trade, because $$b^*$$ depends on the whole future.

## The universal portfolio

Let $$\mu$$ be the uniform probability measure on $$\Delta_m$$. The universal portfolio uses, in period $$t + 1$$,

$$
\hat b_{t+1} = \frac{\int_{\Delta_m} b\, S_t(b)\, \mu(\dd b)}{\int_{\Delta_m} S_t(b)\, \mu(\dd b)}, \qquad \hat b_1 = \int b \,\mu(\dd b) = \bigl(\tfrac1m, \dots, \tfrac1m\bigr),
$$

with $$S_0 \equiv 1$$. It is the average of all portfolios, weighted by how well each would have done so far. In statistical language, $$\mu$$ is a prior on $$\Delta_m$$, $$S_t(b)$$ plays the role of a likelihood, and $$\hat b_{t+1}$$ is the posterior mean. Its wealth $$\hat S_n = \prod_t \hat b_t^\top x_t$$ has a simple form.

<div class="result" markdown="1">
**Lemma 1.** $$\hat S_n = \int_{\Delta_m} S_n(b)\, \mu(\dd b)$$ for every $$n \ge 0$$.
</div>

<div class="proof" markdown="1">
*Proof.* By induction; the case $$n = 0$$ is trivial. If $$\hat S_{n-1} = \int S_{n-1}\, \dd\mu$$, then

$$
\hat S_n = \hat S_{n-1}\, \hat b_n^\top x_n = \int S_{n-1}\, \dd\mu \cdot \frac{\int (b^\top x_n)\, S_{n-1}(b)\, \mu(\dd b)}{\int S_{n-1}\, \dd\mu} = \int S_{n-1}(b)\, b^\top x_n\, \mu(\dd b) = \int S_n\, \dd\mu .
$$

The denominator is positive because $$S_{n-1}(b) > 0$$ for every $$b$$ in the interior of $$\Delta_m$$.
</div>

The universal portfolio is thus equivalent to dividing the initial wealth among all CRPs according to $$\mu$$ and letting each run without further intervention. Its wealth is the prior average of the CRP wealths, which is the marginal likelihood in the Bayesian analogy. An average can be far below a maximum, but not when the integrand is a product of $$n$$ factors that varies smoothly in $$b$$: portfolios near $$b^*$$ do almost as well as $$b^*$$, and they carry a non-negligible share of the prior.

## The regret bound

<div class="result" markdown="1">
**Theorem 2.** For every $$n \ge 1$$ and every sequence $$x_1, \dots, x_n$$ of nonnegative vectors, each with a positive entry,

$$
\log S_n^* - \log \hat S_n \le (m - 1)\log(n + 1) + 1 .
$$
</div>

<div class="proof" markdown="1">
*Proof.* Let $$b^*$$ attain $$S_n^*$$ and fix $$\alpha \in (0, 1)$$. Consider the set of portfolios that keep a fraction $$1 - \alpha$$ in $$b^*$$,

$$
N_\alpha = \bigl\{(1 - \alpha) b^* + \alpha z : z \in \Delta_m\bigr\} \subseteq \Delta_m .
$$

For $$b = (1-\alpha)b^* + \alpha z \in N_\alpha$$ and every $$t$$, $$b^\top x_t = (1-\alpha) b^{*\top} x_t + \alpha z^\top x_t \ge (1-\alpha) b^{*\top} x_t$$ because $$z^\top x_t \ge 0$$. Multiplying over $$t$$, $$S_n(b) \ge (1-\alpha)^n S_n^*$$ on $$N_\alpha$$. The set $$N_\alpha$$ is the image of $$\Delta_m$$ under a translation composed with a scaling by $$\alpha$$, and $$\Delta_m$$ is $$(m-1)$$-dimensional, so $$\mu(N_\alpha) = \alpha^{m-1}$$. By Lemma 1,

$$
\hat S_n = \int_{\Delta_m} S_n\, \dd\mu \ge \int_{N_\alpha} S_n\, \dd\mu \ge \alpha^{m-1} (1-\alpha)^n S_n^* .
$$

Choose $$\alpha = 1/(n+1)$$. Then $$(1 - \alpha)^n = (1 + 1/n)^{-n} \ge e^{-1}$$, and taking logarithms gives the claim.
</div>

The bound holds for every sequence, including one chosen by an adversary who knows the strategy. It implies that the growth rates agree asymptotically, $$n^{-1}(\log S_n^* - \log \hat S_n) \to 0$$, uniformly over all markets. This is the sense in which the portfolio is universal: whatever the market does, it achieves the growth rate of the best CRP for that market, without knowing which CRP that is.

The proof also shows which features matter. The factor $$\alpha^{m-1}$$ is the prior mass near $$b^*$$, so the regret grows with the dimension of the simplex; the factor $$(1 - \alpha)^n$$ controls how much wealth is lost by portfolios in that neighbourhood. Balancing the two gives a logarithmic price for not knowing $$b^*$$. With the Dirichlet$$(\tfrac12, \dots, \tfrac12)$$ prior in place of the uniform one, Cover and Ordentlich ("Universal portfolios with side information", *IEEE Transactions on Information Theory*, 1996) obtain a bound of the form $$\tfrac{m-1}{2}\log n + O(1)$$, and Ordentlich and Cover (*Mathematics of Operations Research*, 1998) show that this order cannot be improved by any strategy in the worst case.

## A simulated market

Figure 1 uses two simulated stocks over $$n = 750$$ periods whose log prices are $$F_t + s_t$$ (with a small drift added) and $$F_t - s_t$$, where $$F_t$$ is a Gaussian random walk shared by both and $$s_t = 0.9\, s_{t-1} + \eta_t$$ is a stationary autoregressive spread with $$\eta_t \sim N(0, 0.04^2)$$. Over the sample, stock A gains $$4\%$$ and stock B loses $$5\%$$, and buying and holding equal amounts ends roughly where it started. The best CRP holds $$w^* = 0.52$$ in A and multiplies its wealth by $$1.78$$: the mean-reverting spread makes $$\sum_t (r_{t,1} - r_{t,2})^2$$ large, and the rebalancing term in the expansion above is worth about $$0.6$$ in log wealth at $$w = \tfrac12$$. The universal portfolio multiplies its wealth by $$1.49$$.

<figure>
  <img src="/assets/figures/cover-universal-portfolio-regret/wealth.svg" alt="Log wealth over time of two stocks, buy and hold, the best constant rebalanced portfolio and the universal portfolio, and final wealth of every constant rebalanced portfolio">
  <figcaption>Figure 1. Left: log wealth of the two stocks, of buy-and-hold with half in each, of the best CRP in hindsight and of the universal portfolio. Right: final wealth $S_n(b)$ of the CRP with weight $w$ on stock A; the horizontal line is the wealth of the universal portfolio, which by Lemma 1 is the average height of the curve.</figcaption>
</figure>

The right panel of Figure 1 is Lemma 1 made visible: the wealth of the universal portfolio is the average of the curve $$w \mapsto S_n(w)$$ over $$[0, 1]$$, computed here on a grid of $$4001$$ points; the product $$\prod_t \hat b_t^\top x_t$$ computed period by period agrees with that average to numerical precision. Figure 2 shows the portfolio weights and the regret. The best CRP for the data observed so far jumps around in the early periods, while the universal portfolio, a posterior mean, moves smoothly and stays close to $$w^*$$. The final regret is $$0.18$$, far below the worst-case bound $$\log 751 + 1 \approx 7.6$$; the bound protects against all sequences, and this one is benign.

<figure>
  <img src="/assets/figures/cover-universal-portfolio-regret/weights-regret.svg" alt="Weight on stock A of the universal portfolio and of the best constant rebalanced portfolio so far, and regret over time against the theoretical bound">
  <figcaption>Figure 2. Left: weight on stock A of the universal portfolio and of the best CRP for the data up to each time; the dashed line is $w^*$ for the whole sample. Right: regret $\log S_t^* - \log \hat S_t$ against the bound of Theorem 2 with $m = 2$.</figcaption>
</figure>

## Remarks

The analysis ignores transaction costs, which a strategy that rebalances every period pays in full, and it compares only with CRPs, which are a narrow class. Computing $$\hat b_t$$ requires an integral over $$\Delta_m$$, which is easy for two or three assets and hard in high dimension; Kalai and Vempala ("Efficient algorithms for universal portfolios", *Journal of Machine Learning Research*, 2002) approximate it by sampling with a random walk. Since $$b \mapsto -\log(b^\top x_t)$$ is exp-concave, the problem is also a special case of online convex optimization, where methods such as the online Newton step of Hazan, Agarwal and Kale ("Logarithmic regret algorithms for online convex optimization", *Machine Learning*, 2007) achieve logarithmic regret at much lower cost. The connection with universal data compression, where mixtures over a parametric family of codes achieve redundancy of order $$\tfrac{k}{2}\log n$$, is developed in Cover and Thomas, *Elements of Information Theory*.
