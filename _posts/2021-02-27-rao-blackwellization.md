---
layout: post
title: "Rao-Blackwellization"
author: "Binh Ho"
blurb: "Estimators based on sampling schemes can be 'Rao-Blackwellized' to reduce their variance."
img: ""
categories: Statistics
tags: []
<!-- image: -->
---

Estimators based on sampling schemes can be "Rao-Blackwellized" to reduce their variance.

Here, I explain some of the background and detail of Rao-Blackwellization as described in [Casella and Robert's paper](https://www.jstor.org/stable/2337434?seq=1#metadata_info_tab_contents). I'll specifically focus on the technique in the cnotext of the Accept-Reject algorithm, which is one of the sampling-based approaches considered in the paper.

## Rao-Blackwell Theorem

The Rao-Blackwell Theorem is a famous and very general result about the relative variance of estimators. Specifically, given an estimator $\widehat{\theta}_n$ of some parameter $\theta \in \Theta$, the theorem provides a way to improve the estimator using a conditional expectation. Below is one version of the theorem.

> **Theorem (Rao-Blackwell)**. Let $\widehat{\theta}\_n$ be an unbiased estimator of some parameter $\theta \in \Theta$, and let $T(X)$ be a sufficient statistic for $\theta$. Then, 1) $\widetilde{\theta}\_n = \mathbb{E}\left[ \widehat{\theta}\_n \| T(X) \right]$ is an ubiased estimator for $\theta$, and 2) $\mathbb{V}\_\theta\left[ \widetilde{\theta}\_n \right] \leq \mathbb{V}\_\theta\left[ \widehat{\theta}\_n \right]$ for all $\theta$.

In other words, if we take the conditional expectation of $\widehat{\theta}_n$ (conditioned on a sufficient statistic $T(X)$), then the resulting estimator will be "better" in the sense that it has lower variance.

As long as we can find a sufficient statistic and compute the required conditional expectation, the Rao-Blackwell theorem provides a practical way to find uniform minimum variance unbiased (UMVU) estimators given a data sample $X_1, X_2, \dots, X_n$.

One domain in which reducing estimators' variance is particularly desirable is in sampling-based procedures. In a generic sampling-based approach, we randomly sample values from some distribution and manipulate these values in some way to compute a desired quantity $h(x)$.

## Accept-Reject algorithm

The Accept-Reject algorithm is one sampling-based approach for drawing samples from a distribution $f(x)$ and subsequently computing some desired function of the random variable $h(X)$. Specifically, the algorithm uses a proposal distribution $g$ to generate random samples from a related (but usually more tractable) distribution, and then uses a rejection rule to decide to keep or reject each sample. One iteration of the algorithm is sketched below, where $M$ is some constant such that $f(x) \leq Mg(x)$ for all x.

1. Draw $Y \sim g(y)$.
2. Draw $U \sim \text{Unif}(0, 1)$.
3. If $U \leq \frac{f(Y)}{M g(Y)}$, return $X := Y$. Otherwise, go to step 1.

The random variable $X$ is then distributed according to $f(x)$. In practice, we typically run the algorithm for multiple iterations, until $t$ samples are returned. Note that in order to get $t$ samples, we have to run the algorithm $n$ times, where $n$ is a random variable depending on the probability of rejection in each iteration.

Once we have $t$ samples $X_1, \dots, X_t$, we can estimate the expectation of any function $h$,

$$\mathbb{E}_{f(x)}[h(X)]$$

using a sample average,

$$\widehat{\theta} := \mathbb{E}_{f(x)}[h(X)] = \frac1t \sum\limits_{i=1}^t h(X_i).$$

Notice, however, that in order to estimate this quantity, we completely threw out the samples that were rejected by the Accept-Reject algorithm. There are $n-t$ of these samples. Instead of "wasting" these samples, is there any way we can make use of them to improve our estimator?

## Rao-Blackwellization

### Setting up the Rao-Blackwell estimator

The Rao-Blackwell theorem provides a direct way to use the samples discarded by the Accept-Reject algorithm: condition on them. 

To start, note that we can write the estimator as 

$$\frac1t \sum\limits_{i=1}^n I(U_i \leq w_i) h(Y_i)$$

where $w_i = \frac{f(Y_i)}{M g(Y_i)}$ and $I$ is the indicator function. By the Rao-Blackwell theorem, we can then construct an estimator that has lower variance than $\widehat{\theta}$. Using $T(X) = (n, Y_1, \dots, Y_n)$ as a sufficient statistic, we have

\begin{equation} \widehat{\theta}\_{\text{RB}} = \mathbb{E}\left[ \widehat{\theta} \middle\| T(X) \right] = \frac1t \mathbb{E}\left[ \sum\limits_{i=1}^n I(U_i \leq w_i) h(Y_i) \middle\| n, Y_1, \dots, Y_n \right]. \label{eq:rb_estimator} \end{equation}

### Computing the Rao-Blackwell estimator

The next task is to actually evaluate this conditional expectation. To start, we can write down the conditional distribution $p(u_1, \dots, u_n, \| N=n, y_1, \dots, y_n)$. 

Notice that the $n$th sample (the last sample) will always be accepted because acceptance of the $n$th sample terminates the algorithm. Thus, this sample is really independent of the random variable $n$. So the conditional distribution for the random variables of this last iteration is

$$p(u_n | N=n, y_n) = \frac{I(u_n \leq w_n)}{w_n}.$$

Intuitively, this is because, given $w_n$, $u_n$ is uniformly distributed in the interval $[0, w_n]$.

Now, to account for iterations $1, \dots, n-1$, we have to average over all possible subsets of the set $\{1, \dots, n-1\}$ of size $t-1$. To simplify the problem first, suppose we know the indices of the accepted samples --- call them $i_1, \dots, i_t$. Then the conditional distribution at hand is

$$p(u_1, \dots, u_{n-1} | N=n, y_1, \dots, y_{n-1}, i_1, \dots, i_{t-1}) = \prod\limits_{j=1}^{t-1} I(u_{i_j} \leq w_{i_j}) \prod\limits_{j=t}^{n-1} I(u_{i_j} > w_{i_j}).$$

Now, we need to average this quantity over all possible subsets of accepted indices. Without the normalizing factor, we have

$$p(u_1, \dots, u_{n-1} | N=n, y_1, \dots, y_{n-1}) \propto \sum\limits_{(i_1, \dots, i_{t-1})} \prod\limits_{j=1}^{t-1} I(u_{i_j} \leq w_{i_j}) \prod\limits_{j=t}^{n-1} I(u_{i_j} > w_{i_j})$$

where the sum is over all possible subsets of the set $\{1, \dots, n-1\}$ of size $t-1$. Including the normalizing factor, we then have

$$p(u_1, \dots, u_n, | N=n, y_1, \dots, y_n) = \frac{\sum\limits_{(i_1, \dots, i_{t-1})} \prod\limits_{j=1}^{t-1} I(u_{i_j} \leq w_{i_j}) \prod\limits_{j=t}^{n-1} I(u_{i_j} > w_{i_j})}{\sum\limits_{(i_1, \dots, i_{t-1})} \prod\limits_{j=1}^{t-1} w_{i_j} \prod\limits_{j=t}^{n-1} (1-w_{i_j})}\cdot \frac{I(u_n \leq w_n)}{w_n}.$$

Now, the last step is to find $p(U_i \leq w_i \| N = n, Y_1, \dots, Y_n)$. Clearly, for the last ($n$th) sample, this probability is $1$, since this sample met the algorithm's termination criterion by definition. For $i=1,\dots,n-1$, we need to find the CDF for $U_i$, conditional on the sufficient statistic $(N, Y_1, \dots, Y_n)$. We also need to marginalize out all $U_j$ for $j \notin \{i, n\}$. in other words, we need to evaluate

\begin{equation} \int_0^{w_i} \sum\limits_{u_j, j \neq i} p(u_1, \dots, u_{n-1} \| N=n, y_1, \dots, y_n) du_i. \label{eq:weights} \end{equation}

The inner sum marginalizes out $u_1, \dots, u_{i-1}, u_{i+1}, \dots, u_{n-1}$, and the outer integral evaluates the CDF of $u_i$ at $w_i$.

Recall that for a uniform random variable $U$ in $[0, 1]$, we have

$$p(U \leq u) = u.$$

Thus, Equation \eqref{eq:weights} amounts to summing over all permutations of $u_j, j \notin \{i, n\}$. Then we have

$$p(U_i \leq w_i | N = n, Y_1, \dots, Y_n) = \frac{\sum\limits_{(i_1, \dots, i_{t-2})} \prod\limits_{j=1}^{t-2} w_{i_j} \prod\limits_{j=t}^{n-2} (1-w_{i_j})}{\sum\limits_{(i_1, \dots, i_{t-1})} \prod\limits_{j=1}^{t-1} w_{i_j} \prod\limits_{j=t}^{n-1} (1-w_{i_j})}.$$

Denote this expression as $\rho_i$. Then, plugging this into the Rao-Blackwell estimator in \eqref{eq:rb_estimator}, we have the following Rao-Blackwellized estimator for $\mathbb{E}_{f(x)}[h(X)]$ using the Accept-Reject algorithm:

$$\widehat{\theta}_{\text{RB}} = \frac1t \sum\limits_{i=1}^n \rho_i h(Y_i).$$

## Conclusion

Estimators based on sampling schemes can have high variance. Rao-Blackwellization provides a principled way to reduce this variance. Furthermore, in the context of the Accept-Reject algorithm, the Rao-Blackwellized estimator makes use of the "rejected" samples that are normally thrown away in the naive version. 

Rao-Blackwellization can be extended to other sampling schemes as well, such as Metropolis algorithms -- this is explored in Casella and Robert's paper as well.

## References
- Casella, George, and Christian P. Robert. "Rao-Blackwellisation of sampling schemes." Biometrika 83.1 (1996): 81-94.
- Prof. [Matias Cattaneo's](https://cattaneo.princeton.edu/home) notes from Princeton's ORF524 course.
