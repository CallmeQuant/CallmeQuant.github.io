---
layout: post
title: "Duality between maximum likelihood and maximum entropy"
author: "Binh Ho"
categories: Statistics
blurb: "There exists a duality between maximum likelihood estimation and finding the maximum entropy distribution subject to a set of linear constraints."
img: ""
tags: []
<!-- image: -->
---

There exists a duality between maximum likelihood estimation and finding the maximum entropy distribution subject to a set of linear constraints.


## Reformulating maximum likelihood

Recall that the maximum likelihood estimator (MLE) for a parametric family $\{p_\theta \; : \; \theta \in \Theta\}$ is the member of the set that maximizes the likleihood of the data:

$$p_{\theta_{\text{MLE}}} = \text{arg}\max_{p_\theta} \prod\limits_{i=1}^n p_\theta(x_i).$$

To show the duality with maximum entropy methods, it will be convenient to rewrite the MLE as

\begin{align} p_{\theta_{\text{MLE}}} &= \text{arg}\max_{p_\theta} \prod\limits_{i=1}^n p_\theta(x_i) \\\ &= \text{arg}\max_{p_\theta} \log \prod\limits_{i=1}^n p_\theta(x_i) \\\ &= \text{arg}\max_{p_\theta} \sum\limits_{i=1}^n \log p_\theta(x_i) \\\ &= \text{arg}\min_{p_\theta} \sum\limits_{i=1}^n \log \frac{1}{p_\theta(x_i)} \\\ &= \text{arg}\min_{p_\theta} \mathbb{E}\_{\hat{p}}\left( \log \frac{1}{p_\theta(x)}\right) \\\ \end{align}

where $\hat{p}(x)$ is the empirical PMF of the observed data. Now, if we add and subtract the term $\log \hat{p}(x)$ inside the expectation, we have

\begin{align} p_{\theta_{\text{MLE}}} &= \text{arg}\min_{p_\theta} \mathbb{E}\left( \log \frac{1}{p_\theta(x)} + \log \hat{p}(x) - \log \hat{p}(x) \right) \\\ &= \text{arg}\min_{p_\theta} \mathbb{E}\left( \log \frac{\hat{p}(x)}{p_\theta(x)} + \frac{1}{\log \hat{p}(x)} \right) \\\ &= \text{arg}\min_{p_\theta} \mathbb{E}\left( \log \frac{\hat{p}(x)}{p_\theta(x)}\right) + \mathbb{E}\left(\frac{1}{\log \hat{p}(x)} \right) \\\ \end{align}

Recognizing the first term as the KL-divergence between $\hat{p}$ and $p_\theta(x)$ and the second term as the entropy of $\hat{p}(x)$, we can rewrite this as

$$p_{\theta_{\text{MLE}}} = \text{arg}\min_{p_\theta} KL(\hat{p}(x) || p_\theta(x)) + H(\hat{p})$$

where $H$ represents the entropy. Since $H(\hat{p})$ is constant w.r.t. $p_\theta(x)$, we can ignore this term and the solution can be written as 

$$p_{\theta_{\text{MLE}}} = \text{arg}\min_{p_\theta} KL(\hat{p}(x) || p_\theta(x)).$$

## Equivalence with information projection

Now, can we somehow show that the MLE, which minimizes $KL(\hat{p}(x) \|\| p_\theta(x))$, is equivalent to the information projection of an exponential family base distribution $p_0$ onto a set of distributions $\mathcal{P}$ with some constraints?

Specifically, we'll try to show that

$$\text{arg}\min_{p_\theta} KL(\hat{p}(x) || p_\theta(x)) = \text{arg}\min_{\substack{p \in \mathcal{P} \\ c_j(X) = a_j}} \text{KL}(p || p_0)$$

where $c_j(X) = a_j$ are constraints on the distribution. Specifically, these constraints require the information projection to have the same sufficient statistics as $\hat{p}(x)$:

$$c_j(X) = a_j \iff \mathbb{E}\_p[T_j(X)] = \mathbb{E}\_{\hat{p}}[T_j(X)]$$

where $T_j(X)$ represent the sufficient statistics of the data.

## Brief aside on information projections

The information projection of a distribution $q$ onto a set of distributions is defined as the member of that set that minimizes the KL-divergence to $q$. In other words, $p^*$ is the information projection of $Q$ onto $\mathcal{P}$ if

$$p^* = \text{arg}\min_{p \in \mathcal{P}} \text{KL}(p || q).$$


It can be shown that the information projection above $p_{\text{IP}}$ will always be a member of the exponential family of distributions, since the constraints are all linear in $p$. An equivalent way to think about the information projection is as the problem of finding the maximum entropy distribution that satisfies the set of constraints.

Notice that we can write any distribution that belongs to the exponential family in the following form:

$$p(x) = p_0(x) \frac{\exp\left(\sum_j \theta_j T_j(x)\right)}{Z_\theta}$$

where $T_j(x)$ are the sufficient statistics. The exponential family is often written using many different notations (often coinciding with different academic communities), but it's useful to be able to recognize it in any of them. Another common notation is

$$p(x) = h(x) \exp\left\{ \eta^\top T(x) - A(\eta) \right\}.$$

## Duality


Now, we must show that the MLE parameters will satisfy the constraints. To do this, let's solve for the MLE by taking the partial derivative of the log-likelihood to each parameter $\theta_j$ and setting to $0$.

\begin{align} \frac{\partial}{\partial \theta_j} LL &= \frac{\partial}{\partial \theta_j} \sum\limits_{i = 1}^n \log\left(p_0(x_i) \frac{\exp\left(\sum_j \theta_j T_j(x_i)\right)}{Z_\theta}\right) \\\ &= \frac{\partial}{\partial \theta_j} \sum\limits_{i = 1}^n \left(\log p_0(x_i) + \sum_j \theta_j T_j(x_i) - \log Z_\theta \right) \\\ &= \sum\limits_{i = 1}^n  T_j(x_i) - \frac{\partial}{\partial \theta_j} \sum\limits_{i = 1}^n \log Z_\theta \\\ &= \sum\limits_{i = 1}^n  T_j(x_i) - n\frac{\partial}{\partial \theta_j} \log Z_\theta \\\ &= \sum\limits_{i = 1}^n  T_j(x_i) - n \frac{1}{Z_\theta} \frac{\partial}{\partial \theta_j} Z_\theta. \\\ \end{align}

Note that $Z_\theta$ is just a normalizing constant that forces the distribution to sum to $1$:

$$Z_\theta = \sum\limits_{i = 1}^n p_0(x_i) \exp\left(\sum_j \theta_j T_j(x_i)\right).$$

Thus, we can expand the partial derivative above as

\begin{align} \frac{\partial}{\partial \theta_j} LL &= \sum\limits_{i = 1}^n  T_j(x_i) - n \frac{1}{Z_\theta} \frac{\partial}{\partial \theta_j} Z_\theta \\\ &= \sum\limits_{i = 1}^n  T_j(x_i) - n \frac{1}{Z_\theta} \frac{\partial}{\partial \theta_j} \sum\limits_{i = 1}^n p_0(x_i) \exp\left(\sum_j \theta_j T_j(x_i)\right) \\\ &= \sum\limits_{i = 1}^n  T_j(x_i) - n \frac{1}{Z_\theta}  \sum\limits_{i = 1}^n p_0(x_i) T_j(x_i) \exp\left(\sum_j \theta_j T_j(x_i)\right) \\\ &= \sum\limits_{i = 1}^n  T_j(x_i) - n  \sum\limits_{i = 1}^n \frac{p_0(x_i) \exp\left(\sum_j \theta_j T_j(x_i)\right)}{Z_\theta} T_j(x_i)  \\\ &= \sum\limits_{i = 1}^n  T_j(x_i) - n  \sum\limits_{i = 1}^n p_\theta(x) T_j(x_i).  \\\ \end{align}

This expression must equal $0$ at the MLE, so we have

\begin{align} &\sum\limits_{i = 1}^n  T_j(x_i) - n  \sum\limits_{i = 1}^n p_\theta(x) T_j(x_i) = 0 \\\ \implies& \sum\limits_{i = 1}^n  T_j(x_i) = n  \sum\limits_{i = 1}^n p_\theta(x) T_j(x_i) \\\ \implies& \sum\limits_{i = 1}^n p_\theta(x) T_j(x_i) = \frac1n \sum\limits_{i = 1}^n  T_j(x_i) \\\ \implies& \mathbb{E}\_{p_\theta}(T_j(X)) = \mathbb{E}\_{\hat{p}}(T_j(X)). \\\ \end{align}

Thus, we can conclude that the MLE will satisfy the linear constraints required by the information projection problem, that these solutions will be equivalent, and that there exists a duality between these problems.

## Conclusion

It's interesting to see the same problem arise in different academic disciplines. Although it can be confusing to connect the dots at first, having multiple lenses through which to view these problems can be very useful. This phenomenon is common among the fields of statistics and information theory.

## References

- Prof. Aarti Singh's lecture notes on [this duality and information projections](https://www.cs.cmu.edu/~aarti/Class/10704_Fall16/lec8.pdf).
- Wikipedia articles for [information projections](https://www.wikiwand.com/en/Information_projection) and [maximum entropy distributions](https://www.wikiwand.com/en/Maximum_entropy_probability_distribution)
