---
layout: post
title: "$M$-estimation"
author: "Binh Ho"
categories: Statistics
blurb: "This post briefly covers a broad class of statistical estimators: M-estimators. We'll review the basic definition, some well-known special cases, and some of its asymptotic properties."
img: ""
tags: []
<!-- image: -->
---



This post briefly covers a broad class of statistical estimators: M-estimators. We'll review the basic definition, some well-known special cases, and some of its asymptotic properties.

## Introduction

"M"-estimators are named as such because they are defined as the **m**inimization (or **m**aximization) of some function. Specifically, $\hat{\theta}$ is an M-estimator if

$$\hat{\theta} = \text{arg}\min_\theta \sum\limits_{i = 1}^n \rho(X_i, \theta)$$

where $\rho$ is a function that can be chosen, and whose choice leads to different types of estimators.

## Special cases

### Maximum likelihood estimation

Perhaps the most well-known type of M-estimation is maximum likelihood estimation (MLE). Recall that MLE seeks the parameter $\theta$ that maximizes the likelihood of the data:

$$\hat{\theta} = \text{arg}\max_\theta \prod\limits_{i = 1}^n f(X_i, \theta)$$

where $f$ is the pdf of the data $X_1, \dots, X_n$. This can be written as an equivalent minimization problem:

$$\hat{\theta} = \text{arg}\min_\theta \prod\limits_{i = 1}^n - f(X_i, \theta).$$

Taking the log doesn't change the maximizer since $\log$ is a monotonic function:

$$\hat{\theta} = \text{arg}\min_\theta \sum\limits_{i = 1}^n - \log f(X_i, \theta).$$

And now we notice that this is the form of the generic M-estimator, where $\rho(X_i, \theta) = -\log f(X_i, \theta)$.

### Method of moments estimation

Method of moments is another popular generic estimation technique. Its technique is very simple: set the first $k$ sample moments equal to the respective $k$ population moments, and solve for the unknowns.

This setup can be couched in the language of M-estimation as follows. Suppose $g(X_i)$ is some function where $\mathbb{E}_\theta[g(X_i)] = \mu(\theta)$. 

The following choice of $\rho$ is equivalent to method of moments estimation:

$$\rho(X_i, \theta) = (g(X_i) - \mu(\theta))^2.$$

## Asymptotic properties

Below are a couple of properties of M-estimators that hold true as $n \to \infty$ (stated without proof here).

### Consistency

Under the proper regularity conditions, M-estimators are consistent. That is, they converge in probability, or stated mathematically,

$$\lim_{n \to \infty} \mathbb{P}[|\hat{\theta}_n - \theta| > \epsilon] = 0.$$

Intuitively, the main regularity condition required is that the expression $\sum\limits_{i = 1}^n \rho(X_i, \theta)$ behaves nicely as $n \to \infty$. By "nicely" here, we mean that it converges uniformly to its expectation, $\mathbb{E}\left[ \sum\limits_{i = 1}^n \rho(X_i, \theta) \right]$. This is analogous to the Law of Large Numbers.

### Asymptotic normality

Under certain regularity conditions, M-estimators are also asymptotically normal. If $\theta_0$ is the true parameter of the data generating process, and $\hat{\theta}$ is the M-estimator, we have that

$$\sqrt{n}(\hat{\theta}_n - \theta) \to_d \mathcal{N}(0, H_0^{-1}\Sigma_0 H_0^{-1})$$

where $\Sigma_0 = \mathbb{V}\left[ \frac{\partial}{\partial \theta} \rho(X_i, \theta) \right]$ and $H_0 = \mathbb{E}\left[ \frac{\partial}{\partial \theta \partial \theta^\top} \rho(X_i, \theta) \right]$. This is often called the "sandwich estimator" for the variance, where $H_0^{-1}$ is the "bread" and $\Sigma_0$ is the "meat".

An interesting note here is that, in the maximum likelihood estimation case, both $\Sigma_0$ and $H_0$ are equal to the Fisher information $\mathcal{I}_0$ by definition. Thus, the asymptotic variance collapses to

$$H_0^{-1}\Sigma_0 H_0^{-1} = \mathcal{I}_0^{-1} \mathcal{I}_0 \mathcal{I}_0^{-1} = \mathcal{I}_0^{-1}.$$

## Robustness

M-estimation was developed in the context of robust estimation. In general, an estimator is robust if it retains its nice properties (consistency, efficiency) even when some of the modeling assumptions are not true.

M-estimation in particular doesn't require the statistician to fully specify the distribution of the data generating process. Rather, only the objective function $\rho$ must be specified.

Furthermore, if the distribution is specified incorrectly, then $H_0 \neq \Sigma_0$, but the variance estimator is still consistent.

## Conclusion

M-estimation is a broad umbrella of techniques that share a common underlying maximization or minimization. The advantage of grouping estimators into broad classes and analyzing them more generically is that all of the results and proofs directly allow us to gain more insight into many different types of estimators "for free". As we saw here, M-estimators are consistent and asymptotically normal, and allow for robust estimation.


## References

- In-person lectures and lecture slides Prof. Matias Cattaneo's ORF524 class at Princeton.
- Stefanski, Leonard A., and Dennis D. Boos. "The calculus of M-estimation." The American Statistician 56.1 (2002): 29-38.


