---
layout: post
title: "Control variates"
author: "Binh Ho"
categories: Statistics
blurb: "Control variates are a class of methods for reducing the variance of a generic Monte Carlo estimator."
img: ""
tags: []
<!-- image: -->
---

Control variates are a class of methods for reducing the variance of a generic Monte Carlo estimator.

## Introduction

Suppose we'd like to estimate the expectation of some function $f(x)$,

$$A = \mathbb{E}[f(x)].$$

A naive Monte Carlo approximation would proceed to estimate this expectation as a simple average over some number of samples:

$$A \approx \frac{1}{n} \sum\limits_{i=1}^n f(x_i) =: \hat{S}.$$

The variance of this approximation will be 

$$\mathbb{V}[\hat{S}] = \frac{\sigma^2_f}{n}$$

where $\sigma^2_f$ is the variance of $f(x)$.

## Control variates

The idea behind control variates is to manipulate the approximation so that its expectation is still $A$, but it has lower variance. In particular, consider another function $g(x)$, where $\mathbb{E}[g(x)] = B$ (assuming we know this value upfront for now), and consider the random variable

$$D = f(x) - c(g(x) - B)$$

where $c$ is a scalar coefficient that we can choose for a given problem.

Notice that the expectation of $D$ is still $A$:

\begin{align} \mathbb{E}[D] &= \mathbb{E}[f(x) - c(g(x) - B)] \\\ &= \mathbb{E}[f(x)] - c\mathbb{E}[g(x)] + cB \\\ &= \mathbb{E}[f(x)] - cB + cB \\\ &= \mathbb{E}[f(x)] \\\ \end{align}

where we have used the fact that $\mathbb{E}[g(x)] = B$. So in effect, all we have done is added and subtracted the same term, to arrive back at $\mathbb{E}[f(x)]$. This might seem silly at first, but it actually helps us!

Recall that the variance of our original approximation was $\sigma^2_f$. The variance of our new random variable $D$ is

\begin{align} \mathbb{V}[D] &= \mathbb{V}[f(x) - c g(x)  + c B] \\\ &= \mathbb{V}[f(x) - c g(x)] \\\ &= \mathbb{V}[f(x)] - 2c \text{cov}(f(x), g(x)) + c^2 \mathbb{V}[g(x)] \\\ &= \sigma^2_f - 2c \text{cov}(f(x), g(x)) + c^2 \sigma^2_g \\\ \end{align}

where we have denoted $\sigma^2_g = \mathbb{V}[g(x)]$. Recall that $c$ is just a coefficient that we can set ourselves. Since we want to minimize variance, we can easily solve for the optimal value of $c$:

\begin{align} &\frac{d}{d c} \left[\sigma^2_f - 2c \text{cov}(f(x), g(x)) + c^2 \sigma^2_g\right] = 0 \\\ \implies& -2 \text{cov}(f(x), g(x)) + 2c \sigma^2_g = 0 \\\ \implies& c^* = \frac{\text{cov}(f(x), g(x))}{\sigma^2_g} \\\ \end{align}

Plugging in $c^*$, we can find the variance of $D$ at this optimal value:

\begin{align} \mathbb{V}[D] &= \sigma^2\_f - 2 \frac{\text{cov}(f(x), g(x))}{\sigma^2_g} \text{cov}(f(x), g(x)) + \left(\frac{\text{cov}(f(x), g(x))}{\sigma^2_g}\right)^2 \sigma^2_g \end{align}

Simplifying, and denoting the correlation between $f$ and $g$ as $\rho_{fg} = \frac{\text{cov}(f(x), g(x))}{\sigma_f\sigma_g}$, we have

$$\mathbb{V}[D] = \sigma^2_f(1 - \rho_{fg}^2).$$

So the variance is reduced by a multiplicative factor of $1-\rho_{fg}^2$, implying that we'll see a greater reduction in variance if we choose a function $g$ that is more correlated with $f$.

Now, instead of estimating $\hat{S}$ as a sample mean, let's compute an approximation of $\mathbb{E}[D]$:

$$\mathbb{E}[D] \approx \frac{1}{n} \sum\limits_{i=1}^n \left[f(x_i) - c g(x_i)\right] + B,$$

which will reduce the variance of our naive estimator.

## References

- Professor Jonathan Goodman's [notes on variance reduction methods](https://www.math.nyu.edu/faculty/goodman/teaching/MonteCarlo2005/notes/VarianceReduction.pdf).
- Tucker, George, et al. "Rebar: Low-variance, unbiased gradient estimates for discrete latent variable models." Advances in Neural Information Processing Systems. 2017.
- Paisley, John, David Blei, and Michael Jordan. "Variational Bayesian inference with stochastic search." arXiv preprint arXiv:1206.6430 (2012).

