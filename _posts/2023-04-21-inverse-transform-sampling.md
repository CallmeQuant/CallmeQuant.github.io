---
layout: post
title: "Inverse Transform Sampling"
author: "Binh Ho"
categories: Statistics
blurb: "Inverse transform sampling is a method for generating random numbers from any probability distribution by using its inverse cumulative distribution $F^{-1}(x)$."
img: ""
tags: []
<!-- image: -->
---

# Introduction

Inverse transform sampling is a method for generating random numbers from any probability distribution by using its inverse cumulative distribution $F^{-1}(x)$.
Recall that the cumulative distribution for a random variable $X$ is $F_X(x) = P(X \leq x)$. In what follows, we assume that our computer can, on demand, 
generate independent realizations of a random variable $U$ uniformly distributed on $[0,1]$. 

# Algorithm
## Continuous Distributions
Assume we want to generate a random variable $X$ with cumulative distribution function (CDF) $F_X$. The inverse transform sampling algorithm is simple:  
1. Generate $U \sim \text{Unif}(0,1)$  
2. Let $X = F_X^{-1}(U)$.

Then, $X$ will follow the distribution governed by the CDF $F_X$, which was our desired result.

Note that this algorithm works in general but is not always practical. For example, inverting $F_X$ is easy if $X$ is an exponential random variable, but its harder if $X$ is Normal random variable.


## Discrete Distributions
Now we will consider the discrete version of the inverse transform method. Assume that $X$ is a discrete random variable such that $P(X = x_i) = p_i$. The algorithm proceeds as follows:  
1. Generate $U \sim \text{Unif}(0,1)$  
2. Determine the index $k$ such that $\sum_{j=1}^{k-1} p_j \leq U < \sum_{j=1}^k p_j$, and return $X = x_k$.

Notice that the second step requires a *search*. 

## Why it works ?
The question pops up is that why this method works. Well!, to see this, we will 
state the followin theorem 

**Theorem**: Let $U$ be a continuous random variable having a standard uniform distribution. That is, $U \sim \operatorname{U}(0,1)$. Then, the random variable

$$ X = F_X^{-1}(U) $$

has a probability distribution characterized by the invertible cumulative distribution function $F_X(x)$.

***Proof***: The cumulative distribution function of the transformation $X = F_X^{-1}(U)$ can be derived as

$$ \begin{split} &\hphantom{=}  \mathrm{Pr}(X \leq x) \ &= \mathrm{Pr}(F_X^{-1}(U) \leq x) \ &= \mathrm{Pr}(U \leq F_X(x)) \ &= F_X(x) , \end{split} $$

because the cumulative distribution function of the standard uniform distribution $\mathcal{U}(0,1)$ is

$$  U \sim \mathcal{U}(0,1) \quad \Rightarrow \quad F_U(u) = \mathrm{Pr}(U \leq u) = u. $$



Computationally, this method involves computing the quantile function of the distribution — in other words, computing the cumulative distribution function (CDF) of the distribution (which maps a number in the domain to a probability between 0 and 1) and then inverting that function many times. This is the source of the term “inverse” or “inversion” in most of the names for this method. Note that for a discrete distribution, computing the CDF is not in general too difficult: we simply add up the individual probabilities for the various points of the distribution. For a continuous distribution, however, we need to integrate the probability density function (PDF) of the distribution, which is impossible to do analytically for most distributions (including the normal distribution). As a result, this method may be computationally inefficient for many distributions that does not have intractable CDF  and other methods are preferred; however, it is a useful method for building more generally applicable samplers such as those based on rejection sampling.

For the normal distribution, the lack of an analytical expression for the corresponding quantile function means that other methods (e.g. the Box–Muller transform) may be  computationally favorable. It is often the case that, even for simple distributions, the inverse transform sampling method can be improved on.
