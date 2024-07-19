---
layout: post
title: "Inference Gaussian Models: A Bayesian Approach"
author: "Binh Ho"
categories: Statistics
blurb: ""
img: ""
tags: []
<!-- image: -->
---

In this post, I will provide the Bayesian inference for one of the most 
fundamental and widely used models, the **normal** models. The Gaussian distribution is 
pivotal to a majority of statistical modeling and estimating its parameters
is a common task in Bayesian framework. I will derive results for three important 
cases: estimating the mean $\mu$ with known variance $\sigma^2$, estimating 
the variance $\sigma^2$ given a known $\mu$, and estimating both $\mu$ and $\sigma^2$.
The first two cases are single-parameter problem, while the last one is multiparameter one.
Furthermore, I will derive the likelihood, the *conjugate* prior, and the posterior of 
three cases. However, before we moving to the main part, I want to revise some helpful and special 
properties of the (multivariate) normal distribution. 

## Properties of Multivariate Normal Distribution (MVN)
The multivariate normal random variables has the following density function 

$$p(x; \mu, \Sigma) = \frac{1}{(2 \pi)^{n/2} |\Sigma|^{1/2}} \exp{\Big(-\frac{1}{2} (x - \mu)^{\intercal} \Sigma^{-1} (x - \mu) \Big)}$$
with the following log density expression

$$\log p(x; \mu, \Sigma) = -\frac{n}{2} \log(2 \pi) -\frac{1}{2} \log  (\det \mathbf{\Sigma})  - \frac{1}{2} (\mathbf{x}-\mathbf{\mu})^\intercal \mathbf{\Sigma}^{-1} (\mathbf{x}-\mathbf{\mu})$$

with $x, \mu \in \mathbb{R}^{n}, \Sigma \in \mathbb{R}^{n \times n}$, and $\Sigma$ must be *symmetric positive definite* (SPD).

Gaussians possess many interesting and remarkable properties that leverage modeling assumptions in a bunch of applications. Here I will list some of these that are most useful

**1. Product of Gaussian are Gaussian**

Let's me clarify this. The product of two Gaussian probability density functions (p.d.fs) will result in another Gaussian p.d.f. One example that arises frequently in statistics and machine learning is the *linear Gaussian model* which we have a Gaussian marginal distribution $p(z)$ and a Gaussian conditional distribution $p(y \lvert z)$ which has a linear function of $z$ as mean. These two density function jointly create a Gaussian distribution. We will state it more clearly as follows.

Assume that $p(z) = \mathcal{N}(z \lvert \mu, \Sigma)$ and $p(y \lvert z) = \mathcal{N}(y \lvert Wz + b, \Omega)$ where $y \in \mathbb{R}^{D}$, $z \in \mathbb{R}^{L}$, and $\Omega$ is a $D \times L$ matrix. Then, the joint distribution of $y$ and $z$ is a Gaussian distribution, i.e., $p(z, y) = p(y \lvert z) p(z) = \mathcal{N}(\tilde{\mu}, \tilde{\Sigma})$ with mean $\tilde{\mu}$ and covariance $\tilde{\Sigma}$ being expressed as

$$
\begin{align*}
\tilde{\mu} &= \begin{pmatrix}
                \mu \\
                W\mu + b
                \end{pmatrix} \\
\tilde{\Sigma} &= \begin{pmatrix}
\Sigma & W\Sigma^{\intercal} \\
W\Sigma & W \Sigma W^{\intercal} + \Omega
\end{pmatrix}
\end{align*}
$$

**Proof.**
We will working with the log of distribution to get rid of the cumbersome of exponential notation. Consider the log of the joint distribution 

$$
\begin{align*}
\ln p(y, z) & = \ln p(y \lvert z) + \ln p(z) \\
            & = -\frac{1}{2} (x - \mu)^{\intercal} \Sigma^{-1} (x - \mu) \\
            &   - \frac{1}{2}(y - Wz - b)^{\intercal} \Omega^{-1} (y - Wz - b) + \text{const} \quad (\ast)
\end{align*}
$$

Now we will expand the log of joint distribution and isolate the second order terms and the first order term to determine the covariance matrix and the mean vector:

$$
(\ast) = -\frac{1}{2} \bigl( z^{\intercal} \Sigma^{-1} z - 2 z^{\intercal} \Sigma^{-1} \mu + \mu^\intercal \Sigma^{-1} \mu + y^\intercal \Omega^{-1} y 
-\underbrace{y^\intercal \Omega^{-1} (Wz + b)}_{y^\intercal \Omega^{-1} Wz + y^\intercal \Omega^{-1} b} 
-\underbrace{(Wz + b)^\intercal \Omega^{-1} y}_{z^\intercal W^\intercal \Omega^{-1} y + b^\intercal \Omega^{-1} y} + \text{const} \bigr)
$$













# Inference normal distribution with known variance
There are three key components in traditional Bayesian settings: prior, likelihood, and posterior. Let's derive all of them.

**Likelihood**

$$
\begin{align}
p(D \mid \mu, \sigma^2)
&= \prod_{n=1}^{N} p(x_n \mid \mu, \sigma^2) \\
&\triangleq \prod_{n=1}^{N} \Bigg( \frac{1}{(2 \pi \sigma^2)^{1/2}} \exp \Big\lbrace -\frac{1}{2 \sigma^2} (x_n - \mu)^2 \Big\rbrace \Bigg) \\
&= \frac{1}{(2 \pi \sigma^2)^{N/2}} \exp \Big\lbrace -\frac{1}{2 \sigma^2} \sum_{n=1}^{N} (x_n - \mu)^2 \Big\rbrace
\end{align}
$$

**Prior**

We can show that the conjugate prior on $\mu$ is another Gaussian distribution 

$$
p(\mu) = \mathcal{N}(\mu \lvert m, \tau^2) = \frac{1}{(2 \pi \tau^2)^{1/2}} \exp{ \Big\lbrace-\frac{1}{2 \tau^2} (\mu - m)^2 \Big\rbrace }
$$ 

**Posterior** 

By applying the Bayes'rule for Gaussians, we can obtain the Gaussian posterior distribution. Note that here we will only focus on the functional form of our parameters of interest and discard all constants that do not depend on the parameters

$$
\begin{align*}
p(\mu \lvert D) & \propto p(D \lvert \mu, \sigma^2) p(\mu) \\
&\triangleq \Biggl( \frac{1}{(2 \pi \sigma^2)^{N/2}} \exp \Big\lbrace -\frac{1}{2 \sigma^2} \sum_{n=1}^{N} (x_n - \mu)^2 \Big\rbrace \Biggr) 
\Biggl( \frac{1}{(2 \pi \tau^2)^{1/2}} \exp{ \Big\lbrace -\frac{1}{2 \tau^2} (\mu - m)^2 \Big\rbrace} \Biggr) \\
&= \color{WildStrawberry}{\frac{1}{( 2 \pi \sigma^2 )^{N/2} (2 \pi \tau^2)^{1/2})} } \exp{ \Bigg\lbrace -\frac{1}{2 \sigma^2} \sum_{n=1}^{N} (x_n - \mu)^2 - \frac{1}{2 \tau^2}(\mu - m)^2 \Bigg \rbrace} \\
& \propto \exp{ \Bigg\lbrace -\frac{1}{2 \sigma^2} \sum_{n=1}^{N} (x_n - \mu)^2 - \frac{1}{2 \tau^2}(\mu - m)^2 \Bigg \rbrace} \\
&= \exp{ \Bigg\lbrace -\frac{1}{2 \sigma^2} \Big( \sum_{n=1}^{N} x_{n}^2 + \mu^2 - 2x_{n}\mu \Big) - \frac{1}{2 \tau^2} \Big( \mu^2 + m^2 - 2\mu m \Big) \Bigg\rbrace } \\
&= \exp{ \Bigg\lbrace -\frac{1}{2 \sigma^2} \Big( \color{WildStrawberry}{ \sum_{n=1}^{N} x_{n}^2} + N \mu^2 - 2 \mu \sum_{n=1}^{N} x_{n} \Big) - \frac{1}{2 \tau^2} \Big( \mu^2 + \color{WildStrawberry}{m^2} - 2\mu m \Big) \Bigg\rbrace } \quad (\ast) \\ 
&\propto \exp{ \Bigg\lbrace -\frac{1}{2 \sigma^2} \Bigg( N \mu^2 - 2\mu \sum_{n=1}^{N} x_{n} \Bigg) - \frac{1}{2 \tau^2} \Big( \mu^2 - 2 \mu m \Big) \Bigg\rbrace } \\
& = \exp{ \Bigg\lbrace -\frac{1}{2} \Bigg( \frac{N \mu^2}{\sigma^2} - \frac{2 \mu N \bar{x}}{\sigma^2} + \frac{\mu^2}{\tau^2} - \frac{2 \mu m}{\tau^2} \Bigg) \Bigg\rbrace }
\end{align*}
$$



