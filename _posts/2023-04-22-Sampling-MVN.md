---
layout: post
title: "Multivariate Gaussian Sampling"
author: "Binh Ho"
categories: Statistics
blurb: ""
img: ""
tags: []
<!-- image: -->
---

## Introduction
It 's quite common that the univariate standard normal random variables with zero mean and unit variance $Z \sim \mathcal{N}(0, 1)$ is easy to simulate. Furthermore, most programming languages offer extremely efficient vectorized/parallelized algorithms for such jobs. On the other hand, it is rather burdensome and challenging to draw samples from *multivariate* random variables directly. Thus, inspired by many built-in functions in programming language, the approach that I cover below will utilize the transformation from standard normal random variables to produce sampling from the desired multivariate Gaussian random variable. It is nothworthy that the standard normal random variables are still implemented via the inverse transformation from uniform distribution (check this post about [inverse transform sampling](https://callmequant.github.io/study/Inverse-Transform-Sampling/) to have a basic understanding of this simple, yet efficient method of sampling).

## Paramterization of Univariate Gaussian Random Variable
In the context of univariate normal random variable $x$, two approaches are often used in practice: the variance parameterization $x \sim \mathcal{N}(\mu, \sigma)$ and the precision parameterization $x \sim \mathcal{N}(\mu, \tau)$, where $\sigma$ in our post is referred to as the variance (not the standard deviation) and $\tau$ is the precision (inverse of the variance or $\tau = 1 / \sigma$). The central idea is that rather than perform sampling from $x$ directly, we could instead sample from $z \sim \mathcal{N}(0, 1)$ which is the standard normal random variable and transform samples of $z$ into samples of $x$ via some operations. The process could be done easily by simply multiplying $z$ with square root of variance and add up a constant $\mu$: $x = \mu + \sqrt{\sigma} z$ for the variance parameterization and $x = \mu + 1 / \sqrt{\tau} z$ for the precision parameterization. Both of these are related to the “non-centered parameterization” of a Gaussian random variable. 

You could justify this parameterization by thinking as follows: you have a standard normal random variable $z$, which is zero mean and unit variance. Then, you first scale the variance of $z$ by a constant that match the variance of $x$ ($\sigma$) and correct the "shape" of the distribution of $x$ by adding up a constant $\mu$. This also the same for the case of precision parameterization. 

Before we move on, note that the same idea holds for the multivariate case; however, in that case we have a matrix (covariance matrix). Thus, careful measure should be undertaken to tackle the issue of square root or inverse square root of a matrix. 

## Cholesky decomposition
Given the covariance matrix $\boldsymbol{\Sigma}$ that is symmetric positive-definite, the matrix *square root* is defined as a matrix $\boldsymbol{\Sigma}^{\frac{1}{2}}$ satisfying

$$\boldsymbol{\Sigma} = \boldsymbol{\Sigma}^{\frac{1}{2}} \left(\boldsymbol{\Sigma}^{\frac{1}{2}}\right)^\intercal  $$

It turns out there are multiple matrix square roots and any of them can be used for sampling from the multivariate Gaussian. The most common and often efficient method is given by the Cholesky decompostion

A symmetric positive definite matrix $\boldsymbol{\Sigma}$ could be represented as 

$$ \boldsymbol{\Sigma} = \mathbf{L} \mathbf{L}^\intercal, $$

where $\mathbf{L}$ is a lower-triangular matrix and  $\mathbf{L}$ is called *Cholesky decomposition* of $\Sigma$. We can use the upper triangular matrix $\mathbf{U}$ for the decomposition as follows: 

$$\boldsymbol{\Sigma} = \mathbf{U^\intercal} \mathbf{U} $$.

Just a quick note, nearly every programming language or
linear algebra library has an implementation of the Cholesky decomposition.

### Evaluating the Multivariate Normal Density
The log-density of the $d$ dimensional multivariate normal is 

$$
\log p(\mathbf{x}) = -\frac{d}{2} \log(2 \pi) -\frac{1}{2} \log | \det \mathbf{\Sigma} | - \frac{1}{2} (\mathbf{x}-\mathbf{\mu})^T \mathbf{\Sigma}^{-1} (\mathbf{x}-\mathbf{\mu}).
$$

There exists two difficult and tedious terms to evaluate:

$$\log | \det \mathbf{\Sigma} | $$
and

$$- \frac{1}{2} (\mathbf{x}-\mathbf{\mu})^T \mathbf{\Sigma}^{-1} (\mathbf{x}-\mathbf{\mu}). $$
Both difficult terms can be evaluated efficiently and in a numerically stable manner using the Cholesky decomposition of $\boldsymbol{\Sigma}$.

### Determinant Evaluation using the Cholesky Decomposition
Assume that $\boldsymbol{\Sigma} = \mathbf{L} \mathbf{L}^\intercal$ with 

$$\mathbf{L} = 
\begin{bmatrix} 
l_{11} & 0 & 0 & \dots & 0 \\  
l_{21} & l_{22} & 0 & \dots & 0 \\  
l_{31} & l_{32} & l_{33} & \dots & 0 \\  
\vdots & \vdots & \vdots & \ddots & 0 \\  
l_{d1} & l_{d2} & l_{d3} & \dots & l_{dd} 
\end{bmatrix}.$$

Then using the elementary of determinant and logarithm, we arrive at: 

$$
\begin{align*}
  \log \det \boldsymbol{\Sigma} &= \log\left( \det ( \mathbf{L} \mathbf{L}^T) \right)
    = \log\left( \det \mathbf{L} \det \mathbf{L}^T \right) \\
    &= \log\left( (\det \mathbf{L})^2 \right)
    = 2 \log\left( \det \mathbf{L} \right) \\
    &= 2 \log\left( \prod_{i=1}^d l_{ii} \right)
    = 2 \sum_{i=1}^d \log\left( l_{ii} \right).
\end{align*}
$$

###  Quadratic form evaluation with Cholesky
We can simplify the quadratic form as

$$
\begin{align*}
    (\mathbf{x}-\boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x}-\boldsymbol{\mu})
    &= (\mathbf{x}-\boldsymbol{\mu})^T (\mathbf{L} \mathbf{L}^T)^{-1} (\mathbf{x}-\boldsymbol{\mu}) \\
    &= (\mathbf{x}-\boldsymbol{\mu})^T \mathbf{L}^{-T} \mathbf{L}^{-1} (\mathbf{x}-\boldsymbol{\mu}) \\
    &= (\mathbf{L}^{-1}(\mathbf{x}-\boldsymbol{\mu}))^T \mathbf{L}^{-1}(\mathbf{x}-\boldsymbol{\mu}) \\
	&= \mathbf{z}^T \mathbf{z} = \sum_{i=1}^d z_i^2,
\end{align*}
$$

where $\mathbf{z} = (z_1, z_2, \dots, z_d)^\intercal = \mathbf{L}^{-1} (\mathbf{x} - \mu)$.
To compute $\mathbf{z} = \mathbf{L}^{-1} (\mathbf{x} - \mu)$, we solve a linear system of equations

$$ \mathbf{Lz} = \mathbf{x} - \mu $$
 
## Sampling from the Multivariate Normal

Generalizing the univariate standard normal above, let us now introduce the vector $x$ with elements $x_i \sim \mathcal{N}(0,1)$. Then we have the following theorem

**Theorem** *Assuming* $\mathbf{x} \sim \mathcal{N}(0, \mathbf{I}_{d}),$ *we can obtain* $\mathbf{y} \sim \mathcal{N}(\mu, \boldsymbol{\Sigma})$ *using the tranformation* 

$$ \mathbf{y} = \mathbf{L} \mathbf{x} + \mu,$$

where $\mathbf{L}$ *is the Cholesky decomposition* of $\boldsymbol{\Sigma}: \boldsymbol{\Sigma} = \mathbf{L} \mathbf{L}^\intercal.$

*Proof*. $\mathbf{y}$ remains multivariate normal after an affine transformation because the (unnormalized) multivariate normal density after transformation $\mathbf{y} = \mathbf{Ax} + \mathbf{b}$ with invertible $\mathbf{A}$ still has the form of a multivariate normal over $\mathbf{y}$:

$$
\begin{align*}
p(\mathbf{y}) &= \frac{1}{Z} \exp(- \frac{1}{2} (\mathbf{y}-\boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{y}-\boldsymbol{\mu}))\\
       &= \frac{1}{Z'} \exp(- \frac{1}{2} (\mathbf{A} \mathbf{x} + \mathbf{b} -\boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{A} \mathbf{x} + \mathbf{b}-\boldsymbol{\mu}))\\
       &= \frac{1}{Z'} \exp(- \frac{1}{2} (\mathbf{x} - \mathbf{A}^{-1}(\boldsymbol{\mu} - \mathbf{b}))^T (\mathbf{A}^{-1} \boldsymbol{\Sigma} (\mathbf{A}^{-1})^{T})^{-1} (\mathbf{x} - \mathbf{A}^{-1}(\boldsymbol{\mu} - \mathbf{b}))).
\end{align*}
$$

The mean and covariance of $\mathbf{y}$ may be evaluated as 

$$
\begin{aligned}
\mathbb{E}[\mathbf{y}] & = \mathbf{L} \mathbb{E}[\mathbf{x}] + \mathbf{\mu} = \mathbf{\mu} \\
\text{Cov}[\mathbf{y}] = \mathbb{E}[ (\mathbf{y} - \mathbb{E}[\mathbf{y}]) (\mathbf{y} - \mathbb{E}[\mathbf{y}])^T ]
& = \mathbb{E}[ (\mathbf{L} \mathbf{x})(\mathbf{L} \mathbf{x})^T]
= \mathbb{E}[ \mathbf{L} \mathbf{x} \mathbf{x}^T \mathbf{L}^T ]
= \mathbf{L} \mathbb{E}[ \mathbf{x} \mathbf{x}^T ] \mathbf{L}^T
= \mathbf{L} \mathbf{I}_d \mathbf{L}^T = \mathbf{\Sigma}.
\end{aligned}
$$

***Note***: If $\boldsymbol{\Sigma}$ is non-positive due to numerical issues (i.e., negative eigenvalues), the use 
$\boldsymbol{\Sigma} + \epsilon I_{d}$ with $\epsilon > |\lambda_{\min}|$

Then, covariane parameterization would be 

$$\mathbf{y} = \mu + \mathbf{L_{\Sigma}} \mathbf{x},$$

and the precision parameterization is

$$ \mathbf{y} = \mu + \mathbf{U}_{\Lambda}^{-1} \mathbf{x}.$$ 

where $\mathbf{\Lambda} = \mathbf{\Sigma}^{-1}$

You might be wondering why I didn’t just write $y=\mu+\mathbf{L}_{\Lambda} \mathbf{x}$ for the precision parameterization where  refers to the lower Cholesky factor of the inverse of $\mathbf{\Sigma}$ (which would have been correct and not required the special note about the upper/lower Cholesky forms). It turns out that inverting a triangular matrix (e.g., the Cholesky form) is more numerically stable and efficient than inverting the original symmetric positive-definite matrix. 
