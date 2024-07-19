---
layout: post
title: "Newton's method and Fisher scoring for fitting GLMs"
author: "Andy Jones"
categories: journal
blurb: "Generalized linear models are flexible tools for modeling various response disributions. This post covers one common way of fitting them."
img: ""
tags: []
<!-- image: -->
---

Generalized linear models are flexible tools for modeling various response disributions. This post covers one common way of fitting them.

## Introduction

Suppose we have a statistical model with log-likelihood $\ell(\theta)$, where $\theta$ is the parameter (or parameter vector) of interest. In maximum likelihood estimation, we seek to find the value of $\theta$ that maximizes the log-likelihood:

$$\ell_n(\theta) = \sum\limits_{i=1}^n \log f(x_i; \theta)$$

$$\hat{\theta}_{\text{MLE}} = \text{arg}\max_{\theta} \ell_n(\theta)$$

There are many ways to solve this optimization problem.

## Newton's Method

One simple numerical method for finding the maximizer is called Newton's Method. This method essentially uses the local curvature of the log-likelihood function to iteratively find a maximum.

The derivation of Newton's method only requires a simple Taylor expansion. Below, we focus on the univeriate case (i.e., $\theta \in \mathbb{R}$), but all results can be easily extended to the multivariate case. 

Recall that the first derivative of the log-likelihood function, $\ell'(\theta)$, is called the score function. For any given initial guess of the value of $\theta$, call it $\theta_0$, we can perform a second-order Taylor expansion around this value:

$$\ell'(\theta) \approx \ell'(\theta_0) + \ell''(\theta_0) (\theta - \theta_0).$$

At the value of $\theta$ that maximizes the loglikelihood, $\theta^\*$, we know that the derivative of the log-likelihood is zero, $\ell'(\theta^\*) = 0$ (this is usually true under some mild regularity conditions, such as the maximizer not being at the edge of the support). Thus, if we plug in $\theta = \theta^\*$ to our expansion, we have

\begin{align} &\ell'(\theta^\*) = 0 \approx \ell'(\theta_0) + \ell''(\theta_0) (\theta^\* - \theta_0) \\\ \implies & \theta^\* \approx \theta_0 - \frac{\ell'(\theta_0)}{\ell''(\theta_0)}. \end{align}

This is known as Newton's method. Specifically, the algorithm proceeds as follows:

1. Initialize $\theta_0$ to a random value.
2. Until converged, repeat: update $\theta_t = \theta_{t-1} - \frac{\ell'(\theta_0)}{\ell''(\theta_0)}$
    
As we can see, Newton's Method is essentially fitting a parabola to the current location of the log-likelihood function, then taking the minmium of that quadratic to be the next value of $\theta$.

One downside of this method is that it assumes that $\ell''(\theta)$ is invertible, which may not always be the case. In the next section, we'll see a method that remedies this issue.

## Fisher scoring

Fisher scoring is has the same form as Newton's Method, but instead of the observed second derivative, it uses the expectation of this second derivative, a quantity that is also known as the Fisher Information. The update then looks like:

$$\theta_t = \theta_{t-1} - \frac{\ell'(\theta_0)}{\mathbb{E}[\ell''(\theta_0)]}.$$

The benefit of this method is that $\mathbb{E}[\ell''(\theta_0)]$ is guaranteed to be positive (or a positive definite matrix in the multivariate case).

## Relating Newton's method to Fisher scoring

A key insight is that Newton's Method and the Fisher Scoring method are identical when the data come from a distribution in canonical exponential form. Recall that $f$ is in the exponential family form if it has the form

$$f(x) = \exp\left\{ \frac{\eta(\theta(x))x - b(\theta(x))}{a(\phi)} + c(x, \phi) \right\}.$$

The "canonical" form occurs when $\eta(\theta) = \theta$, and so

$$f(x) = \exp\left\{ \frac{\theta(x)x - b(\theta(x))}{a(\phi)} + c(x, \phi) \right\}.$$

The $\log$ density is

$$\log f(x) = \frac{\theta(x)x - b(\theta(x))}{a(\phi)} + c(x, \phi)$$

and the first and second derivatives with respect to $\theta$ are then

\begin{align} \frac{\partial \log f}{\partial \theta} &= \frac{x - b'(\theta(x))}{a(\phi)} \\\ \frac{\partial^2 \log f}{\partial \theta^2} &= \frac{-b''(\theta(x))}{a(\phi)}. \\\ \end{align}

In canonical exponential distributions, the second derivative of $b(\theta)$ wrt $\theta$ is also the Fisher information. This can be seen by inspecting the definition of the Fisher information: it is defined as $\mathcal{I} = -\mathbb{E}[\nabla^2 \log f(x)]$. We saw above that this evaluates to $\mathbb{E}[- A''(\theta)]$. The expression inside the expectation is constant with respect to $f(x)$, so we have $\mathcal{I} = - b''(\theta)$, and this implies that the observed second derivative is identical to the expected second derivative. Thus, in the case of canonical exponential forms, Newton's Method and Fisher Scoring are the same.


## Generalized linear models

An important application in which these methods play a big role is in fitting generalized linear models (GLMs). As the name suggests, GLMs are a generalization of traditional linear models where the responses are allowed to come from different types of distributions. GLMs are typically written as

$$g(\mu(x)) = X^\top \beta$$

where $\mu(x) = \mathbb{E}[Y \| X = x]$ is called the "regression function", and $g$ is called the "link function". In GLMs, we assume the conditional density $Y \| X = x$ belongs to the exponential family:

$$f(y|x) = \exp\left\{ \frac{\theta(x)y + b(\theta(x))}{a(\phi)} + c(y, \phi) \right\}.$$

If we have a data sample $\{(X_i, Y_i)\}$ for $i = 1, \dots, n$, we can estimate $\theta$ using maximum likelihood estimation (MLE). Notice that $\mu_i = b'(\theta_i)$, which implies that $\theta_i = (b')^{-1}(\mu_i)$. Furthermore, since $\mu_i = g^{-1}(X_i^\top \beta$, we have that $\theta_i = (b' \circ g)^{-1}(X_i^\top \beta)$. For ease of notation, we can define this as a new function $h$: $h(X_i^\top \beta) \equiv (b' \circ g)^{-1}(X_i^\top \beta)$.

Then, writing out the log-likelihood function, we have

$$\ell_n(\beta, \phi) = \sum\limits_{i=1}^n \left[\frac{h(X_i^\top \beta)y_i - b(h(X_i^\top \beta))}{a_i(\phi)} + c(y_i, \phi)\right].$$

It's quite common to take $a(\phi) = \frac{\phi}{w_i}$, which simplifies our expression to

$$\ell_n(\beta, \phi) = \sum\limits_{i=1}^n \left[\frac{w_i h(X_i^\top \beta)y_i - b(h(X_i^\top \beta))}{\phi} + c(y_i, \phi)\right].$$

Ignoring terms that don't depend on $\beta$, and noticing that $\phi$ is a positive constant, the expression we'd like to maximize is

$$\ell_n(\beta, \phi) = \sum\limits_{i=1}^n w_i\left[h(X_i^\top \beta)y_i - b(h(X_i^\top \beta))\right].$$

After some expert-mode chain rule, we obtain that the first derivative with respect to $\beta$ is

$$\ell'(\beta) = \mathbf{X}^\top \mathbf{W} \mathbf{G} (\mathbf{Y} - \boldsymbol{\mu})$$

where 

$\mathbf{W}$ is a diagonal matrix with $w_i (b''(\theta)g'(\mu_i)^2)^{-1}$ as the $i$'th diagonal, and $\mathbf{G}$ is a diagonal matrix with $g'(\mu_i)$ as the $i$'th diagonal.

The expectation of the second derivative is then

$$\mathbb{E}[\ell''(\beta)] = -\mathbf{X}^\top \mathbf{W} \mathbf{X}$$

Plugging these results into the Fisher scoring algorithm, we have that the update at time $t+1$ will be 

\begin{align} \beta_{t+1} &= \beta_t + (\mathbf{X}^\top \mathbf{W} \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{W} \mathbf{G} (\mathbf{Y} - \mu) \\\ &= (\mathbf{X}^\top \mathbf{W} \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{W} (\mathbf{G} (\mathbf{Y} - \mu) + \mathbf{X}\beta_t) \\\ \end{align}

Notice that this is similar to the estimating equation for weighted least squares

$$\hat{\beta} = (\mathbf{X}^\top \mathbf{W} \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{W}\mathbf{Y}.$$

In our case, we essentially have $\mathbf{Y} = \mathbf{G} (\mathbf{Y} - \mu) + \mathbf{X}\beta_t.$

One can interpret this as the estimated response $\mathbf{X}\beta_t$, plus the current residuals, $\mathbf{G} (\mathbf{Y} - \mu)$. Thus, we're basically using a version of the response that has been "corrected" for the errors in the current estimate for $\beta$. 

It turns out that this algorithm is equivalent to iteratively reweighted least squares (IRLS) for maximum likelihood.

## References 

- Fan, J., Li, R., Zhang, C.-H., and Zou, H. (2020). Statistical Foundations of Data Science.
CRC Press, forthcoming.
- Prof. Steffen Lauritzen's [lecture notes](http://www.stats.ox.ac.uk/~steffen/teaching/bs2HT9/scoring.pdf).
- Hua Zhou's [blog post](https://hua-zhou.github.io/index.html).
