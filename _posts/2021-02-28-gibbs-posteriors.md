---
layout: post
title: "Gibbs posteriors"
blurb: "Bayesian posterior inference requires the analyst to specify a full probabilistic model of the data generating process. Gibbs posteriors are a broader family of distributions that are intended to relax this requirement and to allow arbitrary loss functions."
img: "/assets/klqp_univariate_densities.png"
author: "Binh Ho"
categories: Statistics
tags: []
<!-- image: -->
---

Bayesian posterior inference requires the analyst to specify a full probabilistic model of the data generating process. Gibbs posteriors are a broader family of distributions that are intended to relax this requirement and to allow arbitrary loss functions.

In this post, we motivate and explore the basics of Gibbs posteriors.

## Bayesian inference

Standard Bayesian inference requires a full model of the data to be posited in order to update beliefs. In particular, let $X = (x_1, \dots, x_n)^\top$ be the data. The analyst must specify a likelihood $p(X \| \theta)$ and a prior $\pi(\theta)$. They can then update their belief about the true value of $\theta$ under the model using the posterior,

$$p(\theta | X) = \frac{p(X | \theta) \pi(\theta)}{\int_\Theta p(X | \theta) \pi(\theta) d\theta}.$$

Specifying the true model of data generation is effectively impossible in practice. Furthermore, using a proper likelihood as a measure of $\theta$'s fidelity to the data might be restrictive. Computationally, computing this posterior can be cumbersome as well. All of these issues motivate the need for a more general and tractable approach to rationally updating beliefs.

## Loss functions

Gibbs posteriors are a particular (and important) type of *generalized posterior*. As their name suggests, generalized posteriors are an extension of Bayesian posteriors. These posteriors include distributions based on arbitrary loss functions. In particular, under a generalized posterior, we first posit a loss function $\ell_\theta(x)$, parameterized by $\theta$, that measures how closely the parameter $\theta$ agrees with the data. Well-known loss functions include the $L_2$ error, $\ell_\theta(x) = (\theta(x) - x)^2$ and the $L_1$ error, $\ell_\theta(x) = \|\theta(x) - x\|$, where in this case we take $\theta(x)$ to be a function of $x$.

Using this loss function, we can define the risk of an estimator $\widehat{\theta}$ of $\theta$. The risk $R(\widehat{\theta})$ is the expected loss we would incur under the true data generating model if we used $\widehat{\theta}$ as our estimator,

$$R(\widehat{\theta}) = \mathbb{E}_{x \sim p_0(x)}\left[\ell_{\widehat{\theta}}(x)\right]$$

where $p_0(x)$ is the true data generating distribution. The goal of any generalized Bayes inference procedure is then to minimize this risk with respect to $\theta$. 

$$\theta^\star = \text{arg}\min_{\theta}R(\theta).$$

In practice, when we observe data $X$, we seek to minimize the empirical risk $R_n(\theta)$,

$$R_n(\widehat{\theta}) = \frac1n \sum\limits_{i=1}^n\ell_{\widehat{\theta}}(x_i).$$

Notice that this optimization problem closely resembles maximum likelihood (ML) estimation and M estimation. In fact, it is identical to ML estimation if we take $\ell_{\theta}(x)$ to be the negative log likelihood under some model.

## Combining fidelity-to-loss and fidelity-to-prior

Now the question is: how do we take this empirical risk quantity and convert it into something that resembles a posterior? Clearly, one missing ingredient so far is a prior for $\theta$. Given an estimate $\widehat{\theta}$, we can think of the loss and risk as a measure of $\widehat{\theta}$'s fidelity to the observed data, and we can think of the prior as a measure of $\widehat{\theta}$'s fidelity to our prior belief. We now need a way to integrate these two types of fidelity and update our belief incrementally as we observe more data, similar to the role of a Bayesian posterior.

Suppose we decide to place a prior distribution on $\theta$, $\Pi(\theta)$. However, it's not immediately clear how to best combine the information from the prior and the risk function to update our beliefs. Plugging these into Bayes theorem is no longer viable because the loss might not be a well-defined likelihood, and we have not specified a full model for the data.

## Setting up the optimization problem

One approach to combining the loss and prior is to find the (generalized) posterior distribution $\mu$ that maximizes the sum of the fidelity to the data and the fidelity to the prior. We can clearly use the risk to measure the (inverse) fidelity to the data. To measure the (inverse) fidelity to the prior, we can define a divergence between the prior and $\mu$. A common choice is the KL-divergence,

$$D_{KL}(\mu \| \Pi) = \mathbb{E}_{\mu}\left[\log \frac{\mu}{\Pi}\right].$$

Now, we have the following optimization problem, where we minimize over all probability measures $\mu$,

\begin{equation} \inf_\mu \left\\{ \int R_n(\theta) \mu(d\theta) + \frac{1}{\omega n} D_{KL}(\mu \| \Pi) \right\\}. \label{eq:optim} \end{equation}

The first term is the expected risk given the measure $\mu$, and the second term is $\mu$'s divergence from the prior. As we will see below, $\omega$ is a tunable parameter that the analyst must set.

## Gibbs posteriors

It turns out that the solution to the optimization problem in \eqref{eq:optim} has a nice form, and is known as the Gibbs posterior. The Gibbs posterior $\Pi_n$ is defined as

$$\mu^\star = \Pi_n(d\theta) \propto \exp\{-\omega n R_n(\theta)\} \Pi(d\theta)$$

where $\omega$ is a learning rate (or "temperature") parameter. We can think of $\omega$ as balancing the influence of the data and the influence of the prior. Smaller values of $\omega$ favor the prior, with $\omega=0$ recovering the prior exactly and eliminating the influece of the data (although we typically require $\omega > 0$). Trivially, the normalizing factor is just the integral of this expression over the parameter space $\Theta$,

$$\int_\Theta \exp\{-\omega n R_n(\theta)\} \Pi(d\theta)$$

Notice that this Gibbs posterior framework is strictly more general than standard Bayesian inference in the sense that we can recover Bayesian inference as a special case. In particular, if we take the loss to be the negative log-likelihood for some model $p$,

$$\ell_\theta(x) = -\log p(x | \theta),$$

then we recover the typical Bayesian posterior distribution.

\begin{align} \Pi_n(d\theta) &\propto \exp\left\\{-\omega n \frac1n \sum\limits_{i=1}^n -\log p(x_i \| \theta)\right\\} \Pi(d\theta) \\\ &= \Pi(d\theta) \prod\limits_{i=1}^n p(x_i \| \theta)^{\omega}. \end{align}

If we set $\omega=1$, this is exactly the unnormalized Bayesian posterior.


## Conclusion

Gibbs posteriors generalize Bayesian posteriors by allowing the "loss" function to be a function other than a likelihood. This is an attractive alternative because it allows for much more flexible models, weaker assumptions, and possibly easier computation.

There's a lot more to say about Gibbs posteriors related to their theoretical properties and guarantees, as well as how to implement them in practice. There's also a deep connection to the rich PAC-Bayes literature, which I hope to explore in future posts.

## References
- McAllester, David A. "Some pac-bayesian theorems." Machine Learning 37.3 (1999): 355-363.
- Syring, Nicholas, and Ryan Martin. "Gibbs posterior concentration rates under sub-exponential type losses." arXiv preprint arXiv:2012.04505 (2020).
- Bissiri, Pier Giovanni, Chris C. Holmes, and Stephen G. Walker. "A general framework for updating belief distributions." Journal of the Royal Statistical Society. Series B, Statistical methodology 78.5 (2016): 1103.
- Rigon, Tommaso, Amy H. Herring, and David B. Dunson. "A generalized Bayes framework for probabilistic clustering." arXiv preprint arXiv:2006.05451 (2020).
