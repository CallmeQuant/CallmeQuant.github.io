---
layout: post
title: "Improper priors"
author: "Binh Ho"
categories: Statistics
blurb: "Choosing a prior distribution is a philosophically and practically challenging part of Bayesian data analysis. Noninformative priors try to skirt this issue by placing equal weight on all possible parameter values; however, these priors are often 'improprer' -- we review this issue here."
img: ""
tags: []
<!-- image: -->
---


Choosing a prior distribution is a philosophically and practically challenging part of Bayesian data analysis. Noninformative priors try to skirt this issue by placing equal weight on all possible parameter values; however, these priors are often "improprer" -- we review this issue here.

## Introduction

Bayesian models can be strongly influenced by the choice of prior distribution, especially in the low-data regime. In many common Bayesian models that have nice inference properties, the conjugate prior (i.e., the prior that's nice to use for downstream inference) has some level of "informativeness", meaning that it makes an assumption about where the true parameter value lies. For example, in a univariate, single-parameter Gaussian model with known variance and unknown mean, a Gaussian prior is often placed on the mean. We can denote this model as follows:

\begin{align} X_1, \dots, X_n &\sim \mathcal{N}(\mu, \sigma^2) & \text{(Likelihood)} \\\ \mu &\sim \mathcal{N}(0, 1) & \text{(Prior)} \\\ \end{align}

In this simple example, we're assuming we know something about the plausible values for $\mu$, since we're assuming that its prior is centered at zero and has Gaussian tails that drop off fairly quickly. If the true value for $\mu$ were very large (100, say), we would only be able to get a somewhat accurate estimate for $\mu$ if we had a ton of data, in which case the likelihood would overpower the prior.

In many cases, positing that $\mu$ is somewhat close to $0$ is a reasonable assumption. But what if we want to assume nothing about the value for $\mu$, and let the data completely speak for itself?

One way of achieving this would be to place a uniform prior distribution on $\mu$ in the range $(-\infty, \infty)$. However, we have immediately violated the laws of probability and Bayesian analysis since this distribution does not have a finite sum. To be more specific, if we assume $p(\mu) = c$ for some constant $c$, then $\int_{-\infty}^\infty p(\mu) d\mu = \infty$, and there's no way to normalize the distribution.

This is the defining feature of an improper prior distribution: it does not sum to 1 (or, it does not sum to a finite value, and cannot be normalized to sum to 1).

However, even though we've violated the rules for the prior distribution, we might still be able to salvage the model if the **posterior** distribution is proper. Indeed, this is the case for a subset of improper priors.

## Gaussian example

To continue our unknown-mean Gauassian example above, suppose we place the following prior on $\mu$: 

$$p(\mu) = 1.$$

This seems odd at first: the probability of $\mu$ doesn't even depend on $\mu$! Instead, every value of $\mu$ in $(-\infty, \infty)$ is equally likely. The posterior distribution for $\mu$ is then

\begin{align} p(\mu \| X) &= \frac{p(X \| \mu)p(\mu)}{p(X)} \\\ &= \frac{p(X \| \mu)p(\mu)}{\int p(X \| \mu)p(\mu) d\mu} \\\ &= \frac{p(X \| \mu)}{\int p(X \| \mu) d\mu} & \text{($p(\mu) = 1$)}. \\\ \end{align}

Clearly, this posterior will still be proper, since $p(X \| \mu)$ is Gaussian and will definitely sum to 1.

## Bernoulli example

Consider a simple Beta-Bernoulli model with unknown Bernoulli rate $p$:

\begin{align} X_1, \dots, X_n &\sim \text{Bern}(p) \\\ p &\sim \text{Beta}(\alpha, \beta) \\\ \end{align}

To achieve a uniform prior distribution on $\theta$, we could again simply say $p(\theta) = c$ again. Alternatively, we could define the uniform through the Beta distribution.

Recall that the Beta PDF is

$$f(\theta) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha) \Gamma(\beta)} \theta^{\alpha - 1} (1 - \theta)^{\beta - 1}.$$

Notice that if $\alpha = \beta = 1$, then

\begin{align} f(\theta) &= \frac{\Gamma(2)}{\Gamma(1) \Gamma(1)} \theta^{0} (1 - \theta)^{0} \\\ &= \frac{\Gamma(2)}{\Gamma(1) \Gamma(1)} \theta^{0} (1 - \theta)^{0} \\\ &= 1. \end{align}

Thus, when $\alpha = \beta = 1$, we effectively have a uniform prior distribution on $\theta$.

Again our posterior distribution on $p$ will be proper since

\begin{align} p(\theta \| X) &= \frac{p(X \| \theta) p(\theta)}{\int_0^1 p(X \| \theta) p(\theta) d\theta} \\\ &= \frac{p(X \| \theta)}{\int_0^1 p(X \| \theta) d\theta} \end{align}

and the denominator will have a finite integral since it's just a Bernoulli distribution.

## Gaussian variance example

Consider a Gaussian model with known mean and unknown variance $\sigma^2$. A common improper prior for $\sigma^2$ is $p(\sigma^2) = 1/\sigma^2$. This prior assumes that larger values of $\sigma^2$ are more improbable.

It can be shown that this prior induces a posteiror $p(\sigma^2 \| X)$ that has the same form as the posterior for a model that assumes an inverse-$\chi^2$ distribution for $\sigma^2$. This equivalence is true only as a limiting distribution as the data overpowers the prior.

The form of the posterior is then

\begin{align} p(\sigma^2 \| X) &= \frac{p(X \| \sigma^2) p(\sigma^2)}{p(X)} \\\ &\propto \frac{1}{\sigma \sqrt{2\pi}} \exp\left( -\frac{1}{2\sigma^2} (x - \mu)^2 \right) \frac{1}{\sigma^2} \\\ &= \frac{1}{\sqrt{2\pi}} \sigma^{-3} \exp\left( -\frac{1}{2\sigma^2} (x - \mu)^2 \right), \\\ \end{align}

which, after normalization, is an inverse-$\chi^2$ distribution.

## Jeffreys' principle

An issue that arises with assigning a straightforward uniform prior distribution is that it is not invariant to a transformation of variables. For example, in the Beta-Bernoulli case, if we reparameterize the model with the log-odds ratio ($\rho = \log \frac{\theta}{1 - \theta}$), then assigning a uniform distribution to $\theta$ is no longer uninformative.

[Harold Jeffreys](https://www.wikiwand.com/en/Harold_Jeffreys) attempted to alleviate this issue by asserting that any uninformative prior distribution should yield the same result, regardless of any reparameterization. 

In particular, Jeffreys asserted that the uninformative prior distribution should be determined as 

$$p(\theta) = [\mathcal{I}(\theta)]^{1/2}$$

where $\mathcal{I}(\theta)$ is the Fisher information of $\theta$.

To see this, suppose we use the transformation $\phi = h(\theta)$, which implies that $\theta = h^{-1}(\phi)$. Then the Fisher information of $\phi$ is

\begin{align} \mathcal{I}(\phi) &= -\mathbb{E}\left[ \frac{d^2 \log p(y \| \phi)}{d \phi^2} \right] & \text{(Definition of FI)} \\\ &= -\mathbb{E}\left[ \frac{d^2 \log p(y \| \theta)}{d \theta^2} \right] \left\|\frac{d\theta}{d\phi}\right\|^2 & \text{(Transformation of vars.)} \\\ &= \mathcal{I}(\theta) \left\|\frac{d\theta}{d\phi}\right\|^2 \\\ \implies& \mathcal{I}(\phi)^{1/2} = \mathcal{I}(\theta)^{1/2} \left\|\frac{d\theta}{d\phi}\right\|. \end{align}

This last line implies that, regardless of any transformation of variables, the prior $p(\theta) = [\mathcal{I}(\theta)]^{1/2}$ will always yield the same result.

This for of prior distribution is known as Jeffreys' prior, and it provides a systematic way to find a reasonable uninformative prior distribution. In some cases, Jeffreys' prior will be improper, but not always.

## Improper priors as limits of proper priors

In some cases, it's possible to interpret improper priors as the limiting distribution for a prior distribution.

For example, in the Gaussian model case with unknown mean and variance $\sigma^2 = 1$, an improper prior $p(\mu) = 1$ led to the posterior

$$p(\mu | X) \propto \frac{1}{\sqrt{2\pi}} \exp\left( -\frac12 (x - \mu)^2 \right).$$

If we use the proper Gaussian prior 

$$p(\mu; \mu_0, \sigma^2_0) = \frac{1}{\sigma_0 \sqrt{2\pi}} \exp\left( -\frac{1}{2\sigma_0^2} (\mu - \mu_0)^2 \right),$$

then the posterior is

\begin{align} p(\mu \| X) &\propto \frac{1}{ \sqrt{2\pi}} \exp\left( -\frac{1}{2} (x - \mu)^2 \right) \frac{1}{\sigma_0 \sqrt{2\pi}} \exp\left( -\frac{1}{2\sigma_0^2} (\mu - \mu_0)^2 \right) \\\ &= \left( \frac{1 + \sigma_0^2}{2\pi \sigma_0^2} \right)^{1/2} \exp\left( -\frac12 \left( \frac{1 + \sigma_0^2}{\sigma_0^2} \right) \left( \mu - \frac{\mu_0 + \sigma_0^2 x}{1 + \sigma_0^2} \right)^2 \right). \end{align}

Then if we let $\sigma_0^2 \to \infty$, we obtain the same posterior as was yielded by the improper prior,

$$p(\mu | X) \propto \frac{1}{\sqrt{2\pi}} \exp\left( -\frac12 (x - \mu)^2 \right).$$

In effect, we can imagine the variance of the prior getting larger and larger, which effectively reduces it to the uniform distribution in the limit.


## Conclusion

Improper priors can be practically useful for Bayesian data analysis when seeking to avoid imposing strong prior beliefs. However, these priors can come at the cost of interpretability and deviation from classical Bayesian principles.

## References

- Gelman, Andrew, et al. Bayesian data analysis. CRC press, 2013.
- Ben Ewing's [post on improper priors](https://improperprior.com/post/2020/03/16/what-is-an-improper-prior/).
- Prof. Michael Jordan's [notes on Jeffreys' prior](https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/lectures/lecture6.pdf).
- Akaike, Hirotugu. "The interpretation of improper prior distributions as limits of data dependent proper prior distributions." Journal of the Royal Statistical Society: Series B (Methodological) 42.1 (1980): 46-52.
