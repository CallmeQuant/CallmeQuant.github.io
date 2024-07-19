---
layout: post
title: "Critiques of Bayesian statistics"
author: "Andy Jones"
categories: journal
blurb: "'Recommending that scientists use Bayes' theorem is like giving the neighborhood kids the key to your F-16' and other critiques."
img: ""
tags: []
<!-- image: -->
---


"Recommending that scientists use Bayes' theorem is like giving the neighborhood kids the key to your F-16" and other critiques.

## Introduction

Bayesian models are often praised for being automatic inference engines: the modeler states her assumptions, and when data arrives, she can draw principled conclusions about parameters, latent variables, future predictions, and other quantities of interest.

However, Bayesian methods also receive an avalanche of criticism. The purpose of this post is to briefly review some of the most common and important challenges to Bayesian statistics. These issues range from practical computational issues to deeper issues with the Bayesian philosophy.

At the end of the post, I discuss some proposed resolutions to these issues.

## Questionable assumptions

The most common critique of Bayesian statistics is that its reliance on subjective prior assumptions can lead to wild conclusions. Of course, Bayesian priors are also an advantage of Bayesian statistics: they require the statistician to be completely transparent about her assumptions.

Nevertheless, there are some ways in which Bayesian inference can lead to confusing conclusions. 

### A simple example

One simple example (given by Andrew Gelman [here](https://statmodeling.stat.columbia.edu/2013/11/21/hidden-dangers-noninformative-priors/)) comes from the following simple model:

\begin{align} x &\sim \mathcal{N}(\mu, 1) \\\ \pi(\mu) &\propto 1 \end{align}

where the prior on $\mu$ is a flat, noninformative prior.

The posterior for $\mu$ is then 

$$p(\mu | x) \propto p(x | \mu) p(\mu) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac12 (x - \mu)^2\right)(1).$$

Now, suppose we observe one data point where $x=1$. Plugging this into the posterior, we have

$$p(\mu | x) \propto \frac{1}{\sqrt{2\pi}} \exp\left(-\frac12 (1 - \mu)^2\right)(1).$$

We can compute the probability that $\mu > 0$, which is

$$\int_0^\infty p(\mu | x) = 1 - \Phi(\mu < 1) \approx 1 - 0.159 = 0.841$$

where $\Phi$ is the Gaussian CDF. This implies that there's an $84\%$ posterior probability that $\mu$ is positive. Hence, after observing just one data point that is within one standard deviation from $0$, we are very confident that $\mu$ is positive. This feels a bit overconfident. If the "true" value of $\mu$ is 0, then observing one value of $x=1$ could be completely consistent with noise. Nevertheless, the staunch Bayesian concludes that $\mu$ is probably positive.

### The sorcery of choosing priors

The above is just one (slightly contrived) example of how Bayesian models can arrive at non-standard conclusions. In this example, the noninformative prior essentially did nothing to influence our conclusions, so any inference was completely reliant on the data (which was just one data point in this case).

Consider instead if we had placed a Gaussian prior on $\mu$, centered at $0$ with variance $1/10$ (i.e., we need lots of evidence to be convinced that $\mu$ is far away from $0$):

\begin{align} x &\sim \mathcal{N}(\mu, 1) \\\ \mu &\sim \mathcal{N}(0, 1/10) \end{align}

The posterior is then

$$\mu | x \sim \mathcal{N}\left(\frac{1}{11}, 11\right).$$

Then the probbaility that $\mu$ is positive is

$$P(\mu > 0) \approx 0.204,$$

which is much lower than before.

Given these very different conclusions, how should $\mu$ be chosen? In general, deciding on prior distributions requires substantial domain knowledge about the problem at hand. The modeler should use any known mechanisms or previous results to make these decisions. However, even then it is usually not obvious how to choose them.

The ability to choose these priors gives the modeler a lot of power, and any conclusions must be prefaced with the assumptions made. I enjoy the following [quote](http://www.stat.columbia.edu/~gelman/research/published/badbayesmain.pdf) from Andrew Gelman (written in jest):

> [...] recommending that scientists use Bayes’ theorem is like giving the neighborhood kids the key to your F-16.

The flip side of this coin is that the F-16 could actually be really useful if used with care, as noted by Uncle Ben:

> With great power comes great responsibility.

## Model selection

The very act of choosing a prior distribution immediately assumes that the true model is in the support of the prior. Thus, the problem of choosing subjective priors goes beyond the possibility that the candidate models are given the wrong weightings --- it's possible that the "true" model isn't in the prior support at all. 

Gelman and Shalizi describe this as a "principal-agent problem" in which the modeler (the principal) and the model (the agent) might not have aligned incentives:

> The problem is one of aligning incentives, so that the agent serves itself by serving the principal (Eggertsson, 1990). There is, as it were, a Bayesian principal–agent problem as well. The Bayesian agent is the methodological fiction (now often approximated in software) of a creature with a prior distribution over a welldefined hypothesis space $\Theta$, a likelihood function $p(y\|\theta)$, and conditioning as its sole mechanism of learning and belief revision. The principal is the actual statistician or scientist.

In other words, if the prior doesn't contain the true underlying model, the modeler will never find it. If we make the very reasonable assumption that all models are wrong, then we need a way to quantify *how* wrong each model is. Are we just a little bit off from the true model, or is the true model not even in the realm of possibilities?

## Computation

An even more practical concern with Bayesian methods is the intense computation that they sometimes require. Consider, for example, computing the posterior distribution for a parameter $\theta$ given some data $X$. The simple Gaussian example above had a closed form; however, in general, computing the posterior involves a nasty integral,

$$p(\theta | X) = \frac{p(X | \theta) p(\theta)}{\int_\Theta p(X | \theta) p(\theta) d\theta}.$$

The denominator --- which is the marginal likelihood $p(X)$ --- will be intractable for any but the simplest models.

Thus, to work around this integral, Bayesians typically use one of two solutions: 1) sampling-based methods for estimating the integral, or 2) approximations.

Sampling methods include a whole zoo of clever Markov chain Monte Carlo (MCMC) methods, like Metropolis-Hastings, Gibbs sampling, Hamiltonian Monte Carlo, and No U-Turn Sampling (NUTS). Although most of these methods will converge to the true posterior in the limit of infinite sampling, they tend to be extremely slow, as drawing sequences of random samples is computationally intensive.

On the other hand, approximation methods tend to be much faster, but the major downside is that they may never converge to the true posterior, even in the limit of infinite data. A popular approximation approach is variational inference. This approach requires the modeler to describe a second model --- the "variational model" --- which will then be fit to resemble the true posterior. However, numerous issues arise with variational methods. First, all of the issues with subjective priors in Bayesian statistics now reappear a second time: we must choose the priors of the underlying model *and* the variational model. Second, the approximate model might be a gross simplification of the desired model.

Taken together, these computational issues can make Bayesian statistics unappealing. Some modelers may prefer to avoid these issues altogether by using other computationally-nice methods that aren't weighed down by Bayesian integrals.

## Possible resolutions

All of the issues raised here lead one to wonder whether Bayesian methods are so inherently flawed that they should be avoided altogether. While some statisticians definitely do take this stance, most of the issues can be at least partially addressed. Below, I sketch out some high-level arguments for dealing with the issues above.

### Model checking

Using Bayesian models blindly and choosing wacky priors is, of course, not the right way to go. However, when extreme care is taken throughout the modeling process, Bayesian methods can be really useful.

A common pitfall of using Bayesian methods as an "automatic inference engine" is that a modeler might compute the posterior and call it a day. However, a crucial part of Bayesian analysis is model checking, as argued in depth in [this piece](https://bpspsychub.onlinelibrary.wiley.com/doi/pdfdirect/10.1111/j.2044-8317.2011.02037.x) by Andrew Gelman and Cosma Shalizi.

Model checking involves a whole series of steps beyond computing the posterior. For example, the modeler should ask: Do samples from the fitted model resemble my data? Can I make accurate predictions about held-out data or future data? Should I refine my model in some way?

In their piece, Gelman and Shalizi argue that Bayesian statistics should not be used simply as an tool for induction. Blind induction here means starting with a prior model, observing some data, updating the model, and viewing the model's predictions/parameters as truth. This approach assumes that the true model of the world exists within the model at hand, which is likely far from true.

Instead, Gelman and Shalizi favor a "hypothetico-deductive" view of data analysis. In this approach, we allow the possibility that our model is wrong, and we actively test for this possibility. This also allows for falsification in some sense, unlike a completely inductive view.

### Empirical Bayes

Empirical Bayesian methods take a concrete step toward making Bayesian priors less subjective. Rather than specifying the prior distribution(s) before observing any data, this family of methods uses the data to estimate the prior. This feels a bit weird --- is it really a "prior" if we set it *after* seeing the data? --- but these methods have been shown to have nice properties. And importantly, they only require placing as much information in the prior as the data allows.

### A willingness to be wrong

As above, many of the abuses of Bayesian statistics occur when a model is blindly believed to be correct. Given a model and some data, it's tempting to apply Bayesian methods as a black box and simply examine its outputs. But this approach can lead to disastrous conclusions, and probably slows down science in the long run.

As a corollary to model checking, Bayesian methods require the modeler to remain steadfast in checking her conclusions and assumptions. There must be a constant suspicion (or even assumption) that the current model is incorrect, and she should always be on the lookout for improvements. Simply relying on the dogma of Bayes will lead to results and conclusions, but possibly not the right ones.

## References
- Efron, Bradley. "Why isn't everyone a Bayesian?." The American Statistician 40.1 (1986): 1-5.
- Gelman, Andrew. "Objections to Bayesian statistics." Bayesian Analysis 3.3 (2008): 445-449.
- Gelman, Andrew, et al. Bayesian data analysis. CRC press, 2013.
- Lindley, Dennis V. "The future of statistics: a Bayesian 21st century." Advances in Applied Probability 7 (1975): 106-115.
- Talbott, William. "Bayesian epistemology." (2001).
- Gelman, Andrew, and Cosma Rohilla Shalizi. "Philosophy and the practice of Bayesian statistics." British Journal of Mathematical and Statistical Psychology 66.1 (2013): 8-38.
