---
layout: post
title: "Whirlwind tour of MCMC for posterior inference"
author: "Binh Ho"
categories: Computational statistics
blurb: "Markov Chain Monte Carlo (MCMC) methods encompass a broad class of tools for fitting Bayesian models. Here, we'll review some of the basic motivation behind MCMC and a couple of the most well-known methods."
img: ""
tags: []
<!-- image: -->
---


Markov Chain Monte Carlo (MCMC) methods encompass a broad class of tools for fitting Bayesian models. Here, we'll review some of the basic motivation behind MCMC and a couple of the most well-known methods.

## Introduction

Suppose we have some data $X$ and a model specifying the data generating process, where the model contains a set of parameters $\theta$. In Bayesian inference, we're interested in estimating the posterior of $\theta$ given $X$:

$$p(\theta | X) = \frac{p(X | \theta) p(\theta)}{\int p(X | \theta) p(\theta) d\theta}.$$

(The integral in the denominator will be a summation in the discrete case.) However, the denominator often isn't easily computable, so we must seek other methods for estimating the posterior.

MCMC methods (and Monte Carlo methods more generally) comprise a broad set of routines for estimating the posterior without having to analytically solve the above equation. The basic premise for this class of methods is to generate many random samples from a distribution that is proportional to the posterior, and then use this set of samples (and their resulting histogram) to compute relevant statistics and carry out the analysis. 

Many sampling-based methods have been developed, with developments being focused on different components of the underlying method, such as how to generate samples in the first place; how to ensure that the samples are truly "representative" of the posterior; and how to minimize the number of samples required.

The emphasis of this post is to build a foundation and intuition for why MCMC methods are important and why they work, with brief overviews of some of the most common methods. Hopefully, I'll have future posts that dive into specific approaches more deeply.

## Basic sampling methods

One of the central goals of performing full Bayesian inference is to get an idea of the shape of the posterior distribution $p(\theta \| X)$, which represents the model's representation of which parameter values are most probable, given that we've observed some data. Since it might be impossible to directly evaluate the posterior, the idea behind sampling-based methods is to iteratively sample a bunch of values from a function $f(x, \theta)$ that is proportional to the posterior, i.e., $f(x, \theta) \propto p(\theta \| x)$, and use these as a proxy.

First, we'll see a brief primer on using Markov chains to do this sampling, and then we'll review four main methods in chronological order of development: rejection sampling, Metropolis-Hastings, Hamiltonian Monte Carlo, and No U-Turn Sampling.

## Markov chains

Recall that Markov chains are defined by a set of states and the probabilities of transitioning between them. They have the crucial property that the probability of hopping to the next state only depends on the current state:

$$p(X_t = x_t | X_{t-1} = x_{t-1}, \dots, X_1 = x_1) = p(X_t = x_t | X_{t-1} = x_{t-1}).$$

It's straightforward to simulate Markov chains: just define a transition matrix $\mathbf{P}$ (where $\mathbf{P}_{ij}$ contains the probability of transitioning to state $x_j$, given that you're at state $x_i$), choose an arbitrary starting state, and then repeatedly choose the next state at random according to $\mathbf{P}$. After letting the chain run for a while, we will have accumulated a list of the states visited, and we could look at the frequency with which certain states were visited. In some cases, the frequency of states will always level out to the same distribution in the long run. This is called the **stationary distribution** of the Markov chain.

More formally, a distribution over states $\pi(y)$ is a stationary distribution if 

$$\pi(y) = \sum\limits_{x \in \mathcal{X}} p(y | x) \pi(x)$$

where $\mathcal{X}$ is the set of all states. We can see this as an expectation: $\pi(y)$ is really the expectation of $p(y \| x)$ w.r.t. $\pi(x)$ --- in other words, we get the same distribution back! We can also think of this in terms of transitions: given a distribution over states $\pi(x)$, if we take a step to the next state $y$, in expectation we'll have the same distribution $\pi$ over states again.

It turns out that some Markov chains have unique stationary distributions; that is, no matter how many times you run the chain, you'll always have the same distribution of states in the long run. The central condition for this to hold is that the chain must be **ergodic**. The details of this are beyond the scope of this post, but essentially an ergodic Markov chain is one in which any state can be reached from any other state in a finite number of steps.

Bringing the discussion back to Bayesian inference, we can think of the "states" as different values that the model parameters can take. Now, if we were able to sample (or transition between) these states such that we reach the stationary distribution, we'll be able to estimate the posterior distribution using these samples.

At some level of abstraction, MCMC methods differ in how they choose the transition probabilities between states (parameter values). Below, we review four of these methods, from the very basic rejection sampling to more sophisticated methods that are used in modern software packages.

### Rejection sampling

One of the simplest sampling methods is the **rejection sampling**. Rejection sampling is most useful in a context where it's difficult or impossible to directly sample from the desired density $p(\theta)$, but it's possible to evaluate the density given a value of $\theta$. In rejection sampling, we have a proposal distribution $q(\theta)$, and we draw samples from $q$, compare these to the desired distribution, and accept or reject the sample probabilistically. Going through the steps, it looks like

1. For $i = 1, \dots, n$:
    1. $x_i \sim q(\theta)$
    2. $u \sim \text{Uniform}(0, 1)$
    3. If $u < p(\theta_i) / q(\theta_i)$, accept $\theta_i$. Else, reject $\theta_i$.

Despite its attractive simplicity, a major disadvantage of rejection sampling is that as the dimension of the parameter space increases, it becomes more difficult to sample from the regions that are dense with probability (due to the curse of dimensionality).


### Metropolis-Hastings

The Metropolis-Hastings algorithm in its original form was developed in the 1950s, and now the name refers to broader family of methods. The algorithm alleviates much of the curse of dimensionality issue raised by simple rejection sampling by directly exploring parts of the parameter space with considerable density. This algorithm also uses a proposal function, but instead of uniformly sampling the parameter space, it makes slightly more informed decisions about how to jump from one sample to the next.

The algorithm seeks to draw samples from $f(\theta)$, a function that is proportional to the posterior $p(\theta \| X)$. In most cases, we can think of $f(\theta)$ as simply the numerator of the posterior (which allows us to avoid having to compute the integral in the denominator). It uses a proposal distribution $q(\theta)$ and a proposal distribution $g(\theta_t \| \theta_{t - 1}$. Here's the algorithm:

0. Start with a random value $\theta_0$
1. For $i = 1, \dots, n$:
    1. Draw new sample from proposal function: $\theta_i \sim g(\theta_i \| \theta_{i-1})$
    2. Calculate the acceptance ratio $\alpha = f(\theta_i) / f(\theta_{i-1})$
    3. Generate random uniform number $u \sim \text{Uniform}(0, 1)$
    4. If $u \leq \alpha$, accept $\theta_i$. Else, reject $\theta_i$.

Notice that the acceptance ratio will be equal to the ratio of the posteriors:

$$\frac{p(\theta_i | X)}{p(\theta_{i-1} | X)}  = \frac{f(\theta_i) / Z}{f(\theta_{i-1}) / Z} = \frac{f(\theta_i)}{f(\theta_{i-1})}$$

where $Z$ is the (intractable) normalizing constant.

Clearly the algorithm will quickly seek parts of the parameter space with high density because the acceptance probability will be higher in those areas. At the same time, it allows for exploring less dense areas with probability exactly proportional to the density in those areas.


### Gibbs sampling

Gibbs sampling is a special case of the generic Metropolis-Hastings algorithm that takes advantage of the conditional distributions of each parameter given the others. Specifically, assume we have $p$ parameters $\{\theta_1, \dots, \theta_p\}$. Gibbs sampling cycles through the list of parameters and updates each one, conditioned on the others:

1. For $i = 1, \dots, n$:
    1. For $k = 1, \dots, p$:
            1. Sample $\theta_k^{(i)} \sim p(\theta_k | \theta_{-k^{(i-1)}})$

### Hamiltonian Monte Carlo

We've seen that Metropolis-Hastings and Gibbs Sampling try to find clever ways to jump from one sample to the next. But what if we could explicitly use the curvature of the desired density to ensure that we're efficientlly sampling the space?

Hamiltonian Monte Carlo, originally developed using ideas in (statistical/computational) physics, does exactly this. The basic idea is to treat the density as a physical system in which the parameters can take on different "positions" within the system. Then, we can simulate the system using Hamiltonian dynamics to get samples that are representative of the density. 

Here, we take a basic look at the algorithm; however, I recommend consulting more detailed references (like those listed below) for the full mathematical details.

The canonical Hamiltonian equation(s) represents the total energy of a system in terms of its kinetic and potential energy:

$$\mathcal{H} = T + V$$

where $T$ is the kinetic energy and $V$ is the potential energy. Then, taking time derivatives of $\mathcal{H}$ lets us describe the evolution of the system through time.

To connect this with Bayesian inference, suppose we'd like to sample from some density $p(\theta)$. To employ HMC, we introduce an auxiliary "momentum" term $\rho$ and such that the joint density of $\theta$ and $\rho$ form a Hamiltonian:

\begin{align} H(\rho, \theta) &= -\log p(\theta, \rho) \\\ &= \underbrace{-\log p(\rho \| \theta)}\_{T(\rho \| \theta)} \underbrace{- \log p(\theta)}\_{V(\theta)}. \\\ \end{align}

Since the momentum isn't specified by our probability model, we must specify that on our own. Thus the basic steps of HMC look like this:

1. Pick a starting momentum.
2. Run the system forward for $k$ steps using Hamilton's equations.
3. Collect the samples from this path.
4. Repeat steps 1-3.

Since Hamilton's equations define a set of differential equations, we need a way to integrate them to get the positions (sampled parameter values). A commonly used method is [leapfrog integration](https://www.wikiwand.com/en/Leapfrog_integration).

To use [Alex Rogozhnikov's](https://arogozhnikov.github.io/2016/12/19/markov_chain_monte_carlo.html) analogy, one can compare HMC to pushing a hockey puck in some direction (choosing momentum), and then letting it slide around the curved ice rink for a while (running the system forward), and then tracking its position (collecting the samples). I also recommend checking out Alex Rogozhnikov's [great visualizations](https://arogozhnikov.github.io/2016/12/19/markov_chain_monte_carlo.html).

### No U-Turn sampler

HMC lets us sample the parameter space much more efficiently by using information in the gradient of the density. However, a key hyperparameter remains that still must be set by the user: the path length of each trajectory (i.e., the number of steps to run the system each time it's given a new momentum). If the path length is too short, the system may revert to doing something akin to a random walk. If the path length is too long, we might be doing wasteful computation and may be able to get samples of equal quality with shorter paths.

The No U-Turn Sampler (NUTS) attempts to select this hyperparameter automatically. Essentially, it tries to select a path length such that the the trajectories don't make "U-turns" in the parameter space. Returning to the hockey puck example, imagine a hockey puck sliding around in a bowl made of ice. If you push the puck in one direction, it's going to turn around once it reaches its peak height on the edge of the bowl. At a high level, NUTS adaptively selects the path length that ends before the puck has a chance to turn around (make a U-turn).

## Conclusion

This post was a high-level introduction to a just a few of the many MCMC algorithms. My focus was on estimating posteriors in Bayesian inference, but it should be noted that MCMC algorithms are useful in a wide range of areas, from phyiscs to finance.



## References 

- Prof. Charles Geyer's (introduction to MCMC](http://users.stat.umn.edu/~galin/Handbook/HandbookChapter1.pdf)
- Prof. Martin Haugh's [notes on MCMC](http://www.columbia.edu/~mh2078/MachineLearningORFE/MCMC_Bayes.pdf)
- Andrieu, Christophe, et al. "An introduction to MCMC for machine learning." Machine learning 50.1-2 (2003): 5-43.
- Stan's [notes on MCMC](https://mc-stan.org/docs/2_21/reference-manual/hamiltonian-monte-carlo.html)
- Wikipedia page on [Hamiltonian mechanics](https://www.wikiwand.com/en/Hamiltonian_mechanics#/overview)
- Colin Caroll's [notes on HMC](https://colindcarroll.com/2019/04/11/hamiltonian-monte-carlo-from-scratch/)
- Alex Rogozhnikov's [notes on HMC](https://arogozhnikov.github.io/2016/12/19/markov_chain_monte_carlo.html)
- Betancourt, Michael. "A conceptual introduction to Hamiltonian Monte Carlo." arXiv preprint arXiv:1701.02434 (2017).
- Hoffman, Matthew D., and Andrew Gelman. "The No-U-Turn sampler: adaptively setting path lengths in Hamiltonian Monte Carlo." J. Mach. Learn. Res. 15.1 (2014): 1593-1623.
- Richard McElreath's [post on NUTS](http://elevanth.org/blog/2017/11/28/build-a-better-markov-chain/)
