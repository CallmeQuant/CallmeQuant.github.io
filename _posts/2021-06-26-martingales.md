---
layout: post
title: "Martingales"
blurb: "Martingales are a special type of stochastic process that are, in a sense, unpredictable."
img: ""
author: "Binh Ho"
categories: Probability theory
tags: []
<!-- image: -->
---

$$\DeclareMathOperator*{\argmin}{arg\,min}$$
$$\DeclareMathOperator*{\argmax}{arg\,max}$$

Stochastic processes model sequences of random variables. A special type is the martingale, which describes a process in which the subsequent observation cannot be predicted from past observations. In this post, we give an overview of the definition of martingales with a simple example.

## Background

In the world of casinos, a martingale refers to a betting strategy. Suppose a person is playing roulette in which they can bet any amount of money. We'll denote the bet amount on round $n$ as $X$. Now, suppose that the player plans to stick around for a while, and will continue playing until they win. Furthermore, after each unsuccessful round, they plan to double their bet in order to recoup their losses and have the opportunity for a cumulative gain. As soon as they win, they plan to stop playing.

If the gambler has infinite money and time, they're definitely going to walk away as a winner eventually. This comes from the fact that on each round, there's a nonzero chance of losing $p$, so the probability of losing $n$ rounds in a row is $p^n$. This probability will vanish as $n$ increases.

However, a more realistic scenario is that the gambler has finite money and time. If their initial bet is $X_0$, then after losing for $n$ rounds, their total loss is

$$\sum\limits_{i=1}^n 2^{i-1} X_0.$$

Then, the expected value of the gambler's overall value is

\begin{align} \underbrace{(1 - p^n) X_0}\_{\text{Expected gain}} - \underbrace{p^n \sum\limits_{i=1}^n 2^{i-1} X_0}\_{\text{Expected loss}} &= X_0 (1 - p^n - p^n \sum\limits_{i=1}^n 2^{i-1}) \\\ &= X_0 (1 - p^n(1 + \sum\limits_{i=1}^n 2^{i-1})) \\\ &= X_0 (1 - (2p)^n). \end{align}

If $p = \frac12$, then the overall expected gain will be zero. Thus, in a "fair" casino, even though the strategy works without fail with infinite resources, the gambler will come out even (in expectation) in a finite world. (Note that in real-world casinos, $p > \frac12$, so this isn't true).

This thought experiment forms the basic principle of martingales: a stochastic process whose expected value at the next step in the current step's value. Below, we first review probability spaces, and then describe a more formal definition of martingales.

## Probability spaces

Recall that a probability space is defined by a triplet $(\Omega, \mathcal{F}, \mathbb{P})$, where

- $\Omega$ is the set of all possible outcomes (the "sample space"),
- $\mathcal{F}$ is a $\sigma$-algebra of subsets of $\Omega$, and
- $\mathbb{P} : \mathcal{F} \rightarrow [0, 1]$ is a probability measure.

Consider the simple example of flipping a fair coin twice in a row. In this case, $\Omega = \\{(H, H), (T, T), (H, T), (T, H)\\}$ contains the possible outcomes (heads and tails in any order). Here, I'm using a pair in parentheses $(f_1, f_2)$ to denote the outcome of the first and second flips. 

For the $\sigma$-algebra $\mathcal{F}$, we can take the power set of $\Omega$. In particular, $\mathcal{F} = 2^{\Omega}$, which will contain $2^4$ elements, so we don't write it out here. 

Finally, our probability measure will map each of these sets to $[0, 1]$. We clearly know that the empty set will receive measure $0$ (because we definitely observed heads or tails on each flip) $\mathbb{P}(\\{\\}) = 0$, and the full sample space will receive probability $1$, $\mathbb{P}(\Omega) = 1$.

Now that we've reviewed probability spaces, we can move onto more martingale-specific ideas. Martingales are made up of two components: a filtration and a stochastic process. We'll describe each of these in turn.

## Filtrations

Suppose we have a measurable space $(\Omega, \mathcal{F})$. A filtration is a non-decreasing family of sub-$\sigma$-algebras $\\{\mathcal{F}_n\\}$ of the measurable space. In other words, each $\mathcal{F}_n$ is a subset of its successor,

$$\mathcal{F}_0 \subseteq \mathcal{F}_1 \subseteq \cdots \subseteq \mathcal{F}_n \subseteq \cdots \subseteq \mathcal{F}.$$

Intuitively, we can think of a filtration as a sequence of subsets, each of which contains more "information" that its predecessor.

Let's continue with the double coin flip example and examine the filtration for these. To help us, let's expand our model's notation a bit. Let $\Omega = \Omega_1 \times \Omega_2$ be the sample space for the two flips, where $\Omega_1 = \Omega_2 = \\{H, T\\}$ are the individual sample spaces for each throw.

First, consider the moment before we've flipped either of the coins. At this point, we can't really say anything about the experiment. The most we know is that _something_ will definitely happen. Mathematically, we know that $\mathbb{P}(\\{\\}) = 0$ and $\mathbb{P}(\\{\Omega\\}) = 1$. Thus our filtration at this time point is 

$$\mathcal{F}_0 = \{\{\}, \{\Omega\}\}.$$

Now, let's consider the moment after the coin has been flipped once. At this point, we know more information (specifically, the outcome of the first flip). Thus, we can answer questions like: Did the first flip come up heads? Did the first flip come up tails? Did the first flip come up heads or tails? Did the first flip come up neither heads nor tails? (Note that we could have answered these final two questions even before observing the first flip.) Our filtration at this time point becomes

\begin{align} \mathcal{F}_1 = 2^{\Omega_1} \times \Omega_2 &= \\{\\{\\}, H, T, \\{H, T\\}\\} \times \\{H, T\\} \\\ &= \\{(\\{\\}, \\{H, T\\}), (H, \\{H, T\\}), (T, \\{H, T\\}), (\\{H, T\\}, \\{H, T\\})\\} \end{align}

Notice that we still can't say anything about the second flip, so the second element in all of these pairs is the full $\Omega_2$.

Finally, after the second flip, we have as much information as we're going to get from this experiment. Our filtration is

\begin{align} \mathcal{F}_1 = 2^\Omega = \\{(\\{\\}, \\{\\}), (H, \\{\\}), (T, \\{\\}), \cdots, (\\{H, T\\}, \\{H, T\\})\\} \end{align}

which will consiste of $2^4$ total elements.

Clearly, we can see that $\mathcal{F}_0 \subset \mathcal{F}_1 \subset \mathcal{F}_2$ in this case.

## Stochastic processes

Recall that a stochastic process is a sequence of random variables. We can relate stochastic processes to filtrations through the concept of adaptation.

> **Definition**. A stochastic process $\\{X_n\\}, n = 0, 1, \dots$ is *adapted* to a filtration $\\{\mathcal{F}_n\\}$ if $\sigma(X_n) \subseteq \mathcal{F}_n$ for every $n$.

Let's again consider our coin flip example, this time expanding the story. Instead of just flipping the coin twice, we'll flip the coin $n$ times. Suppose we start with $\\$0$, and whenever the coin flip reads heads we receive $\\$1$, and whenever it reads tails, we lose $\\$1$. In this case, $X_n$ denotes our total earnings or losses, which is a random variable. This is a random walk.

Clearly, the easiest way to construct a filtration such that $X_n$ is adapted to it is to take $\mathcal{F}_n = \sigma(X_0, X_1, \dots, X_n)$. This is sometimes called the canonical filtration.

## Martingales

Finally, we can describe martingales using the concepts above. A martingale is a pair $(X_n, \mathcal{F}_n)$, where $\\{\mathcal{F}_n\\}$ is a filtration, and $\\{X_n\\}$ is a stochastic process adapted to this filtration such that for all $n$,

$$\mathbb{E}[X_{n+1} | \mathcal{F}_n] = X_n,$$

almost surely.

In other words, at step $n$, given all of the current information encoded in $\mathcal{F}_n$, our expectation of the value of $X$ at the next step is equal to the current step's value.

Let's turn back to our coin flip example. Recall that we earn $\\$1$ when the coin comes up heads and lose $\\$1$ when the coin comes up tails. Denote the cumulative amount of money at step $n$ as $X_n$, and the amount of money won or lost at step $n$ as $x_n \in \\{0, 1\\}$. 

We can show that this is a martingale. When we reflip the coin on each step, it has equal chance of coming up heads and tails -- and thus we have equal chance of gaining or losing $\\$1$. Thus, the expected value of our cumulative gain on step $n+1$ is going to be our gain so far, $X_n$. More formally,

\begin{align} \mathbb{E}[X_{n + 1} \| \mathcal{F}\_n] &= \mathbb{E}[x_{n + 1} \| \mathcal{F}\_n] + \mathbb{E}[X_n] \\\ &= 0.5 \cdot 1 + 0.5 \cdot -1 + \mathbb{E}[X_n] \\\ &= \mathbb{E}[X_n]. \end{align}

Furthermore, it's also enlightening to compute the expectation of $X_n$ without conditioning on the filtration. Using the tower property, we have

$$\mathbb{E}[X_n] = \mathbb{E}[ \mathbb{E}[X_n | \mathcal{F}_0]] = \mathbb{E}[X_0].$$

This shows that if we don't know anything about the sequence, the expected value at any point in the future will be our starting value.


## References

- Amir Dembo's [lecture notes](https://statweb.stanford.edu/~adembo/stat-310c/lnotes.pdf)
- James Aspnes's [notes](https://www.cs.yale.edu/homes/aspnes/pinewiki/Martingales.html).
- Wikipedia pages on martingales in [probability theory](https://www.wikiwand.com/en/Martingale_(probability_theory)) and [gambling](https://www.wikiwand.com/en/Martingale_(betting_system)).






