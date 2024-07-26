---
layout: post
title: "Ito's lemma"
author: "Binh Ho"
categories: Stochastic process
blurb: "A sketch of the derivation for Ito's Lemma and a simple example."
img: ""
tags: []
<!-- image: -->
---

A sketch of the derivation for Ito's Lemma and a simple example.

## Ito's Lemma

Suppose a variable $x$ follows an Ito process: $$dx = a(x, t) dt + b(x, t) dz$$ where $dz$ is a Wiener process (see my [previous post](https://callmequant.github.io/posts/2020/10/wiener-ito-processes/) for a brief introduction to Wiener processes. 

Ito's lemma shows that if we take a function of the variable $f(x, t)$, we can again write down the process that this function follows. Specifically, the lemma shows that we can write down the form of the differential $df$ in a fairly simple form. Ito's lemma is very similar in spirit to the chain rule, but traditional calculus fails in the regime of stochastic processes (where processes can be differentiable nowhere).

Here, we show a sketch of a derivation for Ito's lemma.

We start by taking the Taylor expansion of $df$:

$$\frac{\partial f}{\partial t} dt + \frac{\partial f}{\partial x} dx + \frac12 \frac{\partial^2 f}{\partial t^2} dt^2 + \frac12 \frac{\partial^2 f}{\partial x^2} dx^2 + \dots$$

Plugging in the Ito process for $dx$, we have

\begin{align} &\frac{\partial f}{\partial t} dt + \frac{\partial f}{\partial x} (a \; dt + b \; dz) + \frac12 \frac{\partial^2 f}{\partial t^2} dt^2 + \frac12 \frac{\partial^2 f}{\partial x^2} (a \; dt + b \; dz)^2 + \dots \\\ &= \frac{\partial f}{\partial t} dt + \frac{\partial f}{\partial x} (a \; dt + b \; dz) + \frac12 \frac{\partial^2 f}{\partial t^2} dt^2 + \frac12 \frac{\partial^2 f}{\partial x^2} (a \;^2 dt^2 + 2 a \; b \; dt dz + b \;^2 dz^2) + \dots \\\ \end{align}

Now, we're interested in the behavior of this differential as $dt \to 0$. Recall that the square of the Wiener process, $dz^2$, is equal to $dt$. We can see this from its variance:

\begin{align} \mathbb{E}[(z_{t1} - z_{t0})^2] &= \mathbb{V}[\Delta z] \\\ &= \Delta t \end{align}

since $dz \sim \mathcal{N}(0, dt)$ by the definition of a Wiener process.

This implies that the $dt^2$ terms will go to zero faster than the $dt$ and $dz^2$ terms. Dropping those terms and substituting $dt$ for $dz^2$, this leaves us with

\begin{align} &\frac{\partial f}{\partial t} dt + \frac{\partial f}{\partial x} (a \; dt + b \; dz) + \frac12 \frac{\partial^2 f}{\partial x^2} b^2 dt \\\ \end{align}

Rearranging, we have a simplified form of Ito's lemma:

$$\left(\frac{\partial f}{\partial x} a + \frac{\partial f}{\partial t} + \frac12 \frac{\partial^2 f}{\partial x^2} b^2 \right) dt  + \frac{\partial f}{\partial x} b \; dz$$

Ito's lemma can be used to model functions of other stochastic processes. Concretely, it's often used in the analysis of derivative securities, which are essentially functions of other securities. In the next section, we go over a simple application of Ito's lemma.

## Log-normal distribution of stock prices

Suppose we want to model the log of the stock price, $f(S) = \log S$. We can create a simple model of a stock price $S$ with an Ito process: $$dS = \mu S dt + \sigma S dz$$ where the drift rate $\mu$ and the variance rate $\sigma$ are constant. $z$ is a Wiener process, which means that $dz \sim \mathcal{N}(0, dt)$. 

Using Ito's lemma, we can easily plug in the partial derivatives to find the process followed by $f = \log S$.

$$df = \left(\frac{1}{S} a + 0 - \frac12 \frac{1}{S^2} b^2 \right) dt + \frac{1}{S} bdz.$$

Plugging in the values for $a$ and $b$, we have 

\begin{align} df &= \left(\frac{1}{S} \mu S - \frac12 \frac{1}{S^2} S^2 \sigma^2\right) dt + \frac{1}{S} \sigma S dz \\\ &= \left(\mu - \frac{\sigma^2}{2} \right) dt + \sigma dz \\\ \end{align}

Note that this is a generalized Wiener process, which implies that the change in the log-stock price $\Delta f$ over a period of time $\Delta t$ follows $$\Delta f \sim \mathcal{N}(\mu - \frac{\sigma^2}{2} \Delta t, \sigma^2 \Delta t).$$

In other words, $$\log S_t - \log S_{t^\prime} \sim \mathcal{N}((\mu - \frac{\sigma^2}{2}) (t - t^\prime), \sigma^2 (t - t^\prime)).$$

Since adding a constant to a Gaussian only shifts the mean, we have $$\log S_t \sim \mathcal{N}(\log S_{t^\prime} + (\mu - \frac{\sigma^2}{2}) (t - t^\prime), \sigma^2 (t - t^\prime)).$$

This implies that the stock price at time $t$, $S_t$, follows a log-normal distribution.

## References

- Hull, John. Options, futures and other derivatives/John C. Hull. Upper Saddle River, NJ: Prentice Hall,, 2009.
- Wikipedia page on [Ito's lemma](https://www.wikiwand.com/en/It%C3%B4%27s_lemma)
- Prof. Michael Stecher's [notes on stochastic differential equations].
