---
layout: post
title: "Variational Inference - An Introduction"
author: "Binh Ho"
categories: Statistics
blurb: ""
img: ""
tags: []
<!-- image: -->
---

In this post, I first develop an intuition for variational inference as a pivotal family of algorithms using optimization for inference problems. From there, we  explore some derivations of the evidence lower bound (ELBO) and its properties.

# **Introduction**

Whenever we refer to the Bayesian inference problem, modern statisticians have demaned computation techniques that are scalable to extremely large data sets with thousand of unknown parameters to infer. Moreover, inference about unknown quantities in the real world often requires the computation of complex posterior densities. That's why an algorithm that could efficiently approximate these densities would be favorable.


While there are other prominent methods such as Markov chain Monte Carlo (MCMC) sampling and Sequential Monte Carlo (SMC), these alternatives tend to suffer from the curse of dimensionality. However, those strategies are studied extensively in the academics in terms of statistical properties and empirical experiment, which makes them still relevant in many settings.

To advocate the strength of Variational Inference (VI), consider the problem of computing posterior distribution in Bayesian frameworks. Recall Bayes' Theorem: 

$$p(Z \mid X ) = \frac{p(X, Z)}{\int_{Z}p(X, Z) \, dZ}$$
where $Z$ is a latent variable that supports the government of the distribution of the data. $P(X, Z)$ is the joint density of latent variables $Z$ and the observations $X$. Normally, inference in a Bayesian model is equivalent to computing the posterior distribution $P(Z \mid X)$ and needs approximate inference (since the denominator is usually multidimensional and intractable. That is, such integrals are usually intractable in the sense that 
+ We do not have an analytic expression to evaluate explicitly those integrals. 
+ The computational costs are so prohibitive that we can't carry out (or fast computation is of primary interest).


Variational inference (VI), unlike the class of sampling methods such as Metropolis-Hasting algorithm or Gibbs sampling, lends techniques from optimization to seek for a simpler and more tractable probability distribution with density $q(Z)$ bounded to some tractable family of distributions $\mathcal{Q}$ like Gaussians such that it can approximate the desired posterior distribution $p(Z \mid X)$. The optimal VI approximation is found by minimizing the **Kullback-Leibler** (KL) divergence from $q(Z)$ to $p(Z \mid X)$

$$q^\ast(Z) = \underset{q(Z) \in \mathcal{Q}}{\operatorname{arg min}} KL[q(Z) || p(Z \mid X)].$$

From the view of KL divergence, minimizing KL divergence can be intepreted as minimizing the relative entropy between two distributions (or the amount of information needed in one distribution when transforming to another distribution)

# **Evidence Lower Bound or (ELBO)**

The previous section offers a brief introduction about the variational inference method. In this section, we will dive into the way the concept of variaional optimization can be applied to the inference problem. As discussed earlier, suppose that we have a fully Bayesian model in which all given parameters are specified with prior distributions. Normally, the evidence is not analytically tractable. That's why we need some bound (or more precisely lower bound) that could be used to approximate the Evidence, $P(X)$. There are two standard ways to compute the ELBO quantity: via Jensen's Inequality and Kullback-Leibler Divergence. 

### **Why we coin the term "Evidence"**

To further understand the evidence lower bound, it is necessary to work out the term "evidence". **Evidence**, to put it simply, is a likelihood function evaluated at some fixed parameter, $\theta$ and is illustrated by the following quantity

$$\operatorname{log}p(X, \theta) $$
or 
$$\operatorname{log} p(X)$$ for the purpose of brevity

This quantity is intuitively called "evidence" because we would like to expect the marginal probability of the observed values $x$ to be high if the  model represented via $p$ and $\theta$ is chosen correctly. Hence, "evidence" indicates the data is modelled by the right choice of $p$ and $\theta$. 


## **Intuition**

We are given a relationship between our observations and latent variables 
Suppose that direct computation of the evidence is impossible due to intractable nature of the problem; however, the computation of the complete data-likelihood is much easier. Thus, we can introduce a distribution over the latent variables and decompose the log marginal probability as follows:


$$\operatorname{ln}p(X) = \mathcal{L}(q) + \operatorname{KL}(q || p) \tag{1}$$

where 

$$
\begin{aligned}
\mathcal{L}(q) & = \int q(Z) \operatorname{ln} \Bigg\lbrace \frac{p(X, Z)}{q(Z)} \Bigg\rbrace dZ \\
\operatorname{KL}(q || p) & = - \int q(Z) \operatorname{ln} \Bigg\lbrace \frac{p(Z \mid X)}{q(Z)} \Bigg\rbrace dZ
\end{aligned}
$$

The above relation could be easily verified by first making use of the product rule upon the joint distribution of $X$ and $Z$

$$\operatorname{ln}p(X, Z) = \operatorname{ln}p(Z \mid X) + \operatorname{ln}p(X). \tag{2}$$

which we then substitute into the expression for $\mathcal{L}(q)$ to give rise to two terms: the first term cancels the presence of $\operatorname{KL}[q \parallel p]$ while the other returns the required log likelihood $\operatorname{ln}p(X)$. 

## **First Derivation: KL Divergence**

Using the definition of Kullback-Leibler divergence, we can easily derive the equation 1:

$$
\begin{aligned}
  \operatorname{KL}[q(Z) || p(Z \mid X)] & = \int_{q} q(Z) \operatorname{log}\frac{q(Z)}{p(Z \mid X)} \\
  & = \mathbb{E}_{q(Z)} \Bigg[\operatorname{log}\frac{q(Z)}{p(Z \mid X)} \Bigg]\\
  & = \underbrace{\mathbb{E}_{q(Z)}[\operatorname{log}q(Z)] - \mathbb{E}_{q(Z)}[\operatorname{log}p(Z, X)]}_{\text{-ELBO(q)}} + \operatorname{log}p(X). 
\end{aligned}
$$

Recall that the computation of our desired KL divergence is impossible, we have to optimize another objective that is equivalent to this KL divergence at least up to a constant. The new objective is called the *evidence lower bound* or ELBO: 

$$\operatorname{ELBO}(q) := \mathbb{E}_{q(Z)}[\operatorname{log}p(Z, X)] - \mathbb{E}_{q(Z)}[\operatorname{log}q(Z)] \tag{3}$$

Then we can rewrite Eq.2 as

$$\operatorname{log}p(X) = \operatorname{ELBO}(q) + \operatorname{KL}[q(Z) \parallel p(Z \mid X)]. \tag{4}$$

Since the KL divergence must be non-negative, we can imply that 

$$\operatorname{log}p(X) \geq \operatorname{ELBO}(q). \tag{5}$$

As can be seen from Eq.5, the log evidence $\operatorname{log}p(X)$ is always a fixed quantity for any set of observations $X$ and it is greater than or equal to the ELBO. That's the reason why we have the so-called *"Evidence Lower Bound"*. Moreover, if we maximize the ELBO term in Eq.2, we are simultaneously minimizing the desired KL divergence (since the log evidence is fixed and independent of the variational distribution). 

## **Second Derivation: Jensen's Inequality**

### **Jensen's Inequality**

The Jensen's inequality could be stated in a probablistic form as

 follows:

>Theorem (Jensen's inequality (Jensen, 1906))

*Let $(\Omega, A, \mu)$ be a probability space, $g$ be a real-valued, $\mu$-integrable function and $\phi$ be a convex function that maps A to the real line. Then*

$$\phi\left(\int_{\Omega} g\, d\mu\right) \leq \int_{\Omega} \phi \circ g \, d\mu.$$

Furthermore, it is stated equivalently in the probability theory setting, in which we can slightly change the notation. Assumed that $(\Omega, \mathcal{F}, P)$ is a probability space, $X$ is an integrable real-valued random variable and $\phi$ a ***convex*** function. Then

$$\phi\left(\mathbb{E}[X]\right) \leq \mathbb{E}[\phi\left(X\right)]$$

In the case of ***concave*** function $\varphi$, the direction of the inequality is reversed 

$$\varphi\left(\mathbb{E}[X]\right) \geq \mathbb{E}[\varphi\left(X\right)]$$

Using Jensen's equality on the log probability distributipn of , we can directly derive Eq.5 by noting that 

$$
\begin{aligned}
  \operatorname{log}p(X) & = \operatorname{log} \int_{Z} p(X, Z) \\
  & = \operatorname{log} \int_{Z} p(X, Z) \frac{q(Z)}{q(Z)} \\
  & = \operatorname{log} \Bigg(\mathbb{E}_{q(Z)} \Bigg[\frac{p(X, Z)}{q(Z)} \Bigg] \Bigg) \\
  & \geq \mathbb{E}_{q(Z)}\Bigg[ \operatorname{log} \frac{p(X, Z)}{q(Z)} \Bigg] \quad (\text{Jensen's inequality})\\
  & = \mathbb{E}_{q(Z)} [\operatorname{log}p(X, Z)] - \mathbb{E}_{q(Z)}[\operatorname{log}q(Z)]
\end{aligned}
$$

We achieve the same result as using the definition of KL divergence. 

To recapitulate, the overal routine of Variational Inference can be summarized as follows
+ Initialize some distribution belongs to the tractable family of distributions $\mathcal{Q}$, $q^{(0)}(Z) \in \mathcal{Q}$.
+ Iteratively minimizes the KL divergence between the approximating at iteration $t$, denoted by $q^{(t)}(Z)$ and the targeted posterior distribution $p(Z \mid X)$. This can be done by alternatively maximizing the negative $\operatorname{ELBO}(q)$ as defined Eq.3.
+ The optimum of the above optimization problem $q^\ast(Z)$ is the best approximation of the distribution.
