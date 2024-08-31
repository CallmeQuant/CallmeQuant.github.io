---
layout: post
title: "Expectation-Maximization"
author: "Binh Ho"
categories: Statistics
blurb: "Expectation maximization is extremely useful when we have to deal with latent variable models (for example, number of mixture components in the mixture model)."
img: ""
tags: []
<!-- image: -->
---


Currently, I am interested in doing some researchs on Poisson mixture model and its application on
modeling count data and it's a good time to revise the Expectation-Maximization (EM) algorithm. This optimization technique is extremely useful when we have to deal with latent variable models (for example, number of mixture components in the mixture model). The underlying idea is that we will maximize the complete log 
likelihood instead of the usual log likelihood. Furthermore, EM algorithm alleviate the computational difficulty related to this complete-data log likelihood by iteratively construct and optimize a tight lower bound. Before delving into details, it's worth noting that EM algorithm has a close connection with variational 
inference and it lays a foundation on Variational Inference (VI); thereby, I also write a brief [introduction](https://callmequant.github.io/statistics/variational-inference.html) on Variational Inference for anyone interested.

## Introduction
*Expectation Maximization* algorithm, or EM for short, is a common approach to tackle the *maximum likelihood estimations* (MLE) for any probabilistic models
containing latent variables. Consider a probabilistic model settings in which the observed variables are denoted by $X$ with observed values $\lbrace x_1, \dots, x_N \rbrace$ 
and all latent variables by $Z$ with $\lbrace z_1, z_2, \dots, z_N \rbrace$. The parameters of the model are succintly denoted by $\theta$. To perform maximum likelihood inference, we need to derive the (log) likelihood $\log p(X \lvert \theta)$. However, since our data generating process comprises some latent variables $Z$, we have to include them in our log likelihood function (although we have not truly observed them). It can be done through marginalizing these variables out

$$
\log p(X \mid \theta) = \sum_{Z} \log p(X, Z \lvert \theta). \tag{1}
$$

The issue related to Eq. (1) is the intractability as the number of values that our hidden variables can take increases. For example, suppose the *Gaussian Mixture Model* (GMM) with $N$ observations whose the latent variables - the number of clusters can take on one of values from $M$ clusers. This results in $M^N$ terms in Eq. (1). 

And here comes the boom! Expectation-Maximization or EM ([Dempster et al., 1977](Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. Journal of the Royal Statistical Society. Series B (Methodological), 1–38.)) delivers a appropriate solution to address this problem. The underlying assumption is that the direct optimization of the log likelihood $\log p(X \lvert  \theta)$ is more challenging than maximizing the *complete-data log likelihood* $\log p(X, Z \lvert \theta)$ in some statistical problems. Additionally, as we have discussed earlier, the *complete-data log likelihood* is also intractable; thereby, EM constructs a lower bound on that likelihood function and iterativel optimize that lower bound. We will see later the EM algorithm always converges to a local maximum of $p(X \lvert \theta)$.

## General EM Algorithm
As we mention above, EM uses a lower bound as its objective function in its optimization problem. The derivation of the lower bound is as follows:
+ By introducing a latent variable $Z$, we may define the log marginal likelihood as 

$$
\log p(X \lvert \theta) =
\log \sum_{Z} p(X, Z \lvert \theta).
\tag{2}
$$

+ We notice that the summation inside the logarithm makes it extremely difficult to optimize such objective function. Hence, we need to get rid of that summation by using the 1-trick. To put it simply, we introduce a unrestrcited distribution  $q(Z)$ over the latent variable, simultaneously multiply and divide the term inside the summation of Eq. (2) by this distribution to obtain the *expectation* form 

$$
\begin{align*}
\log p(X \lvert \theta) &=
\log \sum_{Z} q(Z) \frac{p(X, Z \lvert \theta)}{q(Z)} \\
&= \log \mathbb{E}_{q(Z)} \frac{p(X, Z \lvert \theta)}{q(Z)}.
\tag{3}
\end{align*}
$$

+ We see that if we denote $f(x) = \log(x)$, then $f$ is a concave function and we can evoke a inequality by using the Jensen's inequality for a **concave** function $f$ (i.e. $f'' < 0$ for the domain of $f$) and a random variable $X$:
 
$$f(\mathbb{E}[X]) \geq \mathbb{E}[f(X)].$$

Then,

$$
\begin{align*}
\log p(X \lvert \theta) &= \log \Bigg( \mathbb{E}_{q(Z)} \frac{p(X, Z \lvert \theta)}{q(Z)} \Bigg) \\
& \geq \mathbb{E}_{q(Z)} \Bigg[ \log \frac{p(X, Z \lvert \theta)}{q(Z)} \Bigg] \\
&= \mathcal{L}(\theta, q). \tag{4}
\end{align*}
$$

The lower bound is a function of our parameter $\theta$ and the distribution $q$. Using the linearity of expection and property of logarithmic function, we can factorize as below

$$\log p(X \lvert \theta) \geq \underbrace{\mathbb{E}_{q(Z)} [\log p(X, Z \lvert \theta)]}_{\ast} \underbrace{- \mathbb{E}_{q(Z)} [\log q(Z)]}_{\ast \ast}. \tag{5}$$

where $(\ast)$ is the expected complete-data log likelihood and $(\ast \ast)$ is the *entropy* of $q$. The question arises is that how we can choose the density $q(Z)$ such that we have a *tight* lower bound? In a sense, we can inspect the difference between our log marginal likelihood and the lower bound to explore such problem 

$$
\begin{align*}
\log p(X \lvert \theta) - \mathcal{L}(\theta, q) &= 
\log p(X \lvert \theta) - \mathbb{E}_{q(Z)} \log \frac{p(X, Z \lvert \theta)}{q(Z)} \\ 
&=\mathbb{E}_{q(T)} \log \frac{p(X \lvert \theta) q(Z)}{p(X, Z \lvert \theta)} \\ 
&= \mathbb{E}_{q(Z)} \log \frac{q(Z)}{p(Z \lvert X, \theta)} \\ 
&= \mathrm{KL}(q(Z) \lvert \lvert p(Z \lvert X, \theta)).
\tag{6}
\end{align*}
$$

Surprisingly, the difference between $\log p(X \lvert \theta)$ and $\mathcal{L}(\theta, q)$ is the Kullback-Leibler (KL) divergence between the density $q(Z)$ and the true posterior over the latent variables. Since the KL divergence is always non-negative, setting the $q(Z)$ ideally to be $p(Z \lvert X, \theta)$ will make this KL divergence equals to zero and Eq. (5) will be an equality

$$\log p(X \lvert \theta) = \mathbb{E}_{p(Z \lvert X, \theta)} [\log p(X, Z \lvert \theta)] - \mathbb{E}_{p(Z \lvert X, \theta)} [\log p(Z \lvert X, \theta)].$$

At this step, we have all the ingredients for the construction of EM algorithm. , Keep in mind that since we can not use the complete-data log likelihood, we consider the expected value under the posterior of the latent variables which we can access from our knowledge about these hidden variables. Furthermore, we will introduce some indices to denote the evolution of the algorithm due to its iterative behaviour.

Assume that the current value for the parameter is $\theta^{t}$. Then, the lower bound is maximized with respect to $q(Z)$ while holding $\theta^{t}$ fixed. The maximum value occurs at $q(Z) = p(Z \lvert X, \theta)$, which causes the KL divergence to be vanished and the lower bound is indeed indetical to the log likelihood 

$$
\begin{align*}
\log p(X \lvert \theta) &= \mathbb{E}_{p(Z \lvert X, \theta^{t})} [\log p(X, Z \lvert \theta)] - \mathbb{E}_{p(Z \lvert X, \theta^{t})} [\log p(Z \lvert X, \theta)] \\
&= Q(\theta, \theta^{t}) + H(\theta, \theta^{t}).\\
\end{align*}
$$

where $Q(.)$ is the expected complete-data log likelihood under the optimal $q(Z) = p(Z \lvert X, \theta^{t})$ and $H(.)$ is simply the negative entropy of the $q$ distribution and hence independent of $\theta$. To clarify, the expression indicates the expectation of $\theta$ with respect to some other $\theta^{t}$. This is the E step in EM algorithm.

There is one more thing I want to elaborate on the E step. How can we guarantee that maximizing the lower bound give rise to the same amount compared to maximizing the log likelihood. I follow the common reasoning by many people when discussing this problem. The KL divergence can be decomposed with regard to the entropy (there is currently a post on [Jensen's inequality, Kullback-Leibler divergence and Entropy](https://callmequant.github.io/study/note-on-KL/) that I am still too busy to working on. Hopefully I have adequate time for it in the near future).

Given the construction of the **exptectation**  of the completed log likelihood, we will maximize this quantity in the M step. Thus EM iteratively establishes the desired tight lower bound of the log likelihood $\log p(x \lvert \theta)$, and optimizes that bound. Furthermore, it has been proven by Wu et al. in 1983 that this iterative scheme converges to the target log likelihood. 

To recap, the two steps included in EM algorithm are

$$
\begin{align*}
\text{E-step}: Q(\theta \lvert \theta^{t}) &= \mathbb{E}_{p(z \lvert x, \theta^{t})} [\log p(x, z \lvert \theta)] \\
\text{M-step}: \theta^{t+1} &= \mathop{\mathrm{argmax}} Q(\theta \lvert \theta^{t})
\end{align*}
$$










