---
layout: post
title: "A friendly introduction to High-Dimensional Statistics (with the Johnson-Lindenstrauss embedding)"
author: "Binh Ho"
categories: Statistics
blurb: "High-dimensional statistics explores the complexities that arise when dealing with data sets where the number of features $d$ is large, often exceeding the number of samples $N$. Traditional statistical methods can struggle in these scenarios due to the curse of dimensionality, making it essential to develop specialized tools and techniques."
img: ""
tags: []
<!-- image: -->
---

$$\newcommand{\abs}[1]{\lvert#1\rvert}$$
$$\newcommand{\norm}[1]{\lVert#1\rVert}$$
$$\newcommand{\innerproduct}[2]{\langle#1, #2\rangle}$$
$$\newcommand{\Tr}[1]{\operatorname{Tr}\mleft(#1\mright)}$$
$$\DeclareMathOperator*{\argmin}{argmin}$$
$$\DeclareMathOperator*{\argmax}{argmax}$$
$$\DeclareMathOperator{\diag}{diag}$$
$$\newcommand{\converge}[1]{\xrightarrow{\makebox[2em][c]{$$\scriptstyle#1$$}}}$$
$$\newcommand{\quotes}[1]{``#1''}$$
$$\newcommand\ddfrac[2]{\frac{\displaystyle #1}{\displaystyle #2}}$$
$$\newcommand{\vect}[1]{\boldsymbol{\mathbf{#1}}}$$
$$\newcommand{\E}{\mathbb{E}}$$
$$\newcommand{\Var}{\mathrm{Var}}$$
$$\newcommand{\Cov}{\mathrm{Cov}}$$
$$\renewcommand{\N}{\mathbb{N}}$$
$$\renewcommand{\Z}{\mathbb{Z}}$$
$$\renewcommand{\R}{\mathbb{R}}$$
$$\newcommand{\Q}{\mathbb{Q}}$$
$$\newcommand{\C}{\mathbb{C}}$$
$$\newcommand{\bbP}{\mathbb{P}}$$
$$\newcommand{\rmF}{\mathrm{F}}$$
$$\newcommand{\iid}{\mathrm{iid}}$$
$$\newcommand{\distas}[1]{\overset{#1}{\sim}}$$
$$\newcommand{\Acal}{\mathcal{A}}$$
$$\newcommand{\Bcal}{\mathcal{B}}$$
$$\newcommand{\Ccal}{\mathcal{C}}$$
$$\newcommand{\Dcal}{\mathcal{D}}$$
$$\newcommand{\Ecal}{\mathcal{E}}$$
$$\newcommand{\Fcal}{\mathcal{F}}$$
$$\newcommand{\Gcal}{\mathcal{G}}$$
$$\newcommand{\Hcal}{\mathcal{H}}$$
$$\newcommand{\Ical}{\mathcal{I}}$$
$$\newcommand{\Jcal}{\mathcal{J}}$$
$$\newcommand{\Lcal}{\mathcal{L}}$$
$$\newcommand{\Mcal}{\mathcal{M}}$$
$$\newcommand{\Pcal}{\mathcal{P}}$$
$$\newcommand{\Ocal}{\mathcal{O}}$$
$$\newcommand{\Qcal}{\mathcal{Q}}$$
$$\newcommand{\Ucal}{\mathcal{U}}$$
$$\newcommand{\Vcal}{\mathcal{V}}$$
$$\newcommand{\Ncal}{\mathcal{N}}$$
$$\newcommand{\Tcal}{\mathcal{T}}$$
$$\newcommand{\Xcal}{\mathcal{X}}$$
$$\newcommand{\Ycal}{\mathcal{Y}}$$
$$\newcommand{\Zcal}{\mathcal{Z}}$$
$$\newcommand{\Scal}{\mathcal{S}}$$
$$\newcommand{\shorteqnote}[1]{ & \textcolor{blue}{\text{\small #1}}}$$
$$\newcommand{\qimplies}{\quad\Longrightarrow\quad}$$
$$\newcommand{\defeq}{\stackrel{\triangle}{=}}$$
$$\newcommand{\longdefeq}{\stackrel{\text{def}}{=}}$$
$$\newcommand{\equivto}{\iff}$$

<style>
.column {
  float: left;
  width: 30%;
  padding: 5px;
}

/* Clear floats after image containers */
.row::after {
  content: "";
  clear: both;
  display: table;
}
</style>

## Introduction
High-dimensional statistics explores the complexities that arise when dealing with data sets where the number of features $d$ is large, often exceeding the number of samples $N$. Traditional statistical methods can struggle in these scenarios due to the curse of dimensionality, making it essential to develop specialized tools and techniques. In this blog, we will introduce fundamental concepts and inequalities pivotal for high-dimensional analysis, including Markov’s inequality, the Chernoff bound, and Hoeffding bounds. Additionally, we will examine the properties of subgaussian random variables and discuss Bernstein-type bounds, which offer tighter concentration results under specific conditions. We will conclude with an exploration of the Johnson-Lindenstrauss embedding, a powerful dimensionality reduction technique that preserves the geometric structure of high-dimensional data in lower-dimensional spaces. Whether you are new to the field or looking to deepen your understanding, this series aims to equip you with the essential tools to navigate and analyze high-dimensional statistical challenges effectively.

## Overview of dimensionality
In traditional statistical frameworks, we analyze a set of $N$ data points $\{\vect{u}_1, \dots, \vect{u}_N\} \subseteq \R^d$. Here, $N$ represents the number of samples, indicating how many observations are collected, while $d$ denotes the ambient dimension of the model, specifying the number of distinct features or variables being measured.

**Example 1.1.** Consider surveying $N = 1000$ UCLA students to inquire about their GPA and their average nightly sleep duration. In this case, the system has $d = 2$ features: GPA and hours of sleep.

Within this classical setting, probability and statistical theories assert that as the number of samples grows indefinitely (i.e., $N \to \infty$), our statistical estimates become increasingly accurate.

**Theorem 1.1 (Strong Law of Large Numbers).** Let $X_1, X_2, \dots$ be independent and identically distributed (i.i.d.) random variables with $\E[\abs{X_1}] < \infty$. Then

$$
\frac{1}{N} \sum_{i=1}^N X_i \converge{\text{a.s.}} \E[X_1] \quad \text{as} \quad N \to \infty.
$$

Similarly, the Central Limit Theorem characterizes the distribution of the normalized sum $\frac{1}{N} \sum_{i=1}^N X_i$ as $N$ becomes large.

Recent advancements in statistics have shifted focus to scenarios where the number of samples $N$ is much smaller than the number of features $d$. This high-dimensional regime introduces unique challenges and necessitates different analytical approaches compared to classical settings.

**Example 1.2.** Suppose we are a biometrics company aiming to conduct statistical analysis of the human genome. We collect DNA data from $N = 100$ individuals, each with $d = 21,\!000$ genes.

In high-dimensional scenarios, traditional limit theorems may not be applicable. To address this challenge, we can adopt one of the following approaches:

1. **Derive quantitative results** that explicitly account for the dependence on both $d$ and $N$.
2. **Reduce the dimensionality** without incurring significant information loss.

The Johnson-Lindenstrauss embedding accomplishes the second approach by leveraging tools related to the first. In this blog, we will introduce fundamental probabilistic bounds in the form of concentration inequalities and utilize them to prove the Johnson-Lindenstrauss theorem.

## 2 Markov's inequality and the Chernoff bound
### Markov's inequality and Chebyshev's inequality
Consider a nonnegative random variable $X \geq 0$ with a known expected value $\E[X]$. For instance, if $\E[X] = 5$, then the probability $\bbP(X \geq 100)$ must be small; otherwise, such large values of $X$ would inflate the expectation beyond $5$. Markov's inequality provides a quantitative bound for this intuition:

**Theorem 2.1 (Markov's Inequality).** *Let $X \geq 0$ be a nonnegative random variable and $a > 0$. Then,*

$$
\bbP(X \geq a) \leq \frac{\E[X]}{a}.
$$

As $a$ increases, the bound on $\bbP(X \geq a)$ becomes tighter, indicating that significant deviations from the mean are less likely.

**Proof.** Since $X \geq 0$, we have:

$$
\begin{align*}
\E[X] &= \E\left[ X \, 1_{\{X < a\}} \right] + \E\left[ X \, 1_{\{X \geq a\}} \right] \\
&\geq 0 + \E\left[ X \, 1_{\{X \geq a\}} \right] \\
&\geq a \, \E\left[ 1_{\{X \geq a\}} \right] \\
&= a \, \bbP(X \geq a).
\end{align*}
$$

Dividing both sides by $a$ yields:

$$
\bbP(X \geq a) \leq \frac{\E[X]}{a}, \quad \forall a > 0.
$$

While Markov's inequality is useful, it often provides a loose bound because it relies solely on the expected value $\E[X]$. Moreover, if random variable $X$ also obtains finite variance, we can deduce the Chebyshev's inequality:

$$
\bbP(\abs{X - \E[X]} \geq a) \leq \frac{\Var[X]}{a^2}, \quad \forall a > 0.
$$

Note that the Chebyshev's inequality follows directly from the Markov's inequality by applying the latter to the non-negative random variable $(X - \E[X])^2$. We can further extend these in the similar fashion by noting that if $X$ has a central moment of order $k$, applying the Markov's inequality to $\abs{X - \E[X]}^k$ produce 

$$
\bbP(\abs{X - \E[X]}^k \geq a) \leq \frac{\E[\abs{X - \E[X]}^k]}{a^k}, \quad \forall a > 0.
$$

To obtain a tighter estimate, we can employ methods that incorporate additional information about the distribution of $X$.

### The Chernoff Bound

**Theorem 2.2 (Chernoff Bound).** *Let $X$ be a random variable and let $a > 0$. Then:*

$$
\bbP(X \geq a) \leq \inf_{t > 0} e^{-at} \, \E\left[ e^{tX} \right].
$$

**Remark 2.1.** The function $\E\left[ e^{tX} \right]$ is known as the moment-generating function (MGF) of $X$. By expanding the exponential function using its Maclaurin series, we have:

$$
\E\left[ e^{tX} \right] = \sum_{k=0}^\infty \E\left[ X^k \right] \frac{t^k}{k!},
$$

which incorporates information about $\E[X]$, $\Var(X)$, and higher-order moments.

**Remark 2.2.** A notable advantage of the Chernoff bound is that it does not require $X$ to be nonnegative, unlike Markov's inequality.

**Proof.** For any $t > 0$, consider:

$$
\begin{align*}
\bbP(X \geq a) &= \bbP(tX \geq ta) \\
&= \bbP\left( e^{tX} \geq e^{ta} \right).
\end{align*}
$$

Applying Markov's inequality to $e^{tX}$ gives:

$$
\bbP\left( e^{tX} \geq e^{ta} \right) \leq \frac{\E\left[ e^{tX} \right]}{e^{ta}} = e^{-at} \E\left[ e^{tX} \right].
$$

Since this inequality holds for all $t > 0$, taking the infimum over $t$ yields the Chernoff bound:

$$
\bbP(X \geq a) \leq \inf_{t > 0} e^{-at} \, \E\left[ e^{tX} \right].
$$

Let's examine an example to see how this bound can be applied in practice.

## Example 2.1

Let $X \distas{d} \Ncal(\mu, \sigma^2)$ be a normally distributed random variable with mean $\mu$ and variance $\sigma^2$. We aim to compute the moment-generating function $\E\left[ e^{t(X - \mu)} \right]$ and apply it to the Chernoff bound.

Since $X - \mu$ is normally distributed with mean $0$ and variance $\sigma^2$, its moment-generating function is:

$$
\E\left[ e^{t(X - \mu)} \right] = e^{\frac{\sigma^2 t^2}{2}}.
$$

Substituting this into the Chernoff bound, we have:

$$
\bbP(X - \mu \geq a) \leq \inf_{t > 0} e^{-at} \E\left[ e^{t(X - \mu)} \right] = \inf_{t > 0} \exp\left( -at + \frac{\sigma^2 t^2}{2} \right).
$$

To find the infimum, we minimize the exponent $-at + \frac{\sigma^2 t^2}{2}$ with respect to $t > 0$. This quadratic function reaches its minimum at:

$$
t = \frac{a}{\sigma^2}.
$$

Plugging this value back into the exponent:

$$
\begin{align*}
- a t + \frac{\sigma^2 t^2}{2} &= -a \left( \frac{a}{\sigma^2} \right) + \frac{\sigma^2}{2} \left( \frac{a}{\sigma^2} \right)^2 \\
&= -\frac{a^2}{\sigma^2} + \frac{a^2}{2 \sigma^2} \\
&= -\frac{a^2}{2 \sigma^2}.
\end{align*}
$$

Thus, the Chernoff bound simplifies to:

$$
\bbP(X - \mu \geq a) \leq \exp\left( -\frac{a^2}{2 \sigma^2} \right).
$$

Applying the same reasoning to $-X$, we find:

$$
\bbP(X - \mu \leq -a) = \bbP(- (X - \mu) \geq a) \leq \exp\left( -\frac{a^2}{2 \sigma^2} \right).
$$

Combining both results, we obtain:

$$
\bbP\left( | X - \mu | \geq a \right) \leq 2 \exp\left( -\frac{a^2}{2 \sigma^2} \right) \ \ (1).
$$

This inequality (also known as the *Gaussian tail bounds*) provides an exponential bound on the probability that $X$ deviates from its mean by at least $a$, showcasing the strength of the Chernoff bound in estimating tail probabilities for normal distributions.

## Sub-Gaussian variables and Hoeffding bounds
The tail bounds obtained using the Chernoff method are closely linked to the growth behavior of the moment-generating function (MGF). Therefore, when analyzing tail probabilities, it is natural to classify random variables based on the properties of their MGFs. As we will explore in the this section, the simplest and most fundamental type of such behavior is known as **sub-Gaussian**.

**Definition 3.1** A random variable $X$ with mean $\mu = \E[X]$ is *sub-Gaussian* if there is a positive number $\sigma$ such that 

$$
\E[\exp{(\lambda (X - \mu))}] \leq \exp{(\sigma^2 \lambda^2 / 2)} \quad \forall \lambda \in \R 
$$

The constant $\sigma$ is known as the **sub-Gaussian parameter**. We define a random variable $X$ to be **sub-Gaussian with parameter** $\sigma$ if it satisfies the condition 3.1. Specifically, this means that $X$ adheres to the inequality:

$$
\E\left[ e^{tX} \right] \leq e^{\sigma^2 t^2 / 2}, \quad \text{for all } t \in \R.
$$

As a direct consequence, any Gaussian random variable with variance $\sigma^2$ is sub-Gaussian with parameter $\sigma$. This is because the moment-generating function of a Gaussian variable exactly meets the above condition. Moreover, this is a form of **concentration inequality**. That is, any sub-Gaussian variable satisfies the inequality (1), which tells us It tells us that X probabilistically concentrates around its mean at an exponential rate in $t$. 




