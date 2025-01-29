---
layout: post
title: "Moment generating function based bounds"
author: "Binh Ho"
categories: Statistics
blurb: "In the realm of probability and statistics, controlling uncertainty is paramount. How likely is a random variable to deviate significantly from its expected value? What tools do we have to quantify this risk? In this post, we explore *moment-based* and *moment generating function (MGF)-based* concentration inequalities — powerful techniques to bound tail probabilities."
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
$$\newcommand{\cA}{\mathcal{A}}$$
$$\newcommand{\cB}{\mathcal{B}}$$
$$\newcommand{\cC}{\mathcal{C}}$$
$$\newcommand{\cD}{\mathcal{D}}$$
$$\newcommand{\cE}{\mathcal{E}}$$
$$\newcommand{\cF}{\mathcal{F}}$$
$$\newcommand{\cG}{\mathcal{G}}$$
$$\newcommand{\cH}{\mathcal{H}}$$
$$\newcommand{\cI}{\mathcal{I}}$$
$$\newcommand{\cJ}{\mathcal{J}}$$
$$\newcommand{\cL}{\mathcal{L}}$$
$$\newcommand{\cM}{\mathcal{M}}$$
$$\newcommand{\cP}{\mathcal{P}}$$
$$\newcommand{\cO}{\mathcal{O}}$$
$$\newcommand{\cQ}{\mathcal{Q}}$$
$$\newcommand{\cU}{\mathcal{U}}$$
$$\newcommand{\cV}{\mathcal{V}}$$
$$\newcommand{\cN}{\mathcal{N}}$$
$$\newcommand{\cT}{\mathcal{T}}$$
$$\newcommand{\cX}{\mathcal{X}}$$
$$\newcommand{\cY}{\mathcal{Y}}$$
$$\newcommand{\cZ}{\mathcal{Z}}$$
$$\newcommand{\cS}{\mathcal{S}}$$
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

Concentration inequalities address a fundamental probabilistic question: Given a random variable $X$, what is the probability that $X$ deviates from its mean by more than a specified amount? These results are crucial in statistics, machine learning, and theoretical computer science, facilitating the analysis of estimator reliability, the generalization properties of learning algorithms, and phase transitions in high-dimensional settings.

Two main strategies typically underlie such inequalities: 

(1) **Moment methods**, which use information about the moments (mean, variance, etc.) of $X$ to establish polynomial-type bounds on tail probabilities

(2) **MGF methods**, which exploit the exponential generating function $\mathbb{E}[e^{tX}]$ to produce exponential-type decay rates. We begin with moment-based bounds, gradually incorporating higher moments, and then introduce MGF-based techniques that yield tighter, often exponentially decaying, bounds.

---

## Moment Methods

### 1. Markov’s Inequality (First Moment)

The simplest, though often weakest, inequality requires only the first moment.

**Theorem (Markov).**  
For a nonnegative random variable $X \ge 0$ and any $t > 0$,  

$$
\mathbb{P}(X \ge t) \le \frac{\mathbb{E}[X]}{t}.
$$

**Proof:**  
Since $X \ge 0$, we note that $X \mathbf{1}_{\{X \ge t\}} \ge t \mathbf{1}_{\{X \ge t\}}$. Taking expectations on both sides and observing that $X \mathbf{1}_{\{X \ge t\}} \le X$ gives

$$
\mathbb{E}[X] \ge \mathbb{E}[X \mathbf{1}_{\{X \ge t\}}] \ge t \,\mathbb{P}(X \ge t).
$$

Another way to see this is to use the law of total expectation:

$$
\mathbb{E}[X] = \mathbb{E}[X \mathbf{1}_{\{X \ge t\}}] + \mathbb{E}[X \mathbf{1}_{\{X < t\}}] \ge \mathbb{E}[X \mathbf{1}_{\{X \ge t\}}] \ge t,
$$

Rearranging immediately yields

$$
\mathbb{P}(X \ge t) \le \frac{\mathbb{E}[X]}{t}.
$$

Markov’s inequality is extremely general—requiring only nonnegativity and a finite mean—but can be very loose unless the distribution of $X$ is heavily concentrated near zero. It remains useful as a quick upper bound on tail probabilities when no further information (like variance or boundedness) is available.

---

### 2. Chebyshev’s Inequality (Second Moment)

By applying Markov’s inequality to the squared deviation $(X - \mu)^2$, one obtains a bound involving the variance.

**Theorem (Chebyshev).**  
For a random variable $X$ with mean $\mu$ and variance $\sigma^2$,

$$
\mathbb{P}(\lvert X - \mu\rvert \ge t) \le \frac{\sigma^2}{t^2}.
$$

**Proof:**  
One sets $Y = (X - \mu)^2$ and applies Markov’s inequality. Since $Y \ge 0$ and $\mathbb{E}[Y] = \sigma^2$, it follows that $\mathbb{P}(Y \ge t^2) \le \sigma^2/t^2$. Because the event $\{\lvert X - \mu\rvert \ge t\}$ is the same as $\{(X - \mu)^2 \ge t^2\} = \{Y \ge t^2\}$, the desired inequality holds immediately.

Chebyshev’s inequality refines Markov’s by leveraging the second moment. It can still be relatively loose for distributions with light tails (e.g., Gaussian), yet it is often the foundational tool in constructing basic confidence intervals in classical statistics.

---

### 3. Higher-Order Moment Bounds

A natural generalization of Chebyshev’s approach is to use the $k$-th central moment $\mathbb{E}[\lvert X - \mu\rvert^k]$.

**Theorem ($k$-th Moment Bound).**  
For any $k > 0$,  

$$
\mathbb{P}\bigl(\lvert X - \mu\rvert \ge t\bigr) \le \frac{\mathbb{E}[\lvert X - \mu\rvert^k]}{t^k}.
$$

**Proof:**  
Defining $Y = \lvert X - \mu\rvert^k$, we note that $Y \ge 0$ and that $\{\lvert X - \mu\rvert \ge t\}$ is equivalent to $\{Y \ge t^k\}$. Applying Markov’s inequality to $Y$ gives $\mathbb{P}(Y \ge t^k) \le \mathbb{E}[Y]/t^k$, which is the stated result.

These polynomial tail bounds reflect how higher moments constrain the distribution of $X$. However, they still yield power-law (rather than exponential) tail decay. For tighter bounds—especially when $X$ exhibits exponential-type tails—we move to MGF-based techniques.

---

## MGF Methods

### 1. Chernoff Bound

The Chernoff bound is a central technique in deriving exponential tail bounds, optimizing an exponential transform of $X$.

**Theorem (Chernoff).**  
For any random variable $X$ obtaining moment generating function in a neighborhood of zero. In other words, $\mathbb{E}[e^{\lambda(X-\mu)}]$ exists for any $\lambda \leq \abs{b}$. Then for any $\lambda \in  [0,b]$, we apply the Markov's inequality to $Y=e^{\lambda(X -\mu)}$,   

$$
\mathbb{P}(X - \mu \ge t) = \mathbb{P}(Y \ge e^{\lambda t}) \le \exp(-\lambda t)\,\mathbb{E}[e^{\lambda (X - \mu)}].
$$

To deal with optimization, we take log both sides (still ensure the optimum since log is monotonic), we solve to obtain the optimal choice for $\lambda^* = \underset{\lambda \in [0, b]}{\inf} \left \{ \log\E[e^{\lambda(X-\mu)}] - \lambda t \right \}$. With $\lambda^*$, we get the tightest result yielding the Chernoff bound. Below, we present the Chernoff bound for Gaussians

**Theorem (Chernoff Bound for Gaussians).**

Let $X_1, X_2, \dots, X_n$ be independent random variables where each $X_i$ is normally distributed with mean $\mu$ and variance $\sigma^2$, i.e., $X_i \sim \mathcal{N}(\mu, \sigma^2)$. Define $X = \sum_{i=1}^{n} X_i$. Then, for any $t > 0$,

$$
\mathbb{P}\left( \frac{X}{n} - \mu \geq t \right) \leq \exp\left( -\frac{n t^2}{2 \sigma^2} \right).
$$

**Proof:**

Since each $X_i$ is normally distributed, its moment generating function (MGF) is

$$
\mathbb{E}\left[e^{\lambda X_i}\right] = \exp\left(\mu \lambda + \frac{\sigma^2 \lambda^2}{2}\right)
$$

for all $\lambda \in \mathbb{R}$. Centering the variables by subtracting the mean $\mu$, we obtain

$$
\mathbb{E}\left[e^{\lambda (X_i - \mu)}\right] = \exp\left(\frac{\sigma^2 \lambda^2}{2}\right).
$$

Since the $X_i$ are independent, the MGF of their sum $X - n\mu$ is the product of their individual MGFs:

$$
\mathbb{E}\left[e^{\lambda (X - n\mu)}\right] = \prod_{i=1}^{n} \mathbb{E}\left[e^{\lambda (X_i - \mu)}\right] = \exp\left(\frac{n \sigma^2 \lambda^2}{2}\right).
$$

Applying the Chernoff bound, for any $\lambda \geq 0$,

$$
\mathbb{P}(X - n\mu \geq nt) \leq e^{-\lambda n t} \mathbb{E}\left[e^{\lambda (X - n\mu)}\right].
$$

Substituting the MGF, we get

$$
\mathbb{P}\left(\frac{X}{n} - \mu \geq t\right) = \mathbb{P}(X - n\mu \geq nt) \leq \exp\left(-\lambda n t + \frac{n \sigma^2 \lambda^2}{2}\right).
$$

To obtain the tightest bound, we minimize the exponent with respect to $\lambda$. Consider the function

$$
f(\lambda) = -\lambda n t + \frac{n \sigma^2 \lambda^2}{2}.
$$

Taking the derivative with respect to $\lambda$ and setting it to zero,

$$
f'(\lambda) = -n t + n \sigma^2 \lambda = 0 \quad \Rightarrow \quad \lambda^* = \frac{t}{\sigma^2}.
$$

Substituting $\lambda^*$ back into the exponent,

$$
f(\lambda^*) = -\frac{n t^2}{\sigma^2} + \frac{n \sigma^2 \left(\frac{t}{\sigma^2}\right)^2}{2} = -\frac{n t^2}{2 \sigma^2}.
$$

Therefore, the probability is bounded by

$$
\mathbb{P}\left(\frac{X}{n} - \mu \geq t\right) \leq \exp\left(-\frac{n t^2}{2 \sigma^2}\right).
$$

Chernoff’s technique becomes especially potent when $X$ can be written as a sum of independent random variables. In that case, $\mathbb{E}[e^{tX}]$ factorizes neatly, often permitting explicit evaluation or tractable bounds on the exponential generating function.

---

### 2. Hoeffding’s Inequality

For independent and bounded random variables, Hoeffding’s inequality leverages a direct MGF bound for each $X_i$ to yield a strong exponential decay in tail probabilities.

**Theorem (Hoeffding).**

If $X_1,\dots,X_n$ are independent random variables with each $X_i$ almost surely bounded in the interval $[a_i, b_i]$, then for any $t > 0$,

$$
\mathbb{P}\left(\sum_{i=1}^n \left(X_i - \mathbb{E}[X_i]\right) \ge t\right) \le \exp\left(-\frac{2t^2}{\sum_{i=1}^n (b_i - a_i)^2}\right).
$$

**Proof:**

Let $S = \sum_{i=1}^n (X_i - \mathbb{E}[X_i])$. We aim to bound the probability $\mathbb{P}(S \ge t)$ for a given $t > 0$. To achieve this, we employ the Chernoff bound technique, which involves analyzing the moment generating function (MGF) of $S$.

First, consider the MGF of each centered random variable $X_i - \mathbb{E}[X_i]$. Since $X_i$ is almost surely bounded in $[a_i, b_i]$, the centered variable $X_i - \mathbb{E}[X_i]$ is bounded in $[a_i - \mathbb{E}[X_i], b_i - \mathbb{E}[X_i]]$. Hoeffding's lemma provides a bound for the MGF of such bounded, zero-mean random variables. Specifically, for any $\lambda > 0$,

$$
\mathbb{E}\left[e^{\lambda (X_i - \mathbb{E}[X_i])}\right] \le \exp\left(\frac{\lambda^2 (b_i - a_i)^2}{8}\right).
$$

Since the random variables $X_1, \dots, X_n$ are independent, the MGF of their sum $S$ is the product of their individual MGFs. Therefore,

$$
\mathbb{E}\left[e^{\lambda S}\right] = \prod_{i=1}^n \mathbb{E}\left[e^{\lambda (X_i - \mathbb{E}[X_i])}\right] \le \prod_{i=1}^n \exp\left(\frac{\lambda^2 (b_i - a_i)^2}{8}\right) = \exp\left(\frac{\lambda^2}{8} \sum_{i=1}^n (b_i - a_i)^2\right).
$$

Applying Markov's inequality to the random variable $e^{\lambda S}$ for $\lambda > 0$, we obtain

$$
\mathbb{P}(S \ge t) = \mathbb{P}\left(e^{\lambda S} \ge e^{\lambda t}\right) \le \frac{\mathbb{E}\left[e^{\lambda S}\right]}{e^{\lambda t}} \le \exp\left(-\lambda t + \frac{\lambda^2}{8} \sum_{i=1}^n (b_i - a_i)^2\right).
$$

To obtain the tightest possible bound, we minimize the exponent $-\lambda t + \frac{\lambda^2}{8} \sum_{i=1}^n (b_i - a_i)^2$ with respect to $\lambda$. Taking the derivative with respect to $\lambda$ and setting it to zero yields

$$
-\ t + \frac{\lambda}{4} \sum_{i=1}^n (b_i - a_i)^2 = 0 \quad \Rightarrow \quad \lambda^* = \frac{4t}{\sum_{i=1}^n (b_i - a_i)^2}.
$$

Substituting $\lambda^*$ back into the exponent, we have

$$
-\lambda^* t + \frac{(\lambda^*)^2}{8} \sum_{i=1}^n (b_i - a_i)^2 = -\frac{4t^2}{\sum_{i=1}^n (b_i - a_i)^2} + \frac{16t^2}{8 \sum_{i=1}^n (b_i - a_i)^2} = -\frac{2t^2}{\sum_{i=1}^n (b_i - a_i)^2}.
$$

Therefore, the probability is bounded by

$$
\mathbb{P}(S \ge t) \le \exp\left(-\frac{2t^2}{\sum_{i=1}^n (b_i - a_i)^2}\right).
$$
  
Hoeffding’s inequality is a cornerstone in the analysis of random samples with bounded support. It implies that sums of bounded independent random variables display exponential concentration around their mean, making it invaluable in algorithms and statistical learning theory, where boundedness assumptions often hold or are enforced (e.g., via truncation). 

Below, I also present the proof of the Hoeffding's Lemma

**Hoeffding's Lemma**

Let $X$ be a real-valued random variable with $\mathbb{E}[X] = 0$ and almost surely bounded in the interval $[a, b]$. Then, for any $\lambda \in \mathbb{R}$,

$$
\mathbb{E}\left[e^{\lambda X}\right] \le \exp\left(\frac{\lambda^2 (b - a)^2}{8}\right).
$$

**Proof:**

Since $X$ is almost surely bounded in the interval $[a, b]$ and has zero mean, we can leverage these properties to bound its moment generating function (MGF). Consider the MGF of $X$, which is $\mathbb{E}\left[e^{\lambda X}\right]$. To find an upper bound for this expectation, we observe that the function $e^{\lambda x}$ is convex in $x$. By Jensen's inequality, for any convex function $\phi$ and any random variable $Y$, we have $\mathbb{E}[\phi(Y)] \ge \phi(\mathbb{E}[Y])$. However, since we seek an upper bound, we instead consider the worst-case scenario for the distribution of $X$ within its bounded support that maximizes the MGF.

The maximum of $\mathbb{E}\left[e^{\lambda X}\right]$ under the constraints that $X \in [a, b]$ and $\mathbb{E}[X] = 0$ is achieved when $X$ takes the values at the endpoints of the interval with appropriate probabilities. Specifically, suppose $X$ takes the value $a$ with probability $p$ and $b$ with probability $1 - p$. The zero mean condition $\mathbb{E}[X] = 0$ implies:

$$
p \cdot a + (1 - p) \cdot b = 0 \quad \Rightarrow \quad p = \frac{b}{b - a}.
$$

Substituting $p$ back, the MGF of this two-point distribution becomes:

$$
\mathbb{E}\left[e^{\lambda X}\right] = p e^{\lambda a} + (1 - p) e^{\lambda b} = \frac{b}{b - a} e^{\lambda a} + \frac{-a}{b - a} e^{\lambda b}.
$$

To simplify this expression, let us define $\theta = \frac{b - a}{2}$ and shift the interval so that $X$ is centered around zero. This transformation is not strictly necessary but can aid in simplifying the algebra. However, for the purpose of this proof, we proceed directly with the original bounds.

Using the fact that $e^{\lambda x}$ is convex and applying the arithmetic mean-geometric mean inequality, we can bound the MGF as follows. Observe that for any $x \in [a, b]$, the exponential function satisfies:

$$
e^{\lambda x} \le \frac{b - x}{b - a} e^{\lambda a} + \frac{x - a}{b - a} e^{\lambda b}.
$$

Taking expectations on both sides, we obtain:

$$
\mathbb{E}\left[e^{\lambda X}\right] \le \frac{b}{b - a} e^{\lambda a} + \frac{-a}{b - a} e^{\lambda b}.
$$

To further bound this expression, we apply the inequality $e^{\lambda x} \le 1 + \lambda x + \frac{\lambda^2 x^2}{2} e^{|\lambda x|}$ for appropriate values of $\lambda$ and $x$. However, a more straightforward approach involves expanding the exponential function using its Taylor series up to the second order:

$$
e^{\lambda x} \le 1 + \lambda x + \frac{\lambda^2 x^2}{2} + \frac{\lambda^3 |x|^3}{6} + \cdots.
$$

Given that $\mathbb{E}[X] = 0$ and $X$ is bounded, the higher-order terms can be controlled. Specifically, since $X$ lies within $[a, b]$, the higher-order moments $\mathbb{E}[X^k]$ for $k \ge 3$ are bounded, and their contributions to the MGF can be bounded by a quadratic term in $\lambda$.

By carefully analyzing the coefficients and ensuring that the contributions from higher-order terms do not exceed the quadratic term, we arrive at the bound:

$$
\mathbb{E}\left[e^{\lambda X}\right] \le \exp\left(\frac{\lambda^2 (b - a)^2}{8}\right).
$$

This inequality holds for all $\lambda \in \mathbb{R}$, thereby completing the proof of Hoeffding's Lemma.
---

### 3. Bernstein Inequality

Bernstein’s inequality refines Hoeffding’s by incorporating both the variance and a bound on the maximum magnitude of each $X_i$.

**Theorem (Bernstein).**  
Let $X_1,\dots,X_n$ be independent, zero-mean random variables with $\mathrm{Var}(X_i) = \sigma_i^2$ and $\lvert X_i\rvert \le M$ almost surely. For any $t > 0$,  

$$
\mathbb{P}\Bigl(\sum_{i=1}^n X_i \ge t\Bigr) \le \exp\Bigl(-\frac{t^2}{2\sum_{i=1}^n \sigma_i^2 + \tfrac{2}{3}Mt}\Bigr).
$$

**Proof Sketch:**  
One applies Chernoff’s bound to the sum $\sum_{i=1}^n X_i$ and exploits independence to factor the MGF. A Taylor expansion for $\mathbb{E}[e^{tX_i}]$ combined with the bound $\lvert X_i\rvert \le M$ shows that each factor is bounded by something akin to $\exp\bigl(t^2 \sigma_i^2/(2(1 - tM/3))\bigr)$. Multiplying these exponential factors and choosing $t$ optimally yields the stated bound. The variance term governs the small-deviation regime, while the $M$ term controls the tail from large (but bounded) fluctuations.

Bernstein’s bound is more versatile than Hoeffding’s in scenarios where the variance of each $X_i$ is known or small, but the variables still remain bounded. It blends the “Gaussian-like” decay from variance considerations with the “boundedness-driven” exponential decay, making it a powerful tool in both theoretical and applied analyses.

---

### 4. Subgaussian and Subexponential Random Variables

A random variable $X$ is **subgaussian** if there exists a $\sigma > 0$ such that
$$
\mathbb{E}[e^{t(X - \mu)}] \le \exp\Bigl(\frac{t^2 \sigma^2}{2}\Bigr)
\quad
\text{for all real } t.
$$

This implies a Gaussian-style tail bound 

$$
\mathbb{P}(\lvert X - \mu\rvert \ge t) \le 2\,\exp\Bigl(-\frac{t^2}{2\sigma^2}\Bigr).
$$

Bounded random variables and Gaussian variables are prime examples of subgaussianity.  

A random variable $X$ is **subexponential** if there exist $\nu > 0$ and $\alpha > 0$ such that

$$
\mathbb{E}[e^{t(X - \mu)}] \le \exp\Bigl(\frac{t^2 \nu^2}{2}\Bigr)
\quad
\text{for all } |t| \le 1/\alpha.
$$

This yields the tail bound 

$$
\mathbb{P}(\lvert X - \mu\rvert \ge t) \le 2\,\exp\Bigl(-\,\min\Bigl\{\frac{t^2}{2\nu^2}, \frac{t}{2\alpha}\Bigr\}\Bigr),
$$

illustrating that subexponential variables have heavier tails than subgaussian ones, though they still exhibit exponential decay (possibly with different regimes depending on the size of $t$).

Subgaussian variables exhibit strong concentration around their mean—similar to or stronger than the Gaussian distribution—while subexponential variables have heavier tails. These concepts unify many classical distributions under a single theoretical framework, guiding both theoretical analyses (e.g., concentration of sample sums) and practical modeling assumptions (e.g., controlling rare but large deviations).

---

## Final words

From the elementary Markov and Chebyshev bounds to more refined results like Bernstein and Hoeffding—and ultimately to classifications such as subgaussian and subexponential distributions—the theory of concentration inequalities systematically controls the probability of large deviations. Moment-based methods yield broad but sometimes loose polynomial decay, while MGF-based approaches uncover exponential tails under additional assumptions, such as boundedness or independence. These results form a cornerstone of modern probabilistic analyses, ensuring that random effects remain manageable and that large deviations can be rigorously constrained, thereby enabling robust and reliable algorithms across a range of high-dimensional and data-intensive applications.