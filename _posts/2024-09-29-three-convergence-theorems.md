---
layout: post
title: "Three convergence theorems in integration theory"
author: "Binh Ho"
categories: Measure theory
blurb: "In this post, we dive into three essential convergence theorems in integration theory and explore how the Lebesgue integral handles limits so effectively."
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

In this post, we dive into three essential convergence theorems in integration theory and explore how the Lebesgue integral handles limits so effectively. We'll break down the Monotone Convergence Theorem along with two other key theorems, showing why the Lebesgue approach is so powerful when working with limits. Whether you're looking to deepen your understanding of measure theory or enhance your integration skills, this guide provides clear explanations and insightful proofs to help you grasp the important role of convergence in mathematical analysis.

## Monotone Convergence Theorem
**Theorem (Monotone Convergence Theorem):** Let $(f_n)$ be a sequence of measurable functions on $X$ such that:
a) $0 \leq f_1(x) \leq f_2(x) \leq \dots \leq f_n(x) \leq \dots \leq \infty$, for all $x \in X$,
b) $f_n(x) \to f(x)$ as $n \to \infty$, for all $x \in X$.
Then $f$ is measurable and

$$
\int_X f_n \, d\mu \xrightarrow{n \to \infty} \int_X f \, d\mu.
$$

**Proof:**

Since $(f_n)$ is a sequence of nonnegative measurable functions increasing pointwise, we have

$$
\int_X f_n \, d\mu \leq \int_X f_{n+1} \, d\mu.
$$

Thus, the sequence $\left( \int_X f_n \, d\mu \right)$ is non-decreasing and bounded above by $\int_X f \, d\mu$. Therefore, the limit exists in $[0, \infty]$:

$$
\lim_{n \to \infty} \int_X f_n \, d\mu = \alpha \in [0, \infty].
$$

Next, since $f = \lim_{n \to \infty} f_n$ and each $f_n$ is measurable, $f$ is measurable as the pointwise limit of measurable functions.
Moreover, because $f_n(x) \leq f(x)$ for all $n$ and all $x \in X$, it follows that

$$
\int_X f_n \, d\mu \leq \int_X f \, d\mu,
$$

for all $n \in \mathbb{N}$. Therefore,

$$
\alpha = \lim_{n \to \infty} \int_X f_n \, d\mu \leq \int_X f \, d\mu. \quad \text{(1)}
$$

To show the reverse inequality $\alpha \geq \int_X f \, d\mu$, consider any simple measurable function $s$ such that $0 \leq s \leq f$. We aim to show that

$$
\int_X s \, d\mu \leq \alpha.
$$

That is, to bridge the limit we have to approximate the non-negative measureable function using simple functions. For a given $c \in (0, 1)$, define the sets

$$
E_n = \{ x \in X : f_n(x) \geq c s(x) \}, \quad n \in \mathbb{N}.
$$

Notice that each $E_n$ is measurable, and $E_1 \subseteq E_2 \subseteq \dots$, and

$$
X = \bigcup_{n=1}^\infty E_n.
$$

To see this, consider any $x \in X$. If $f(x) = \lim_{n \to \infty} f_n(x) = 0$, then since $(f_n)$ is increasing and nonnegative, $f_1(x) = 0$, hence $x \in E_1$. If $f(x) > 0$, then $c s(x) < f(x)$ (because $0 < c < 1$), and thus there exists some $n$ such that $f_n(x) \geq c s(x)$, implying $x \in E_n$.
Therefore,

$$
\int_X f_n \, d\mu \geq \int_{E_n} f_n \, d\mu \geq c \int_{E_n} s \, d\mu.
$$

Taking the limit as $n \to \infty$, we obtain

$$
\alpha = \lim_{n \to \infty} \int_X f_n \, d\mu \geq c \int_X s \, d\mu.
$$

Since this holds for all $c \in (0, 1)$, letting $c \to 1^{-}$ gives

$$
\alpha \geq \int_X s \, d\mu.
$$

Taking the supremum over all such simple functions $s$, we get

$$
\alpha \geq \sup \left\{ \int_X s \, d\mu : 0 \leq s \leq f, \ s \text{ simple} \right\} = \int_X f \, d\mu. \quad \text{(2)}
$$

From (1) and (2), we conclude that

$$
\alpha = \int_X f \, d\mu,
$$

and hence

$$
\int_X f_n \, d\mu \xrightarrow{n \to \infty} \int_X f \, d\mu.
$$

This completes the proof.

One feature of measure theory is that we can usually weaken the assumptions to hold $\mu-\text{a.e.}$ instead of everywhere. For instance, we can do this in the case of Monotone Convergence Theorem, as follows:

**Corollary:** Let $(f_n)$ be a sequence of measurable functions such that $f_n \geq 0$ and $f_n \leq f_{n+1}$ hold $\mu$-a.e. for each $n$. Suppose that $f$ is a measurable function such that $f_n \to f$ $\mu$-a.e. Then

$$
\int_X f \, d\mu = \lim_{n \to \infty} \int_X f_n \, d\mu.
$$

**Proof:**

By countable additivity of $\mu$, the set

$$
S = \bigcup_{n=1}^\infty \{ f_n < 0 \} \cup \bigcup_{n=1}^\infty \{ f_n < f_{n+1} \} \cup \left\{ f_n \nrightarrow f \right\}
$$

is of measure $0$. Thus, if we define

$$
\tilde{f}_n = f_n \chi_{X \setminus S}, \quad \tilde{f} = f \chi_{X \setminus S},
$$

then

$$
0 \leq \tilde{f}_1 \leq \tilde{f}_2 \leq \dots \leq \tilde{f}_n \leq \dots
$$

and

$$
\tilde{f}_n \to \tilde{f} \quad \text{on all of } X.
$$

By the **Monotone Convergence Theorem**, we have

$$
\int_X f \, d\mu = \int_X \tilde{f} \, d\mu = \lim_{n \to \infty} \int_X \tilde{f}_n \, d\mu = \lim_{n \to \infty} \int_X f_n \, d\mu.
$$

$\square$

## Fatou's Lemma

**Corollary (Fatou's Lemma):** Let $ (f_n) $ be a sequence of measurable functions $ f_n : X \to [0, \infty) $ for each $ n \in \mathbb{N} $. Then

$$
\int_X \liminf_{n \to \infty} f_n \, d\mu \leq \liminf_{n \to \infty} \int_X f_n \, d\mu.
$$

**Proof:**

Define, for each $ k \in \mathbb{N} $ and for all $ x \in X $,

$$
g_k(x) = \inf_{n \geq k} f_n(x).
$$

That is, for each $ x \in X $,

$$
g_k(x) = \inf \{ f_k(x), f_{k+1}(x), \dots \}.
$$

Since $ g_k \leq f_k $ for all $ k $, it follows that

$$
\int_X g_k \, d\mu \leq \int_X f_k \, d\mu.
$$

Taking the infimum over all $ k \geq n $, we have

$$
\inf_{k \geq n} \int_X g_k \, d\mu \leq \inf_{k \geq n} \int_X f_k \, d\mu,
$$

which implies

$$
\sup_n \left( \inf_{k \geq n} \int_X g_k \, d\mu \right) \leq \sup_n \left( \inf_{k \geq n} \int_X f_k \, d\mu \right).
$$

Therefore,

$$
\liminf_{k \to \infty} \int_X g_k \, d\mu \leq \liminf_{k \to \infty} \int_X f_k \, d\mu. \quad \text{(1)}
$$

By **MCT**, the sequence $ (g_k) $ consists of measurable functions. Moreover, since

$$
\inf \{ f_k(x), f_{k+1}(x), \dots \} \leq \inf \{ f_{k+1}(x), f_{k+2}(x), \dots \} \leq \dots,
$$

the sequence $ (g_k) $ is non-decreasing. Applying the **Monotone Convergence Theorem** to the increasing sequence $ (g_k) $, we obtain

$$
\lim_{k \to \infty} \int_X g_k \, d\mu = \int_X \lim_{k \to \infty} g_k \, d\mu. \quad \text{(2)}
$$

Since $ g_k $ is non-decreasing and converges to $ \liminf_{n \to \infty} f_n $, we have

$$
\lim_{k \to \infty} g_k(x) = \liminf_{n \to \infty} f_n(x) \quad \text{for all } x \in X.
$$

Thus,

$$
\int_X \liminf_{n \to \infty} f_n \, d\mu = \int_X \lim_{k \to \infty} g_k \, d\mu.
$$

From equations (1) and (2), we conclude that

$$
\int_X \liminf_{n \to \infty} f_n \, d\mu \leq \liminf_{n \to \infty} \int_X f_n \, d\mu.
$$

$\square$

**Note:** The inequality in Fatou's Lemma can be strict. For instance, consider the following example:

Let $ X = \mathbb{N} $, let $ \mu $ be the counting measure on $ \mathbb{N} $, and define $ f_n = \chi_{\{n\}} $ for each $ n \in \mathbb{N} $. Then,

$$
\int_X \liminf_{n \to \infty} f_n \, d\mu = \int_X 0 \, d\mu = 0,
$$

$$
\liminf_{n \to \infty} \int_X f_n \, d\mu = \liminf_{n \to \infty} 1 = 1.
$$

Thus,

$$
\int_X \liminf_{n \to \infty} f_n \, d\mu < \liminf_{n \to \infty} \int_X f_n \, d\mu.
$$

## Lebesgue Dominated Convergence Theorem

**Theorem (Lebesgue Dominated Convergence Theorem):** Let $ (f_n) $ be a sequence of measurable functions $ f_n : X \to \mathbb{R} $ that converge to $ f $ pointwise almost everywhere (μ-a.e.) on $ X $. Suppose there exists an integrable function $ g : X \to [0, \infty) $ such that

$$
|f_n(x)| \leq g(x) \quad \text{for all } n \in \mathbb{N} \text{ and } x \in X.
$$

Then,

$$
\lim_{n \to \infty} \int_X f_n \, d\mu = \int_X f \, d\mu.
$$

**Proof:**

Since $ g \pm f_n \geq 0 $ for all $ n $, we can apply Fatou’s Lemma to the sequences $ g - f_n $ and $ g + f_n $. By Fatou’s Lemma,

$$
\int_X g \, d\mu - \limsup_{n \to \infty} \int_X f_n \, d\mu = \liminf_{n \to \infty} \int_X (g - f_n) \, d\mu \geq \int_X \liminf_{n \to \infty} (g - f_n) \, d\mu = \int_X g \, d\mu - \int_X f \, d\mu,
$$

and

$$
\int_X g \, d\mu + \liminf_{n \to \infty} \int_X f_n \, d\mu = \liminf_{n \to \infty} \int_X (g + f_n) \, d\mu \geq \int_X \liminf_{n \to \infty} (g + f_n) \, d\mu = \int_X g \, d\mu + \int_X f \, d\mu.
$$

Rearranging the inequalities, we obtain

$$
\limsup_{n \to \infty} \int_X f_n \, d\mu \leq \int_X f \, d\mu \leq \liminf_{n \to \infty} \int_X f_n \, d\mu.
$$

Since $ \limsup $ is always greater than or equal to $ \liminf $, the only possibility is that all three expressions are equal. Therefore,

$$
\lim_{n \to \infty} \int_X f_n \, d\mu = \int_X f \, d\mu.
$$

$\square$

**Corollary:** Let $ (f_n) $ be a sequence of measurable functions $ f_n : X \to \mathbb{R} $ that converge to $ f $ μ-a.e. Suppose there exists an integrable function $ g : X \to [0, \infty) $ such that

$$
|f_n(x)| \leq g(x) \quad \text{μ-a.e. for all } n \in \mathbb{N}.
$$

Then,

$$
\lim_{n \to \infty} \int_X f_n \, d\mu = \int_X f \, d\mu.
$$

**Corollary (Bounded Convergence Theorem):** Let $ (X, \Sigma, \mu) $ be a finite measure space (i.e., $ \mu(X) < \infty $). Let $ (f_n) $ be a sequence of measurable functions such that there exists a constant $ M > 0 $ satisfying

$$
|f_n(x)| \leq M \quad \text{for all } n \in \mathbb{N} \text{ and } x \in X.
$$

If $ f_n \to f $ μ-a.e. on $ X $, then

$$
\lim_{n \to \infty} \int_X f_n \, d\mu = \int_X f \, d\mu.
$$

**Proof:**

Apply the **Lebesgue Dominated Convergence Theorem** with the dominating function $ g(x) = M $. Since $ \mu(X) < \infty $ and $ M $ is a constant,

$$
\int_X |g| \, d\mu = M \mu(X) < \infty,
$$

which means $ g $ is integrable. Therefore, the conditions of the Dominated Convergence Theorem are satisfied, and we conclude that

$$
\lim_{n \to \infty} \int_X f_n \, d\mu = \int_X f \, d\mu.
$$

$\square$

## Convergence in Measure

**Definition:** Let $ (f_n) $ be a sequence of measurable functions. We say that $ (f_n) $ **converges in measure** to a measurable function $ f $ if for any $ \varepsilon > 0 $, we have

$$
\lim_{n \to \infty} \mu\{ |f_n - f| \geq \varepsilon \} = 0.
$$

**Example:**

Consider the space $ X = \mathbb{N} $ with $ \mu $ as the counting measure. Let $ f_n = \chi_{\{n\}} $, the characteristic function of the singleton set $ \{n\} $. Then $ (f_n) $ is a sequence of measurable functions converging pointwise to zero, but it does **not** converge in measure to any function $ f $.

To see this, note that:
+ If $ |f(m)| \geq \varepsilon $ for some $ m \in \mathbb{N} $ and $ \varepsilon > 0 $, then

  $$
  \mu\{ |f_n - f| \geq \varepsilon \} \geq 1
  $$

  for all $ n > m $.
+ Otherwise, $ f \equiv 0 $, so

  $$
  \mu\{ |f_n - f| \geq 1 \} \geq 1
  $$
  
  for all $ n $.

In both cases, the measure does not tend to zero as $ n \to \infty $, hence $ (f_n) $ does not converge in measure to any function $ f $.

**Proposition 1:** Let $ f $ and $ g $ be measurable functions. Suppose that $ (f_n) $ is a sequence of measurable functions converging in measure to $ f $. Then $ (f_n) $ also converges in measure to $ g $ if and only if $ f = g $ $ \mu $-a.e.

**Proof:**

(⇒) **Assume** $ (f_n) $ converges in measure to both $ f $ and $ g $. For any $ \varepsilon > 0 $, by the triangle inequality,

$$
|f - g| \leq |f - f_n| + |f_n - g|.
$$

Thus,

$$
\{ |f - g| \geq \varepsilon \} \subseteq \{ |f - f_n| \geq \varepsilon/2 \} \cup \{ |f_n - g| \geq \varepsilon/2 \}.
$$

Taking measures on both sides,

$$
\mu\{ |f - g| \geq \varepsilon \} \leq \mu\{ |f - f_n| \geq \varepsilon/2 \} + \mu\{ |f_n - g| \geq \varepsilon/2 \}.
$$

As $n \to \infty$, both $\mu\{ |f - f_n| \geq \varepsilon/2 \}$ and $\mu\{ |f_n - g| \geq \varepsilon/2 \}$ tend to zero. Therefore,

$$
\mu\{ |f - g| \geq \varepsilon \} = 0 \quad \text{for all } \varepsilon > 0,
$$

which implies $ f = g $ $ \mu $-a.e.

(⇐) **Assume** $ f = g $ $ \mu $-a.e. Then,

$$
\{ |f_n - g| \geq \varepsilon \} \subseteq \{ |f_n - f| \geq \varepsilon \} \cup \{ |f - g| \geq \varepsilon \}.
$$

Since $ f = g $ $ \mu $-a.e., $ \mu\{ |f - g| \geq \varepsilon \} = 0 $. Hence,

$$
\mu\{ |f_n - g| \geq \varepsilon \} \leq \mu\{ |f_n - f| \geq \varepsilon \}.
$$

As $n \to \infty$, $ \mu\{ |f_n - f| \geq \varepsilon \} \to 0 $, so $ \mu\{ |f_n - g| \geq \varepsilon \} \to 0 $. Therefore, $ (f_n) $ converges in measure to $ g $.

$\square$

**Proposition 2:** Let $ (f_n) $ be a sequence of measurable functions converging in measure to a measurable function $ f $. Then there exists a subsequence $ (f_{n_k}) $ that converges pointwise to $ f $ $ \mu $-a.e.

**Proof:**

For each $ k \in \mathbb{N} $, choose $ n_k $ such that

$$
\mu\{ |f_{n_k} - f| > 1/(2k) \} < 1/(2k).
$$

Let $ S_k = \{ |f_{n_k} - f| > 1/(2k) \} $. Then,

$$
\mu(S_k) < \frac{1}{2k}.
$$

Consider the limsup set:

$$
S = \bigcap_{m=1}^\infty \bigcup_{k \geq m} S_k.
$$

Using the **subadditivity** of measures,

$$
\mu\left( \bigcup_{k \geq m} S_k \right) \leq \sum_{k \geq m} \mu(S_k) \leq \sum_{k \geq m} \frac{1}{2k} = \frac{1}{2} \sum_{k \geq m} \frac{1}{k} \leq \frac{1}{2(m - 1)} \quad \text{for } m \geq 2.
$$

As $ m \to \infty $,

$$
\mu(S) \leq \lim_{m \to \infty} \frac{1}{2(m - 1)} = 0.
$$

Thus, $ \mu(S) = 0 $. For $ \omega \in X \setminus S $, there exists $ N \geq 1 $ such that $ \omega \notin S_k $ for all $ k \geq N $. Hence,

$$
|f_{n_k}(\omega) - f(\omega)| \leq \frac{1}{2k} \quad \text{for all } k \geq N.
$$

As $ k \to \infty $, $ |f_{n_k}(\omega) - f(\omega)| \to 0 $. Therefore, $ f_{n_k}(\omega) \to f(\omega) $ for all $ \omega \in X \setminus S $, which has measure 1.

$\square$

**Extension of the Lebesgue Dominated Convergence Theorem (Convergence in Measure):**

**Theorem:** Let $ (f_n) $ be a sequence of measurable functions converging in measure to a measurable function $ f $. Suppose that there exists an integrable function $ g $ such that

$$
|f_n| \leq g \quad \text{for all } n \in \mathbb{N}.
$$

Then,

$$
\lim_{n \to \infty} \int_X f_n \, d\mu = \int_X f \, d\mu.
$$

**Proof:**

Recall a fundamental result from real analysis: a sequence of real numbers $ (a_n) $ converges to $ L $ if and only if every subsequence $ (a_{n_k}) $ has a further subsequence $ (a_{n_{k_l}}) $ converging to $ L $.

Consider any subsequence $ (f_{n_k}) $ of $ (f_n) $. Since $ (f_n) $ converges in measure to $ f $, so does $ (f_{n_k}) $. By **Proposition 2**, there exists a further subsequence $ (f_{n_{k_l}}) $ that converges pointwise to $ f $ $ \mu $-a.e.

On this subsequence, by the **Lebesgue Dominated Convergence Theorem**, we have

$$
\lim_{l \to \infty} \int_X f_{n_{k_l}} \, d\mu = \int_X f \, d\mu.
$$

Therefore, every subsequence $ \left( \int_X f_{n_k} \, d\mu \right) $ has a further subsequence converging to $ \int_X f \, d\mu $. This implies that the entire sequence $ \left( \int_X f_n \, d\mu \right) $ must converge to $ \int_X f \, d\mu $.

$\square$


**Corollary (Bounded Convergence Theorem):** Let $ (X, \Sigma, \mu) $ be a finite measure space (i.e., $ \mu(X) < \infty $). Let $ (f_n) $ be a sequence of measurable functions such that there exists a constant $ M > 0 $ satisfying

$$
|f_n(x)| \leq M \quad \text{for all } n \in \mathbb{N} \text{ and } x \in X.
$$

If $ f_n \to f $ $ \mu $-a.e. on $ X $, then

$$
\lim_{n \to \infty} \int_X f_n \, d\mu = \int_X f \, d\mu.
$$

**Proof:**

Apply the **Lebesgue Dominated Convergence Theorem** with the dominating function $ g(x) = M $. Since $ \mu(X) < \infty $ and $ M $ is a constant,

$$
\int_X |g| \, d\mu = M \mu(X) < \infty,
$$

which means $ g $ is integrable. Therefore, the conditions of the Dominated Convergence Theorem are satisfied, and we conclude that

$$
\lim_{n \to \infty} \int_X f_n \, d\mu = \int_X f \, d\mu.
$$

$\square$






