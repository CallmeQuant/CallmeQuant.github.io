---
title: "Monotone convergence, Fatou's lemma and dominated convergence"
date: 2024-09-29
topic: Probability
summary: "The three limit theorems of Lebesgue integration with complete proofs, examples showing why each hypothesis is needed, and the extension of dominated convergence to sequences that converge only in measure."
---

Pointwise convergence $$f_n \to f$$ does not imply $$\int f_n \, \dd\mu \to \int f \, \dd\mu$$. On $$(0, 1]$$ with Lebesgue measure, $$f_n = n\mathbf 1_{(0, 1/n]}$$ converges to zero at every point while $$\int f_n = 1$$ for all $$n$$; on $$\R$$, $$f_n = \mathbf 1_{[n, n+1)}$$ does the same. In the first case the mass escapes upwards into a vanishing set, in the second it escapes to infinity (Figure 1). The convergence theorems of Lebesgue integration say exactly when such escape is impossible: monotonicity prevents it, nonnegativity limits it to a loss, and domination by a single integrable function rules it out. In probability they are the standard tools for interchanging a limit and an expectation.

## Setting

Throughout, $$(X, \mathcal A, \mu)$$ is a measure space. A nonnegative simple function is $$s = \sum_{i=1}^m a_i\mathbf 1_{A_i}$$ with $$a_i \in [0, \infty)$$ and $$A_1, \dots, A_m \in \mathcal A$$ forming a partition of $$X$$ (some $$a_i$$ may be zero); its integral is $$\int s \, \dd\mu = \sum_i a_i\mu(A_i)$$. For a measurable $$f : X \to [0, \infty]$$,

$$
\int f \, \dd\mu = \sup\Bigl\{\int s \, \dd\mu : s \text{ simple},\ 0 \le s \le f\Bigr\} \in [0, \infty].
$$

The integral is monotone: $$0 \le f \le g$$ implies $$\int f \le \int g$$. A measurable $$f : X \to \R$$ is integrable if $$\int\abs f < \infty$$, and then $$\int f = \int f^+ - \int f^-$$. One more fact is needed: for a nonnegative simple $$s$$, the set function $$\nu(A) = \int s\mathbf 1_A \, \dd\mu = \sum_i a_i\mu(A_i \cap A)$$ is a measure on $$\mathcal A$$, as a finite nonnegative combination of measures.

A property holds $$\mu$$-almost everywhere (a.e.) if it fails only on a set contained in a measurable set of measure zero.

<div class="result" markdown="1">
**Lemma 1.** If $$f, g : X \to [0, \infty]$$ are measurable and $$f = g$$ a.e., then $$\int f \, \dd\mu = \int g \, \dd\mu$$. The same holds for integrable real-valued $$f, g$$.
</div>

<div class="proof" markdown="1">
*Proof.* Let $$N \in \mathcal A$$ with $$\mu(N) = 0$$ and $$f = g$$ off $$N$$. If $$s = \sum_i a_i\mathbf 1_{A_i}$$ is simple with $$0 \le s \le f$$, then $$s\mathbf 1_{N^c}$$ is simple, satisfies $$s\mathbf 1_{N^c} \le g$$, and $$\int s\mathbf 1_{N^c} = \sum_i a_i\mu(A_i \cap N^c) = \int s$$. Taking the supremum over $$s$$ gives $$\int f \le \int g$$, and symmetry gives equality. The real-valued case follows by applying this to positive and negative parts.
</div>

## Monotone convergence

<div class="result" markdown="1">
**Theorem 2 (Monotone convergence).** Let $$f_n : X \to [0, \infty]$$ be measurable with $$f_1 \le f_2 \le \cdots$$ pointwise, and let $$f = \lim_n f_n$$. Then $$f$$ is measurable and

$$
\lim_{n\to\infty}\int f_n \, \dd\mu = \int f \, \dd\mu .
$$

</div>

<div class="proof" markdown="1">
*Proof.* The limit exists in $$[0, \infty]$$ at each point because the sequence is nondecreasing, and $$f = \sup_n f_n$$ is measurable since $$\{f > a\} = \bigcup_n\{f_n > a\}$$ for every $$a \in \R$$. By monotonicity of the integral, $$\int f_n$$ is nondecreasing in $$n$$ and bounded by $$\int f$$, so $$\alpha = \lim_n\int f_n$$ exists and $$\alpha \le \int f$$.

For the reverse inequality fix a simple $$s = \sum_i a_i\mathbf 1_{A_i}$$ with $$0 \le s \le f$$ and a number $$c \in (0, 1)$$, and let

$$
E_n = \{x : f_n(x) \ge c\,s(x)\} = \bigcup_i \bigl(A_i \cap \{f_n \ge c\,a_i\}\bigr) \in \mathcal A .
$$

The sets $$E_n$$ increase because the $$f_n$$ do, and their union is $$X$$: if $$s(x) = 0$$ then $$x \in E_1$$, and if $$s(x) > 0$$ then $$c\,s(x) < s(x) \le f(x) = \lim_n f_n(x)$$, so $$f_n(x) \ge c\,s(x)$$ for large $$n$$. Since $$f_n \ge f_n\mathbf 1_{E_n} \ge c\,s\mathbf 1_{E_n}$$,

$$
\int f_n \, \dd\mu \ge c\int s\mathbf 1_{E_n} \, \dd\mu = c\,\nu(E_n),
$$

where $$\nu(A) = \int s\mathbf 1_A \, \dd\mu$$ is the measure introduced above. Continuity of $$\nu$$ from below gives $$\nu(E_n) \to \nu(X) = \int s$$, hence $$\alpha \ge c\int s$$. Letting $$c \uparrow 1$$ and taking the supremum over $$s$$ gives $$\alpha \ge \int f$$.
</div>

The factor $$c < 1$$ is what makes the sets $$E_n$$ exhaust $$X$$: with $$c = 1$$, a point where $$f_n(x) \uparrow f(x) = s(x)$$ strictly from below would belong to no $$E_n$$.

Since null sets do not affect integrals, the hypotheses need only hold almost everywhere.

<div class="result" markdown="1">
**Corollary 3.** Let $$f_n, f : X \to [0, \infty]$$ be measurable, with $$f_n \le f_{n+1}$$ a.e. for each $$n$$ and $$f_n \to f$$ a.e. Then $$\int f_n \to \int f$$.
</div>

<div class="proof" markdown="1">
*Proof.* The set $$S = \bigcup_n\{f_n > f_{n+1}\} \cup \{f_n \not\to f\}$$ is measurable, because $$\{f_n \to f\} = \{\liminf_n f_n = f\} \cap \{\limsup_n f_n = f\}$$ and $$\liminf_n f_n$$, $$\limsup_n f_n$$ are measurable. It is a countable union of null sets, so $$\mu(S) = 0$$. The functions $$f_n\mathbf 1_{S^c}$$ increase everywhere to $$f\mathbf 1_{S^c}$$, and Theorem 2 together with Lemma 1 gives $$\int f_n = \int f_n\mathbf 1_{S^c} \to \int f\mathbf 1_{S^c} = \int f$$.
</div>

The measurability of $$f$$ is assumed here and not deduced: an a.e. limit of measurable functions need not be measurable unless the measure space is complete.

## Fatou's lemma

Without monotonicity, nonnegative functions can lose mass in the limit but cannot gain it.

<div class="result" markdown="1">
**Theorem 4 (Fatou's lemma).** If $$f_n : X \to [0, \infty]$$ are measurable, then

$$
\int \liminf_{n\to\infty} f_n \, \dd\mu \le \liminf_{n\to\infty}\int f_n \, \dd\mu .
$$

</div>

<div class="proof" markdown="1">
*Proof.* Let $$g_k = \inf_{n \ge k} f_n$$. Each $$g_k$$ is measurable, since $$\{g_k \ge a\} = \bigcap_{n \ge k}\{f_n \ge a\}$$, and $$g_k \uparrow \liminf_n f_n$$ pointwise. Because $$g_k \le f_n$$ for every $$n \ge k$$, monotonicity gives $$\int g_k \le \inf_{n \ge k}\int f_n$$. By monotone convergence,

$$
\int\liminf_n f_n \, \dd\mu = \lim_{k\to\infty}\int g_k \, \dd\mu \le \lim_{k\to\infty}\inf_{n \ge k}\int f_n \, \dd\mu = \liminf_n\int f_n \, \dd\mu .
$$

</div>

Both examples of the introduction make the inequality strict: $$\liminf_n f_n = 0$$ while $$\int f_n = 1$$. So does $$f_n = \mathbf 1_{\{n\}}$$ on $$\N$$ with counting measure. Nonnegativity cannot be dropped: $$-f_n$$ with $$f_n = n\mathbf 1_{(0, 1/n]}$$ has $$\int\liminf_n(-f_n) = 0 > -1 = \liminf_n\int(-f_n)$$.

## Dominated convergence

<div class="result" markdown="1">
**Theorem 5 (Dominated convergence).** Let $$f_n, f : X \to \R$$ be measurable with $$f_n \to f$$ a.e., and suppose there is an integrable $$g : X \to [0, \infty)$$ with $$\abs{f_n} \le g$$ a.e. for every $$n$$. Then $$f$$ is integrable and

$$
\lim_{n\to\infty}\int\abs{f_n - f} \, \dd\mu = 0, \qquad\text{in particular}\qquad \lim_{n\to\infty}\int f_n \, \dd\mu = \int f \, \dd\mu .
$$

</div>

<div class="proof" markdown="1">
*Proof.* Let $$N$$ be the union of $$\{f_n \not\to f\}$$ and the sets $$\{\abs{f_n} > g\}$$; it is a countable union of null sets. Replacing $$f_n$$, $$f$$ and $$g$$ by $$0$$ on $$N$$ changes no integral (Lemma 1), so we may assume convergence and domination everywhere. Then $$\abs f \le g$$, so $$f$$ is integrable, and $$h_n = 2g - \abs{f_n - f} \ge 0$$ with $$h_n \to 2g$$. Fatou's lemma gives

$$
\int 2g \, \dd\mu \le \liminf_n\int\bigl(2g - \abs{f_n - f}\bigr)\dd\mu = \int 2g \, \dd\mu - \limsup_n\int\abs{f_n - f} \, \dd\mu ,
$$

where splitting the integral is legitimate because $$2g$$ and $$\abs{f_n - f}$$ are integrable. Since $$\int 2g < \infty$$, it can be cancelled, leaving $$\limsup_n\int\abs{f_n - f} \le 0$$. Finally $$\abs{\int f_n - \int f} \le \int\abs{f_n - f}$$.
</div>

The dominating function must be integrable and must not depend on $$n$$. In the first example, $$\sup_n f_n(x) = \lfloor 1/x \rfloor$$ for $$x \in (0, 1]$$, which behaves like $$1/x$$ and is not integrable; in the second, $$\sup_n f_n = \mathbf 1_{[1, \infty)}$$ is not integrable on $$\R$$. Panel (c) of Figure 1 shows a sequence where domination holds: for $$x \ge 0$$ the sequence $$(1 + x/n)^n$$ increases to $$e^x$$, so for $$n \ge 2$$

$$
0 \le \Bigl(1 + \frac xn\Bigr)^{-n} \le \Bigl(1 + \frac x2\Bigr)^{-2}, \qquad \int_0^\infty\Bigl(1 + \frac x2\Bigr)^{-2}\dd x = 2 < \infty,
$$

and dominated convergence gives $$\int_0^\infty(1 + x/n)^{-n}\,\dd x \to \int_0^\infty e^{-x}\,\dd x = 1$$. Here the integrals can be computed exactly, $$\int_0^\infty(1 + x/n)^{-n}\,\dd x = n/(n-1)$$, which confirms the limit.

<figure>
  <img src="/assets/figures/three-convergence-theorems/escaping-mass.svg" alt="Three panels: rectangles of height n on (0, 1/n], unit rectangles moving to the right, and decreasing curves (1 + x/n) to the power minus n under a shaded dominating function.">
  <figcaption>Figure 1. (a) $f_n = n\mathbf{1}_{(0,1/n]}$ and (b) $f_n = \mathbf{1}_{[n,n+1)}$ converge to zero pointwise, yet every $f_n$ has integral one; no integrable function dominates them. (c) $f_n(x) = (1 + x/n)^{-n}$ for $n = 2, 4, 8$ (colours as in (a)) lies under the integrable envelope $g(x) = (1 + x/2)^{-2}$ (shaded), which is $f_2$ itself, and converges to $e^{-x}$ (dashed); the integrals $n/(n-1)$ converge to $1$.</figcaption>
</figure>

When $$\mu$$ is finite, a constant bound is an integrable dominating function, which gives the **bounded convergence theorem**: if $$\mu(X) < \infty$$, $$\abs{f_n} \le M$$ a.e. and $$f_n \to f$$ a.e., then $$\int f_n \to \int f$$. On a probability space this says $$\E X_n \to \E X$$ for uniformly bounded random variables converging almost surely.

## Convergence in measure

<div class="result" markdown="1">
**Definition 6.** Measurable functions $$f_n : X \to \R$$ **converge in measure** to a measurable $$f : X \to \R$$ if $$\mu\bigl(\{\abs{f_n - f} \ge \varepsilon\}\bigr) \to 0$$ for every $$\varepsilon > 0$$.
</div>

On a probability space this is convergence in probability. On a general measure space it is neither weaker nor stronger than a.e. convergence. With counting measure on $$\N$$, $$f_n = \mathbf 1_{\{n\}}$$ converges to $$0$$ everywhere, but $$\mu(\{\abs{f_n} \ge 1\}) = 1$$ for all $$n$$; by Proposition 8 below, any limit in measure would have to agree with the pointwise limit $$0$$ a.e., so $$(f_n)$$ does not converge in measure to anything. Conversely, the "typewriter" sequence of indicators of the dyadic intervals $$[j2^{-k}, (j+1)2^{-k})$$, $$j = 0, \dots, 2^k - 1$$, $$k = 0, 1, \dots$$, on $$[0, 1)$$ converges to $$0$$ in Lebesgue measure but converges at no point. When $$\mu(X) < \infty$$, a.e. convergence does imply convergence in measure: if $$f_n \to f$$ a.e., then $$\mathbf 1\{\abs{f_n - f} \ge \varepsilon\} \to 0$$ a.e., and bounded convergence gives $$\mu(\abs{f_n - f} \ge \varepsilon) \to 0$$.

<div class="result" markdown="1">
**Proposition 7.** If $$f_n \to f$$ in measure, then $$f_n \to g$$ in measure if and only if $$f = g$$ a.e.
</div>

<div class="proof" markdown="1">
*Proof.* If $$f_n$$ converges in measure to both, then for $$\varepsilon > 0$$ the triangle inequality gives $$\{\abs{f - g} \ge \varepsilon\} \subseteq \{\abs{f - f_n} \ge \varepsilon/2\} \cup \{\abs{f_n - g} \ge \varepsilon/2\}$$, so $$\mu(\abs{f - g} \ge \varepsilon)$$ is bounded by a quantity tending to zero and equals zero. Taking $$\varepsilon = 1/k$$ and a union over $$k$$ gives $$\mu(f \ne g) = 0$$. Conversely, if $$f = g$$ a.e., then $$\{\abs{f_n - g} \ge \varepsilon\}$$ and $$\{\abs{f_n - f} \ge \varepsilon\}$$ differ by a null set.
</div>

<div class="result" markdown="1">
**Proposition 8.** If $$f_n \to f$$ in measure, there is a subsequence $$n_1 < n_2 < \cdots$$ with $$f_{n_k} \to f$$ a.e.
</div>

<div class="proof" markdown="1">
*Proof.* Choose $$n_k > n_{k-1}$$ with $$\mu(S_k) \le 2^{-k}$$, where $$S_k = \{\abs{f_{n_k} - f} > 2^{-k}\}$$. Let $$S = \bigcap_{m \ge 1}\bigcup_{k \ge m}S_k$$, the set of points lying in infinitely many $$S_k$$. For every $$m$$, countable subadditivity gives

$$
\mu(S) \le \mu\Bigl(\bigcup_{k \ge m}S_k\Bigr) \le \sum_{k \ge m}2^{-k} = 2^{1-m},
$$

so $$\mu(S) = 0$$. If $$x \notin S$$, then $$x \notin S_k$$ for all large $$k$$, so $$\abs{f_{n_k}(x) - f(x)} \le 2^{-k}$$ eventually and $$f_{n_k}(x) \to f(x)$$.
</div>

The summability of the bounds $$\mu(S_k)$$ is essential; with bounds such as $$1/k$$, whose sum diverges, the argument gives nothing. Proposition 8 turns any statement about a.e. convergence into one about convergence in measure along subsequences, and in particular extends dominated convergence.

<div class="result" markdown="1">
**Theorem 9 (Dominated convergence in measure).** Let $$f_n \to f$$ in measure, and suppose $$\abs{f_n} \le g$$ a.e. for every $$n$$, with $$g$$ integrable. Then $$f$$ is integrable and $$\int\abs{f_n - f} \, \dd\mu \to 0$$; in particular $$\int f_n \to \int f$$.
</div>

<div class="proof" markdown="1">
*Proof.* By Proposition 8 a subsequence converges to $$f$$ a.e., so $$\abs f \le g$$ a.e. and $$f$$ is integrable. Let $$a_n = \int\abs{f_n - f}$$. A sequence of real numbers converges to $$0$$ if every subsequence has a further subsequence converging to $$0$$. Given a subsequence $$(f_{n_k})$$, it still converges to $$f$$ in measure, so by Proposition 8 a further subsequence converges to $$f$$ a.e., and along it $$a_n \to 0$$ by Theorem 5. Hence $$a_n \to 0$$.
</div>

## Remarks

On a probability space, Theorem 9 reads: if $$X_n \to X$$ in probability and $$\abs{X_n} \le Y$$ with $$\E Y < \infty$$, then $$\E\abs{X_n - X} \to 0$$. Domination is sufficient but not necessary. The sharp condition is uniform integrability, $$\lim_{M\to\infty}\sup_n\E\bigl[\abs{X_n}\mathbf 1\{\abs{X_n} > M\}\bigr] = 0$$: on a probability space, $$X_n \to X$$ in $$L^1$$ if and only if $$X_n \to X$$ in probability and $$(X_n)$$ is uniformly integrable (Vitali's convergence theorem). Example (a) of Figure 1 fails uniform integrability, since its mass sits where $$f_n$$ is large. Example (b) does not: it is bounded by $$1$$, so the truncated integrals vanish for $$M \ge 1$$. On a space of infinite measure, Vitali's theorem needs a second condition, tightness: for every $$\eta > 0$$ there is a set $$E$$ with $$\mu(E) < \infty$$ and $$\sup_n\int_{E^c}\abs{f_n}\,\dd\mu < \eta$$. Example (b) fails this condition, because its mass escapes every set of finite measure.

Standard references are Folland, *Real Analysis: Modern Techniques and Their Applications* (Chapter 2), Rudin, *Real and Complex Analysis* (Chapter 1), and, for the probabilistic versions, Durrett, *Probability: Theory and Examples*.
