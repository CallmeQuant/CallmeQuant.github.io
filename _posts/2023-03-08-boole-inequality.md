---
layout: post
title: "Boole's Inequality"
author: "Binh Ho"
categories: Probability theory
blurb: "Boole's Inequality, an upperbound on the probability of occurrence of at least one of a countable number of events under the context of individual chances of each event."
img: ""
tags: []
<!-- image: -->
---

In this post, I will introduce Boole's Inequality, an upperbound on the probability of occurrence of at least one of a countable number of events under the context of individual chances of each event.  In some senses, Boole's inequality is so straightforward and often emerges as a definitely compelling inequality for any finite or countable set of events. The attractive point of this inequality is due to its weakly required assumptions, that is, Boole's inequality does not require *independence*. Hence, it is a useful methods when we are working with collection of events.

# **A short proof**
Boole's inequality can be stated formally as follows:

**Boole's inequality**. *If* $A_1, A_2, \dots, A_{n}$ *are finite events in a probability space* $\Omega$, *then*

$$ P\Bigg(\bigcup_{i=1}^n A_i\Bigg) \le \sum_{i=1}^n P(A_i) $$

*Moreover, for countable events* $A_1, A_2, \dots,$ *then,*

$$ P\Bigg(\bigcup_{i=1}^\infty A_i\Bigg) \le \sum_{i=1}^\infty P(A_i) $$

We will prove this inequality by two approaches: by mathematical induction and by measure theory

## *First approach: Mathematical induction*
Suppose that a probability space $\Omega$ contains a countable collection of events $A_1, A_2, \dots, A_{n}, \dots$, then

For the case n = 1, we have 

$$P(A_1) \leq P(A_1)$$

For the case $n=2$, it is trivial to show that

$$P(A_1 \cup A_2) \leq P(A_1) + P(A_2)$$

Since $P(A_1 \cup A_2)  = P(A_1) + P(A_2) - P(A_1 \cap A_2)$

We assume that the inequality holds for the case $n$, 

$$ P \Bigg(\bigcup_{i=1}^n A_i\Bigg) \le \sum_{i=1}^n P(A_i) $$

Then,

$$
\begin{align}
P \Bigg(\bigcup_{i=1}^{n+1} A_i\Bigg) &= P\Bigg(\bigcup_{i=1}^n A_i\Bigg) + P(A_{n+1}) - \underbrace{P\Bigg(\bigcup_{i=1}^n A_i \cap A_{n+1}\Bigg)}_{\geq 0} \quad (\text{ by the first axiom of probability}) \\
& \leq P \Bigg(\bigcup_{i=1}^n A_i\Bigg) + P(A_{n+1}) \\
&= \sum_{i=1}^n P(A_i) + P(A_{n+1}) \\
&= \sum_{i=1}^{n+1} P(A_{i}).
\end{align}
$$

Hence, by repeating the same inductive argument for $n^\prime = n+1, n+2, \dots,$, we can prove desired result.

## *Second approach: Measure theory*
Using measure-theoretic agruments, we can prove the general form of Boole's inequality for any countable collection of events in a given probability space. Assume a countable collection of events $\lbrace A_{i} \rbrace_{i=1}^{\infty}$ from a probability space $(\Omega, \mathcal{F}, P)$ (where the sample space $\Omega$ and the event space $\mathcal{F}$ are equipped with a *measure* or *set function* $P: \mathcal{F} \rightarrow \mathbb{R}, \quad P(\Omega) = 1$. From the formal definition of probability measure $P: \mathcal{F} \rightarrow [0,1]$, we have the following property:

$$P \Bigg(\bigcup_{i=1}^\infty A_i\Bigg) \le \sum_{i=1}^\infty P(A_i).$$ 

This property is called *countably additive* or $\sigma-\text{additive}$ provided that 
$\lbrace A_{i} \rbrace_{i=1}^{\infty} \subseteq \mathcal{F}$ is countable collection of *pairwise disjoint sets* - $A_{i} \cap A_{j} = \emptyset, \quad i \neq j.$
Since our countable collection of sets $A_{i}$ is not restricted to disjoint, we can create a countable collection of disjoint sets $\lbrace B_{i} \rbrace_{i=1}^{\infty}$ as follows (we suppose further that the collection of sets $\lbrace A_{i} \rbrace_{i=1}^{\infty}$ is increasing, i.e., $A_1 \supseteq A_2 \supseteq \dots$:

$$
\begin{align}
B_{1} &= A_{1}, \\
B_{2} &= A_{2} \setminus A_{1}, \\
\vdots \\
B_{i} &= A_{i} \setminus \bigcup_{j=1}^{i-1} A_j. \\
\end{align}
$$

then obviously 

$$\bigcup_{i=1}^\infty B_i = \bigcup_{i=1}^\infty A_i$$

and $B_{i}\text{'s}$ are pairwise disjoint. Furthermore, by set construction, $B_{i} \subseteq A_{i}$, which gives us the following inequality by monocity of probability measure:

$$P(B_{i}) \leq P(A_{i}), \quad i = 1, 2, \dots$$

Hence, we can derive the desired inequality 

$$P\Bigg(\bigcup_{i=1}^\infty A_i\Bigg) = P\Bigg(\bigcup_{i=1}^\infty B_i\Bigg) = \underbrace{\sum_{i=1}^{\infty} P(B_{i})}_{\text{ countably additive}} \leq \sum_{i=1}^{\infty} P(A_{i}).$$









