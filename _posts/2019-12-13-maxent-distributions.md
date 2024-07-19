---
layout: post
title: "Maximum entropy distributions"
author: "Binh Ho"
categories: Statistics
blurb: "Maximum entropy distributions are those that are the 'least informative' (i.e., have the greatest entropy) among a class of distributions with certain constraints. The principle of maximum entropy has roots across information theory, statistical mechanics, Bayesian probability, and philosophy. For this post, we'll focus on the simple definition of maximum entropy distributions."
img: ""
tags: []
<!-- image: -->
---


Maximum entropy distributions are those that are the "least informative" (i.e., have the greatest entropy) among a class of distributions with certain constraints. The principle of maximum entropy has roots across information theory, statistical mechanics, Bayesian probability, and philosophy. For this post, we'll focus on the simple definition of maximum entropy distributions.

## Definition

Recall the definition of entropy for $n$ bits, where bit $i$ has probability $p_i$ of being $1$:

$$H(X) = -\sum\limits_{i = 1}^n p_i \log p_i.$$

This can be generalized to the continuous case:

$$H(X) = -\int p(x) \log p(x).$$

It's useful to also recall that the entropy is simply the expectation of the information, which is defined as $I(X) = -\log p_i$.

\begin{align}\mathbb{E}[I(X)] &= \sum\limits_{i = 1}^n p_i I(X) \\\ &= \sum\limits_{i = 1}^n p_i \left(-\log p_i\right) \\\ &= -\sum\limits_{i = 1}^n p_i \log p_i\end{align}

As mentioned before, a maximum entropy distribution is the member of a family of distributions that maximizes the above expression for entropy. To find a particular maximum entropy distribution, one can think of it as a constrained optimization problem. Suppose you're searching within a family of distributions that has constraints $c_1, c_2, \dots, c_n$ (for example, these could be moment conditions, the support, etc.). Then the optimization problem to solve is:

\begin{align}&\max H(X) \\\ &\text{s.t. } c_1, \dots, c_n \text{ are true}.\end{align}

# Uniform distribution

Let's first consider the simplest example that places the fewest constraints on the family of distributions: the family of discrete distributions that has support on the the integers in the interval $[a, b]$. Notice that our only constraint here is that the distribution sums to $1$, so that it's a valid distribution. So the corresponding optimization problem is 

\begin{align}&\max H(X) = -\sum\limits_{i = a}^b p_i \log p_i \\\ &\text{s.t. } \sum\limits_{i = a}^b p_i = 1\end{align}

where $p_i$ is the size of the mass placed at position $i$. We'll use our old friend Lagrange multipliers to solve this. The Lagrangian is:

$$L = -\sum\limits_{i = a}^b p_i \log p_i - \lambda(\sum\limits_{i = a}^b p_i - 1)$$

The first derivatives are:

\begin{align}\frac{d}{dp_i} L &= -\log p_i + \frac{-p_i}{p_i} - \lambda = -\log p_i - 1 - \lambda \\\ \frac{d}{d\lambda} L &= -\sum\limits_{i = a}^b p_i + 1\end{align}

Setting the first equation to zero and solving, we have:

$$p_i = \exp(-\lambda - 1)$$

which, using the second equation (and setting it to zero), implies

\begin{align}-\sum\limits_{i = a}^b \exp(-\lambda - 1) + 1 &= 0 \\\ \lambda &= \log(b - a + 1) + 1\end{align}

Plugging this back into the expression for $p_i$, we get

$$p_i = \frac{1}{b - a + 1}.$$

This means we're placing equal mass on each of the points in $[a, b]$, so this is the uniform distribution! This should make intuitive sense because the uniform is maximally spread out over the interval, so it will have maximum entropy.

Finding the maximum entropy distribution for other families with more constraints amounts to adding in more constraints to the constrained optimization problem above.

# Gaussian

As another important special case, we can show that the Gaussian is the maximum entropy distribution in the family with known mean and variance. Instead of showing this from first principles (i.e., solving the constrained optimization problem), we'll show a different proof here that uses KL divergence to bound the entropy of any other distribution in the family.

Recall that the Gaussian pdf is:

$$p(x) = \frac{1}{\sqrt{2\pi \sigma^2}} \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right).$$

Using our formula for entropy above, we can show that it has entropy 

\begin{align}H(p) &= -\int_{-\infty}^\infty p(x) \log p(x) dx \\\ &= -\int_{-\infty}^\infty \frac{1}{\sqrt{2\pi \sigma^2}} \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right) \log\left[\frac{1}{\sqrt{2\pi \sigma^2}} \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)\right] \\\ &= \frac{1}{2} \log (2\pi \sigma^2) \int_{-\infty}^\infty \frac{1}{\sqrt{2\pi \sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2} \right)  \\\ &+ \frac{1}{2\sigma^2} \int_{-\infty}^\infty \frac{1}{\sqrt{2\pi \sigma^2}} (x - \mu)^2 \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right) \\\ &= \frac{1}{2}\log (2\pi \sigma^2)(1) + \frac{1}{2\sigma^2} \sigma^2 \\\ &= \frac{1}{2}\log (2\pi \sigma^2) + \frac{1}{2} \\\ &= \frac{1}{2}\log (2\pi \sigma^2) + \frac{1}{2}\log(e) \\\ &= \frac{1}{2} \log(2\pi e \sigma^2)\end{align}

where we have used the definition of the variance 

$$\sigma^2 = \int_{-\infty}^\infty \frac{1}{\sqrt{2\pi \sigma^2}} (x - \mu)^2 \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)$$

and the fact that the Normal distribution sums to 1:

$$\int_{-\infty}^\infty \frac{1}{\sqrt{2\pi \sigma^2}} \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right) = 1.$$

Now, we'd like to show that for any other distribution $q(x)$ in the family, it will have lower entropy, that is, $H(p) \geq H(q)$.

Consider the KL divergence between $p$ and $q$:

\begin{align}D_\text{KL}(q, p) &= \int_{-\infty}^\infty q(x) \log \left( \frac{q(x)}{p(x)} \right) dx \\\ &= -H(q, p) - H(q)
\end{align}

where $H(q, p) = \int q(x) \log p(x)$ is the cross-entropy. An important property of the KL-divergence is that it's always non-negative, so we know that $-H(q, p) - H(q) \geq 0 \implies -H(q, p) \geq H(q)$. So if we can show that the LHS of this inequality is equal to $H(P)$, then we've shown that $q$ has lower entropy than $p$. Writing out the cross-entropy, we have

\begin{align}H(q, p) &= \int_{-\infty}^\infty q(x) \log \left[\frac{1}{\sqrt{2\pi \sigma^2}} \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)\right] \\\ &= \mathbb{E}_{q}\left[ \log \left[\frac{1}{\sqrt{2\pi \sigma^2}} \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)\right] \right] \\\ &= -\frac{1}{2} \log(2\pi \sigma^2) - \frac{1}{2\sigma^2} \mathbb{E}_q[(x - \mu)^2].\end{align}

Now, by the definition of the variance of $q$, we know that $\mathbb{E}_q[(x - \mu)^2] = \sigma^2$, so

\begin{align}-\frac{1}{2} \log(2\pi \sigma^2) - \frac{1}{2\sigma^2} \mathbb{E}_q[(x - \mu)^2] &= -\frac{1}{2} \log(2\pi \sigma^2) - \frac{1}{2\sigma^2} \sigma^2 \\\ &= -\frac{1}{2} \log(2\pi \sigma^2) - \frac{1}{2} \\\ &= -\frac{1}{2}\left(\log(2\pi \sigma^2) + 1\right) \\\ &= -\frac{1}{2}\log(2\pi e \sigma^2) \\\ &= H(p)\end{align}

So the cross entropy is equal to the entropy of $p$, and we're done! In conclusion, we've shown that if $p(x)$ is Gaussian with mean $\mu$ and variance $\sigma^2$, then $H(p) \geq H(q)$ for any distribution $q$ with mean $\mu$ and variance $\sigma^2$. Thus, the Gaussian has maximum entropy within the family of distributions with given first and second moments.

## Conclusion

In this post, we've covered the definition of maximum entropy distributions, and we reviewed two examples: the discrete uniform distribution and the Gaussian.

## References

- McElreath, Richard. Statistical rethinking: A Bayesian course with examples in R and Stan. Chapman and Hall/CRC, 2018.
- Brian Keng's blog post on the topic: <http://bjlkeng.github.io/posts/maximum-entropy-distributions/>
- Entropy of the Gaussian: <http://www.biopsychology.org/norwich/isp/chap8.pdf>



