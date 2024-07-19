---
layout: post
title: "Lindley's paradox"
author: "Binh Ho"
categories: Statistics
blurb: "Bayesian and frequenist methods can lead people to very different conclusions. One instance of this is exemplified in Lindley's paradox, in which a hypothesis test arrives at opposite conclusions depending on whether a Bayesian or a frequentist test is used."
img: ""
tags: []
<!-- image: -->
---

Bayesian and frequenist methods can lead people to very different conclusions. One instance of this is exemplified in Lindley's paradox, in which a hypothesis test arrives at opposite conclusions depending on whether a Bayesian or a frequentist test is used.

## Setup

Consider the canonical statistical toy problem: testing whether a coin is fair. Specifically, define a coin as "fair" if, after flipping it randomly, it shows heads half of the time and tails half of the time.

To test the coin, suppose we it $n$ times and record the number of times that the coin showed tails (call this number $k$). Let $\hat{p} = k/p$ denote the observed fraction of coin flips that showed heads in our sample.

Our null and alternative hypotheses are then

\begin{align} &H_0: p = 1/2 \\\ &H_1: p \neq 1/2. \\\ \end{align}

We will test this hypothesis in both a frequentist and a Bayesian framework.

## Frequentist

In the frequentist hypothesis setting, we take a "sampling" view of the world. Imagine we had a fair coin, and we could rerun the experiment many times. That is, on round $1$, we would flip the fair coin $n$ times and record $r_1$, the fraction flips that wre tails; on round $2$, we would again flip the coin $n$ times and record $r_2$; and this would continue for $M$ rounds, where $M$ is a large number.

Then, if we look at all the data $r_1, r_2, \dots, r_M$ that we sampled from our $M$ experiments, we can count how many of them had a $r$ value larger than our original experiment's fraction $\hat{p}$. That is, we would calculate

$$\omega = \frac{1}{M} \sum\limits_{i=1}^M I(r_i > \hat{p})$$

where $I$ is the indicator function. $\omega$ is then our p-value, indicating the fraction of times we would have observed a fraction of tails larger than $p$ if we reran the experiment many times, under the null hypothesis.

We can easily run this experiment in Python. Suppose our coin is actually slightly biased such that the true fraction of tails it produces is $p^* = 0.504$, and suppose we flip this coin $n = 100,000$ times. This is how it's done in Python:

```python
p_true = 0.504
n = 100000
X = np.random.binomial(n=1, p=p_true, size=n)
X_sum = np.sum(X)
p_observed = X_sum / n
```


Now, let's run a sampling experiment with a fair coin $M = 1,000$ times. In Python:

```python
# Rerun this experiment M times repeatedly where p=0.5 and see how many achieve mean >= X_bar
M = 1000
Xsims = []
for ii in range(M):
    Xsim = np.random.binomial(n=1, p=0.5, size=n)
    Xsims.append(np.sum(Xsim))
print("Frequentist p-value: {}".format(np.mean(Xsims > X_sum)))
```

The result will depend on the sampled data, but for me the p-value was $\omega = 0.01$, a level of fairly decent significance in most communities.

## Bayesian

Now, let's consider testing the same hypothesis in a Bayesian setting. Here, we're interested in the posterior probability of the null hypothesis, given the data, which can be computed from Bayes' rule:

$$p(H_0 | k) = \frac{p(k | H_0) p(H_0)}{\sum\limits_{i \in \{0, 1\}} p(k | H_i) p(H_i)}.$$

Let's place equal prior weight on each hypothesis for now, such that $p(H_0) = p(H_1) = 0.5$. This leaves us to compute the likelihood of the data under each hypothesis, $p(k \| H_0)$ and $p(k \| H_1)$.

Under $H_0$, we can simply plug in our value of $k$ to the binomial PMF where the rate $p = 0.5$:

$$p(k | H_0) = {n \choose k} 0.5^k (1 - 0.5)^{n-k}.$$

Under $H_1$, where $p \neq 0.5$, we must average over all these values of $p$, since we're not just testing a single value. This requires us to place a prior over the possible values of $p$. In this case, suppose we place a uniform prior on $p$ under $H_1$ ($p(p) = 1$). Then the likelihood under $H_1$ is

\begin{align} p(k \| H_1) &= \int_0^1 \underbrace{p(k; p)}\_{\text{binomial}} \underbrace{p(p)}\_{1} dp \\\ &= \int_0^1 {n \choose k} p^k (1-p)^{n-k} dp \\\ &= \{n \choose k\} \int_0^1 p^k (1-p)^{n-k} dp \\\ \end{align}

Solving the integral using integration by parts, we have

\begin{align} p(k \| H_1) &= {n \choose k} \int_0^1 p^k (1-p)^{n-k} dp \\\ &= {n \choose k} \frac{k! (n - k)!}{(n + 1)!} \\\ &= \frac{n!}{k!(n - k)!} \frac{k! (n - k)!}{(n + 1)!} \\\ &= \frac{1}{n + 1}. \\\ \end{align}

Now that we have simplified expressions for all of the elements of the posterior, let's put them together.

\begin{align} p(H_0 \| k) &= \frac{p(k \| H_0) p(H_0)}{\sum\limits_{i \in \{0, 1\}} p(k \| H_i) p(H_i)} \\\ &= \frac{\{n \choose k\} 0.5^k (1 - 0.5)^{n-k} (\frac{1}{2})}{\{n \choose k\} 0.5^k (1 - 0.5)^{n-k} (\frac{1}{2}) + \frac{1}{n + 1} (\frac12)}. \end{align}

## Lindley's paradox

Suppose we observe a sample of $n = 100,000$ coin flips, again with a coin that is very slightly biased ($p^* = 0.505$). Suppose $50,340$ of these flips come up as tails.

Now, let's compute the relevant values for the hypothesis tests in both frameworks.

In the frequentist (sampling-based) framework, this results in a p-value of $0.021$, a level of decent significance in most situations. Thus, we can be fairly confident in rejection $H_0$, and we can say the coin is biased.

In the Bayesian framework, we can compute the posterior probability of $H_0$ being true. The expression looks a little nasty, but all we're doing is plugging in $n = 100,000$ and $k = 50,340$.

\begin{align} p(H_0 \| 50,340) &= \frac{\{100,000 \choose 50,340\} 0.5^{50,340} (1 - 0.5)^{n-50,340} (\frac{1}{2})}{\{100,000 \choose 50,340\} 0.5^{50,340} (1 - 0.5)^{n-50,340} (\frac{1}{2}) + \frac{1}{100,000 + 1} (\frac12)} \\\ &= 0.96. \end{align}

So in the Bayesian framework, we would be very confident that $H_0$ holds.

The frequentist and Bayesian frameworks then disagree on how to conclude with this experiment.

## Conclusion

Although the Bayesian and frequentist viewpoints agree in many situations, it's possible to construct/encounter situations where they lead to diametrically opposed conclusions. 

## References

- Wikipedia article on [Lindley's paradox](https://www.wikiwand.com/en/Lindley%27s_paradox)
- Shafer, Glenn. "Lindley's paradox." Journal of the American Statistical Association 77.378 (1982): 325-334.
