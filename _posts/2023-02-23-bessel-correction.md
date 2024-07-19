---
layout: post
title: "Bessel's Correction - Why Sample Variance Should Be Divided By N-1"
author: "Binh Ho"
categories: Statistics
blurb: "Bessel's Correction is a formula of an unbiased estimator for the population variance."
img: ""
tags: []
<!-- image: -->
---

In this post, I would display a brief and short proof of the Bessel's Correction, which is a formula of an unbiased estimator for the population variance. After this post, readers will comprehend the reason behind the denominator in the sample variance formula (which is $N-1$ rather than $N$). 

## **Introduction**
In every introductory statistics course, students learn how to compute the sample moments (usually the fisrt and second moments or the mean and variance respectively). The sample mean is familiar to derive. Let us present some notations using for this post. Suppose that $X = \\{X_1, X_2, \dots, X_{n} \\}$ be a sample of $n$ i.i.d random variabels where i.i.d stands for identically and independently distributed. The sample mean $\bar{X}$ is calculated as

$$ \bar{X} = \frac{1}{n} \sum_{i = 1}^{n} X_{i}. \tag{1} $$ 

Concerning the sample variance, we are often told to divide the sum of the squared difference between the realized values of each random variables and the sample mean by $n-1$ instead of $n$.

$$ s^2 = \frac{1}{n-1} \sum_{i = 1}^{n} (X_{i} - \bar{x}). \tag{2}$$

The above rectification is sometimes called Bessel's correction (this is a jargon!). It can be easily shown by computational experiment that if repeat multiple times the simulation of the sample variance using Bessel's correction, it will be approximately idential to the population variance (of course since we are conducting experiment, it is reasonable to assume that the population variance is known). Hence, the idea of the proof revolves around the unbiased property of an estimator. 

We will give a formal definition of unbiased estimator:

**Definition (Unbiased estimator)** Given an estimator $\hat{\theta}$ of a parameter $\theta$, the quantity $\operatorname{Bias} [\hat{\theta}] = \mathbb{E}[\hat{\theta}] - \theta$ is the *bias* of the estimator $\hat{\theta}$. If the bias is zero, i.e., $\mathbb{E}[\hat{\theta}] = 0$ the estimator $\hat{\theta}$ is unbiased. 

To put it simply and specific to our case, let $s^2$ and $\sigma^2$ be the sample variance and population variance respectively. An unbiased estimator of the population variance must statisfy the following condition:

$$ \mathbb{E} [s^2] - \sigma^2 = 0 \Leftrightarrow \mathbb{E} [s^2] = \sigma^2. \tag{3} $$

Now, we will commence with the proof of the Bessel's correction.

## **A Brief Proof**
First, I advocate the examination of the *biasedness* of the below estimator for the population variance:

$$ s^2 = \frac{1}{n} \sum_{i=1}^{n} (X_{i} - \bar{X})^2. \tag{4} $$

Taking the expection of the RHS and do some manipulations:

$$
\begin{align*}
\mathbb{E} \Bigg[\frac{1}{n} \sum_{i=1}^{n} (X_{i} - \bar{X})^2 \Bigg] &= \mathbb{E} \Bigg[ \frac{1}{n} \sum_{i=1}^{n} (X_{i}^2 - 2 X_{i} \bar{X} + \bar{X}^2) \Bigg] \\
&= \mathbb{E} \Bigg[ \frac{1}{n} \sum_{i=1}^{n} X_{i}^2 - 2 \bar{X} \frac{1}{n} \sum_{i=1}^{n} X_{i} + \frac{1}{n} \sum_{i=1}^{n} \bar{X}^2 \Bigg] \\
&= \mathbb{E} \Bigg[\frac{1}{n} \sum_{i=1}^{n} X_{i}^2 \Bigg] - \mathbb{E} [2\bar{X}^2] + \mathbb{E}[\bar{X}^2] \tag{5} \\
&= \mathbb{E} \Bigg[\frac{1}{n} \sum_{i=1}^{n} X_{i}^2 \Bigg] - \mathbb{E}[\bar{X}^2] \\
&= \mathbb{E}[X_{i}^2] - \mathbb{E}[\bar{X}^2]. \tag{6}
\end{align*}
$$

where the equation at step (5) is based on the fact that

$$  \sum_{i=1}^{n} X_{i} = n \bar{X}. \tag{7}$$

and the equality at the last step (6) holds as the assumption our random variables $X_{i}$'s are i.i.d 

$$ \mathbb{E}  \Bigg[\frac{1}{n} \sum_{i=1}^{n} X_{i}^2 \Bigg] = \frac{1}{n} \sum_{i=1}^{n} \mathbb{E} [X_{i}^2] = \mathbb{E}[X_{i}^2]. \tag{8} $$

At this stage, if we can identify each term in the equation (6), we could determine the final expression for the targeted quantity. Before diving into any details, it is critically important to bear in mind that since we assume the data are withdrawn from the same probability distribution then all random variables $X_{i}$'s are equidispersed, i.e., they share the same variance. Moreover, recall this useful identity that illustrates the relation between variance and expectation. For any random variable $X$,

$$ \operatorname{Var}[X] = \mathbb{E}[X^2] - \mathbb{E}[X]^2, \tag{9}$$

or we can rewrite it as follows,

$$ \mathbb{E}[X^2] = \operatorname{Var}[X] + \mathbb{E}[X]^2. \tag{10}$$

Then, assume that we know the population mean $\mu$, we have

$$
\begin{align*}
\mathbb{E}[X_{n}^2] &= \operatorname{Var}[X_{n}] + \mathbb{E}[X_{n}]^2 \\
&= \sigma^2 + \mu^2, 
\end{align*}
$$

and 

$$
\begin{align*}
\mathbb{E}[\bar{X}^2] &= \operatorname{Var}[\bar{X}] + \mathbb{E}[\bar{X}]^2 \\
&= \frac{\sigma^2}{n} + \mu^2. 
\end{align*}
$$

where 

$$
\begin{align*}
\operatorname{Var} [\bar{X}] &= \operatorname{Var} \Bigg[\frac{1}{n} \sum_{i=1}^{n} X_{i} \Bigg] \\
&= \frac{1}{n^2} \sum_{i=1}^{n} \operatorname{Var}[X_{i}] \quad (\text{ i.i.d assumption}) \\
&= \frac{1}{n^2} \sum_{i=1}^{n} \sigma^2 \tag{11} \\
&= \frac{1}{n^2} n \sigma^2 = \frac{\sigma^2}{n}.
\end{align*}
$$

As we have find out all the necessary terms, the expectation of the sample variance $s^2$ would be 

$$
\begin{align*}
\mathbb{E}[s^2] &= \sigma^2 + \mu^2 - \Bigg( \frac{\sigma^2}{n} + \mu^2 \Bigg) \\
&= \sigma^2 \Bigg(1 - \frac{1}{n} \Bigg). \tag{12}
\end{align*}
$$

The term $(1 - \frac{1}{n})$ is the major component resulting in the biasedness of the sample variance. If $n$ goes to infinity (i.e., we have a large enough sample), then this term will shrink to one and $\mathbb{E}[s^2] = \sigma^2$. However, in reality we often don't obtain such sample and we demand an unbiased estimator. That is the reason why we should multiply both sides of equation (12) by $(1 - \frac{1}{n}) = (\frac{n-1}{n})$ to obtain

$$
\begin{align*}
\mathbb{E} \Bigg[ \Bigg(\frac{n}{n-1}\Bigg) s^2 \Bigg] &= \mathbb{E} \Bigg[\frac{1}{n-1} \sum_{i=1}^{n} (X_{i} - \bar{X})^2 \Bigg] \\
&= \sigma^2. \tag{13}
\end{align*}
$$

Hence, Bessel's correction leads to the desired quantity to arrive at the unbiased estimator for population variance, which is $\frac{1}{n-1} \sum_{i=1}^{n} (X_{i} - \bar{X})^2$.


