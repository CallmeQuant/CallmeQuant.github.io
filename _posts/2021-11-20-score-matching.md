---
layout: post
title: "Score matching"
blurb: "Complicated likelihoods with intractable normalizing constants are commonplace in many modern machine learning methods. Score matching is an approach to fit these models which circumvents the need to approximate these intractable constants."
img: ""
author: "Binh Ho"
categories: Machine learning 
tags: []
<!-- image: -->
---

$$\DeclareMathOperator*{\argmin}{arg\,min}$$
$$\DeclareMathOperator*{\argmax}{arg\,max}$$

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

One of the most frequently occurring problems in statistical modeling is dealing with intractable normalizing constants. Outside of relatively simple or conjugate models, it is often impossible to evaluate these constants.

In modern machine learning, this problem often arises when a model's likelihood function is complicated and difficult to normalize. For example, this occurs in [energy-based models (EBMs)](https://www.cs.toronto.edu/~hinton/absps/tr00-004.pdf), [generative adversarial networks (GANs)](https://arxiv.org/abs/1406.2661), and [variational autoencoders (VAEs)](https://arxiv.org/abs/1312.6114).

Score matching is one approach to fitting these models in the face of intractable partition functions. In this post, we'll explain the details of score matching and provide a simple example.

## Problem setting

Suppose we observe a set of $n$ data samples $x_1, \dots, x_n$ whose underlying distribution is $p_d(x)$. This distribution is unknown to us, but we'd like to model it another distribution $p_m(x; \theta)$, where $\theta$ is a parameter (or parameter vector). At a high level, the goal is to find a value $\theta$ such that $p_m(x; \theta)$ resembles $p_d(x)$.

In a maximum likelihood estimation approach, we would maximize the (log) likelihood of the data with respect to $\theta$:

$$\widehat{\theta}_{MLE} = \argmax_{\theta} \log p_m(x; \theta).$$

However, in many settings this will be impossible due to an intractable normalizing constant. To see this, let's write $p_m(x; \theta)$ in terms of an unnormalized density $\widetilde{p}(x; \theta)$ and the normalizing constant:

$$p_m(x; \theta) = \frac{\widetilde{p}(x; \theta)}{Z_\theta},~~~~Z_\theta = \int_{\mathcal{X}} \widetilde{p}(x; \theta) dx.$$

Here, the integral in $Z_\theta$ is often intractable. Several approaches try to get around this problem by approximating this integral. Here, we'll explore another approach --- *score matching* --- that doesn't require working with the normalizing constants at all.

## Score matching

The basic motivation for score matching is this: Instead of directly maximizing the likelihood function, what if we try to find a $\theta$ such that the *gradient* of the model's log likelihood is approximately the same as the gradient of the data distribution's log likelihood? If we could do this, it may provide a way to only work with the unnormalized density $\widetilde{p}(x; \theta)$, rather than the normalized density $p(x; \theta)$.

Let's make this more precise. To start, let's introduce some terminology. In this post (and in the score matching literature in general), the score function (sometimes also called the Stein score) refers to the gradient of the log likelihood function with respect to the data $x$,

$$\nabla_x \log p_m(x; \theta).$$

Note that this is slightly confusing because in most of the statistical literature, the "score" refers to the gradient of the log likelihood with respect to the *parameter*, $\nabla_\theta \log p_m(x; \theta)$. Nevertheless, this terminology has persisted, so we'll continue with it in this post.

At first glance, the score function might seem unintuitive -- it's not clear why one would want to take the gradient of a function with respect to the data. The primary motivation is to remove the normalizing constant. Clearly, we can expand the score as

$$\nabla_x \log p_m(x; \theta) = \nabla_x \log \widetilde{p}_m(x; \theta) - \underbrace{\nabla_x Z_\theta}_{0},$$

where the final term drops off because the normalizing constant doesn't depend on $x$.

Note that, even though the score function isn't the model distribution in its original form, it still contains information about this distribution. In particular, it tells us the gradient of the distribution for a given value of the parameter $\theta$, giving us some sense of the first-order structure of the function.

In score matching, our goal will be to make the score function of the model distribution as "close" as possible to the score function of the data distribution, $\nabla_x p_d(x)$. 

In particular, the objective in score matching is to minimize the Fisher divergence between these two score functions,

\begin{align} \widehat{\theta}\_{SM} &= \argmin_{\theta} D_F(p_d, p_m) \\\ &= \argmin_\theta \frac12 \mathbb{E}\_{p_d} \left[ \\| \nabla_x \log p_d(x) - \nabla_x \log p_m(x; \theta)\\|\_2^2 \right]. \tag{1} \label{eq:loss_fn} \end{align}

Clearly, minimizing this divergence directly will still be difficult since the objective depends on the data distribution $p_d(x)$ and the normalized model distribution $p_m(x; \theta)$. The key insight in score matching from [Hyvärinen et al.](https://jmlr.org/papers/volume6/hyvarinen05a/old.pdf) is that **we can rewrite the objective in Equation \ref{eq:loss_fn} so that it only depends on the unnormalized model density,** thus circumventing the need to ever deal with the normalizing constant and the need to approximate the data distribution.

We've already seen that the normalizing constant trivially drops out after we take a gradient with respect to $x$, so let's show that we can rewrite this objective without $p_d(x)$. Here, we'll show it for the univariate case, but it can be easily extended to the multivariate setting. To start, let's expand out the norm above:

\begin{align} &\frac12 \\| \nabla_x \log p_d(x) - \nabla_x \log p_m(x; \theta)\\|\_2^2 \\\ =& \underbrace{\frac12 (\nabla_x \log p_d(x))^2}\_{\text{constant}} - \nabla_x \log p_m(x; \theta) \nabla_x \log p_d(x) + \frac12 (\nabla_x \log p_m(x; \theta))^2. \end{align}

The first term doesn't depend on $\theta$ (it only contains the data density), so we can ignore it. The third term is already easily approximable using a finite sample of data points, since it doesn't depend on the data density. Thus, our focus turns to the second term. Let's expand out the expectation and gradient:

\begin{align} &\mathbb{E}\_{p_d}\left[ -\nabla_x \log p_m(x; \theta) \nabla_x \log p_d(x) \right] \\\ =& -\int_{-\infty}^{\infty} \nabla_x \log p_m(x; \theta) \nabla_x \log p_d(x) p_d(x) dx \\\ =& -\int_{-\infty}^{\infty} \nabla_x \log p_m(x; \theta) \frac{\nabla_x p_d(x)}{p_d(x)} p_d(x) dx \\\ =& -\int_{-\infty}^{\infty} \nabla_x \log p_m(x; \theta) \nabla_x p_d(x) dx. \tag{2}\label{eq:eq2} \end{align}

To simplify this further, we can use integration by parts. Recall that for any two continuously differentiable functions $u(x)$ and $v(x)$, it holds that

\begin{align} \int_a^b u(x) v'(x) dx &= \left[u(x) v(x)\right]\rvert_a^b - \int_a^b u'(x) v(x) dx \\\ &= u(b) v(b) - u(a) v(a) - \int_a^b u'(x) v(x) dx. \end{align}

Letting $u(x) = \nabla_x \log p_m(x; \theta)$ and $v'(x) = \nabla_x p_d(x)$, Equation \ref{eq:eq2} simplifies to

$$\underbrace{- \lim_{b \to \infty} \nabla_x \log p_m(b; \theta) p_d(b)}_{0} + \underbrace{\lim_{a \to -\infty} \nabla_x \log p_m(a; \theta) p_d(a)}_{0} + \int_{-\infty}^{\infty} \nabla_x^2 p_m(x; \theta) p_d(x) dx.$$

Hyvärinen et al. make the assumption (this is a regularity condition in their Theorem 1) that for any $\theta$,

$$p_d(x) \nabla_x \log p_m(x; \theta) \to 0 \text{ when } \|x\|_2 \to \infty,$$

which allows us to drop the first two terms.

Collecting our progress, we have

$$D_F(p_d, p_m) \propto L(\theta) \triangleq \mathbb{E}_{p_d}\left[ \text{tr} \left( \nabla_x^2 \log p_m(x; \theta)\right) + \frac12 \|\nabla_x \log p_m(x; \theta)\|_2^2 \right]$$

And that's it! We have now written the objective only in terms of $\log p_m(x; \theta)$, which doesn't depend on the normalizing constant or the data distribution.

We can then approximate the objective with our data sample:

\begin{equation} L(\theta) \approx \frac1n \sum\limits_{i=1}^n \left[ \text{tr} \left( \nabla_x^2 \log p_m(x_i; \theta)\right) + \frac12 \\|\nabla_x \log p_m(x_i; \theta)\\|\_2^2 \right]. \tag{3}\label{eq:eq3} \end{equation}

Fortunately, Hyvärinen showed that if the data distribution is in the model class, then minimizing this objective will find the optimal parameter value. Specifically, assume $p_d(x) = p(x; \theta^\star)$ for some $\theta^\star$. Then it holds that

$$L(\theta) = 0 \iff \theta = \theta^\star.$$

We can then minimize Equation \ref{eq:eq3} with respect to $\theta$ using standard optimization methods.

## Understanding the objective function

Let's build some intuition for the objective function in Equation \ref{eq:eq3}.

For the first term, $\\|\nabla_x \log p_m(x_i; \theta)\\|\_2^2$, we expect this to be small (close to zero) when the data point $x_i$ is well explained by $\theta$. Intuitively, if we were to perturb $x_i$ for a given $\theta$, we hope that the likelihood doesn't change much.

For the second term, $\text{tr} \left( \nabla_x^2 \log p_m(x_i; \theta)\right)$, we can think of this term as measuring how "sharp" of a local minimum we're at. The trace of the Hessian (or second derivative in the univariate case) should be more negative if we're at a "sharp" minimum, as opposed to a flat minimum. Intuitively, we can say that we prefer minima that more uniquely explain the data, as opposed to flat minima where several values of surrounding $\theta$ could explain the data equally well.

Now that we understand the objective function, let's see an example.

## Gaussian example

Consider a simple example with a univariate Gaussian model. Suppose we observe data points $x_1, \dots, x_n \in \mathbb{R}$ and wish to fit a univariate Gaussian (by finding its mean $\mu$ and variance $\sigma^2$) to these data using maximum likelihood. Of course, a simple closed-form solution exists in this case because the normalizing constant is tractable, but here we show how to find the MLE with score matching for demonstration.

Recall the Gaussian PDF, which we use as our model's distribution:

$$p_m(x; \mu, \sigma^2) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left\{ -\frac{1}{2 \sigma^2} (x - \mu)^2 \right\},$$

where $Z_\theta = \sqrt{2 \pi \sigma^2}$ is known in this case. Taking the log of the PDF, we have

$$\log p_m(x; \mu, \sigma^2) = -\frac12 \log(2 \pi \sigma^2) - \frac{1}{2 \sigma^2} (x - \mu)^2.$$

The score function can easily be computed as

$$\frac{\partial}{\partial x} \log p_m = \frac{1}{\sigma^2} (\mu - x).$$

We can now form the objective function in Equation \ref{eq:eq3}. The gradient (derivative in this case) of the score function with respect to $x$ is then

$$\frac{\partial^2}{\partial x^2} \log p_m = -\frac{1}{\sigma^2},$$

and the norm of the score function is

$$\left\|\frac{\partial}{\partial x} \log p_m\right\|_2^2 = \frac{1}{\sigma^4}(x - \mu)^2.$$

Putting these together, our objective function is

$$L(\mu, \sigma^2) = \frac1n \sum\limits_{i=1}^n \left[ -\frac{1}{\sigma^2} + \frac{1}{2 \sigma^4}(x - \mu)^2 \right]$$

Taking derivativevs with respect to $\mu$ and $\sigma^2$, we find that we recover the traditional Gaussian MLEs:

$$\widehat{\mu}_{SM} = \frac1n \sum\limits_{i=1}^n x_i,~~~\widehat{\sigma^2}_{SM} = \frac1n \sum\limits_{i=1}^n (x_i - \widehat{\mu}_{SM})^2.$$

We visualize this below. Here, we sample $n=100$ data points from a standard Gaussian, with $\mu=0, \sigma^2=1$. In the left panel of the animation below, we fix $\mu=0$ and plot the PDF for the Gaussian with different values for $\sigma^2$. The data locations are shown in red at the bottom. In the right panel, we make a bar plot of each term in the objective function (called "Hessian" and "Norm" to refer to the respective terms in Equation \ref{eq:eq3}), as well as the total value of the objective function.

<center>
<video style="width:100%; text-align:center; display:block; margin-top:50px;" autoplay loop>
<source src="/assets/score_matching_univariate_gaussian.mp4" type="video/mp4">
</video>
<figcaption style="margin-bottom:50px;"><i></i></figcaption>
</center>

Let's examine the behavior of each term separately. The Hessian term is very small at the beginning of the animation, when $\sigma^2$ is small. This makes sense with our previous intuition -- this term prefers the log likelihood to be sharply peaked. As $\sigma^2$ becomes larger, the log likelihood becomes more flat, and this Hessian loss term increases.

On the other hand, the "Norm" term is large at the beginning of the animation, when $\sigma^2$ is too small to explain the data. As $\sigma^2$ increases, it reaches a maximum of the log likelihood around $\sigma^2=1$. Notably, the Norm term continues to decrease even past $\sigma^2=1$, showing that this term will prefer more "flat" regions of the log likelihood. Thus, the Hessian and Norm terms balance each other out in order to find a value of $\sigma^2$ that both explains the data and is appropriately peaked.

## References

- Hyvärinen, Aapo, and Peter Dayan. "Estimation of non-normalized statistical models by score matching." Journal of Machine Learning Research 6.4 (2005).
- Song, Yang, and Stefano Ermon. "Generative modeling by estimating gradients of the data distribution." arXiv preprint arXiv:1907.05600 (2019).
- Song, Yang, et al. "Sliced score matching: A scalable approach to density and score estimation." Uncertainty in Artificial Intelligence. PMLR, 2020.


