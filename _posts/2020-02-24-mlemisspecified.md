---
layout: post
title: "MLE under a misspecified model"
author: "Binh Ho"
categories: Statistics
blurb: "When we construct and analyze statistical estimators, we often assume that the model is correctly specified. However, in practice, this is rarely the case --- our assumed models are usually approximations of the truth, but they're useful nonetheless."
img: ""
tags: []
<!-- image: -->
---



When we construct and analyze statistical estimators, we often assume that the model is correctly specified. However, in practice, this is rarely the case --- our assumed models are usually approximations of the truth, but they're [useful](https://www.wikiwand.com/en/All_models_are_wrong) nonetheless.

Suppose we observe some data $X_1, X_2, \dots, X_n$ that in truth come from a distribution $f(x; \theta_0)$, where $f$ is the true distribution, and $\theta_0$ is the true parameter (or parameter vector). Suppose, however, that we try to model these data with a distribution $g(x; \theta)$. If we try to compute the maximum likelihood estimate of the parameter $\theta$ assuming that the data came from $g$, will we be guaranteed anything about the correctness of the solution, or how close it will be to the true parameter $\theta$ under the true model $f$?

Recall that computing the MLE is equivalent to finding the parameter value that maximizes the (log-)likelihood:

$$\frac1n \sum\limits_{i = 1}^n \log g(X_i; \theta).$$

If the model $g$ were in fact correct, then this quantity would converge to $\mathbb{E}_g \log g(X_i; \theta)$. However, because the true model is $f$, it instead converges to the expectation under $f$:

$$\frac1n \sum\limits_{i = 1}^n \log g(X_i; \theta) \to \mathbb{E}_f [\log g(X_i; \theta)]$$.

Inspecting this quantity can give us intuition for what it means to compute the MLE under a misspecified model.

Notice that by adding and subtracting $\mathbb{E}_f [\log f(X_i; \theta)]$, we have:

\begin{align} \mathbb{E}_f [\log g(X; \theta)] &= \mathbb{E}_f [\log g(X; \theta)] - \mathbb{E}_f [\log f(X; \theta)] + \mathbb{E}_f [\log f(X; \theta)] \\\ &= \mathbb{E}_f \left[ \frac{\log g(X; \theta)}{\log f(X; \theta)} \right] + \mathbb{E}_f [\log f(X; \theta)] \\\ &= \mathbb{E}_f [\log f(X; \theta)] - \mathbb{E}_f \left[ \frac{\log f(X; \theta)}{\log g(X; \theta)} \right]. \\\ \end{align}

We can now notice that the last term is equal to the KL divergence between $f$ and $g$.

$$\mathbb{E}_f [\log g(X; \theta)] = \mathbb{E}_f [\log f(X; \theta)] - D_{\text{KL}}(f(X; \theta), g(X; \theta)).$$

Thus, the expected likelihood is the true expected likelihood $\mathbb{E}_f [\log f(X; \theta)]$, plus a term measuring the distance between the true value and the value under the incorrect model $g$. This is a useful interpretation, since it tells us that the MLE under the misspecfied model $g$ is effectively minimizing the KL divergence between $g$ and the true density $f$.

## Asymptotic normality

A well-known property of the MLE under a correctly specified model is that it's asymptotically normal. That is,

$$\sqrt{n}(\hat{\theta}_{\text{MLE}}) - \theta_0 \to_d \mathcal{N}(0, I^{-1}(\theta_0))$$

where $I$ is the Fisher information.

A natural question is: is the MLE still asymptotically normal if the model is misspecified?

It turns out the answer is yes. We can begin to investigate this question by examining the limiting distribution of the score function. Recall that the score is simply the first derivative of the log-likelihood function with respect to the parameter $\theta$ of interest. Denote the log-likelihood as $\mathcal{L}_n(\theta) = \sum\limits_{i=1}^n \log g(x; \theta)$, and let $\theta_{0, g}$ be the true MLE of the misspecified model:

$$\theta_{0, g} = \text{arg}\max_\theta \mathbb{E}_f [\log g(x; \theta)]$$

Then the "score identity" is:

$$\mathbb{E}\left[\frac{\partial \mathcal{L}_n(\theta)}{\partial \theta}\right] = 0,  \forall \theta \in \Theta.$$

This identity is true under some fairly standard regularity conditions (most notably, that differentiation and integration can be interchanged).

The central limit theorem says that the score will be Gaussian asymptotically:

$$\frac{1}{\sqrt{n}}\frac{\partial \mathcal{L}_n(\theta)}{\partial \theta}\rvert_{\theta = \theta_{0,g}} \to_d \mathcal{N}\left(0, \mathbb{V}\left[\frac{\partial \log f(X; \theta)}{\partial  \theta}\right] \rvert_{\theta = \theta_{0,g}} \right).$$


Then, if we locally approximate (linearize) the score identity 

$$0 = \mathcal{L}'(\hat{\theta}_\text{MLE})$$

around $\theta_{0, g}$ using a Taylor expansion, we have:

$$0 \approx \mathcal{L}'(\theta_{0, g}) + \mathcal{L}''(\theta_{0,g})(\hat{\theta}_\text{MLE} - \theta_{0,g})$$

which implies, 

$$\sqrt{n}(\hat{\theta}_\text{MLE} - \theta_{0,g}) \approx -\frac{n^{-1/2}\mathcal{L}'(\theta_{0, g})}{n^{-1}\mathcal{L}''(\theta_{0,g})}.$$

The term in the denominator $n^{-1}\mathcal{L}''(\theta_{0,g})$ will converge in probability to the expected value of the first derivative of the score (by the LLN):

$$n^{-1}\mathcal{L}''(\theta_{0,g}) \to_p \mathbb{E}_f[\ell''(\theta_{0,g})]$$

where $\ell(\theta_{0, g}) = \log f(X; \theta_{0, g})$.

At the same time, the numerator $n^{-1/2}\mathcal{L}'(\theta_{0, g})$ will be asymptotically normal, by the CLT:

$$n^{-1/2}\mathcal{L}'(\theta_{0, g}) \to_d \mathcal{N}\left( 0, \mathbb{V}[\ell'(\theta_{0, g})] \right).$$

Then we can conclude, using Slutsky's theorem, that

$$\sqrt{n} (\hat{\theta}_{\text{MLE}} - \theta_{0,g}) \to_d \mathcal{N}(0, \mathbb{E}_f[\ell''(\theta_{0,g})]^{-1} \mathbb{V}[\ell'(\theta_{0, g})] \mathbb{E}_f[\ell''(\theta_{0,g})]^{-1}).$$

Thus, the MLE is still asymptotically normal under a misspecified model. Notice that the asymptotic variance has the same form as the "sandwich estimator" for the variance, which I covered in a post about M-estimators [here](https://callmequant.github.io/statistics/m-estimation.html).

## Conclusion

Many classical statistics results rest on the assumption of the underlying model being correctly specified. However, this assumption rarely, if ever, holds true in practice. In this post, we saw that the maximum likelihood approach to estimation is robust to model misspecification, in the sense that the MLE under a misspecified model is still consistent and asymptotically normal.

## References

- Suhasini Subba Rao's [lecture notes](https://www.stat.tamu.edu/~suhasini/teaching613/chapter5.pdf).
- Zhou Fan's [lecture notes](https://web.stanford.edu/class/archive/stats/stats200/stats200.1172/Lecture16.pdf).
- White, Halbert. "Maximum likelihood estimation of misspecified models." Econometrica: Journal of the Econometric Society (1982): 1-25.
