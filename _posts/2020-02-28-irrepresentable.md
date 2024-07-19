---
layout: post
title: "LASSO and the irrepresentable condition"
author: "Binh Ho"
categories: Statistics
blurb: "In this post, we cover a condition that is necessary and sufficient for the LASSO estimator to work correctly."
img: ""
tags: []
<!-- image: -->
---


In this post, we cover a condition that is necessary and sufficient for the LASSO estimator to work correctly.

## Introduction

Model selection is an important part of statistical modeling, especially when working with high-dimensional data. Generically, model selection refers to the process of deciding which of a set of models is best (each of which encodes a different set of assumptions). In the context of linear regression, it usually refers to the process of deciding which covariates to include in a model.

Consider the basic linear model:

$$\mathbf{Y} = \mathbf{X} \boldsymbol{\beta} + \boldsymbol{\epsilon}$$

where $\mathbf{Y} \in \mathbb{R}^n$ is a vector of response variables, $\mathbf{X} \in \mathbb{R}^{n\times p}$ is a matrix of data, $\boldsymbol{\beta} \in \mathbb{R}^p$ is a vector of coefficients, and $\epsilon \in \mathbb{R}^n$ is noise.

Recall that OLS estimates a parameter vector $\hat{\boldsymbol{\beta}}$ that minimizes the squared error:

$$\hat{\boldsymbol{\beta}} = \text{arg}\min_{\boldsymbol{\beta}} ||\mathbf{Y} - \mathbf{X}\boldsymbol{\beta}||_2^2.$$

The OLS estimator has a closed form:

$$\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top \mathbf{X})^{-1} (\mathbf{X} \mathbf{Y}).$$

It's not uncommon to have hundreds, thousands, or even millions of covariates in a dataset (i.e., $p$ is large), and it is often of interest to find just a handful of covariates that are truly related to the response. Of course, the most thorough way to do this would be to fit a linear model with every possible subset of variables, but this would require sweeping through an exponentially large number of combinations, and is thus computationally prohibitive.

Modern methods of model selection have concocted convex relaxations of the "best subset" problem, which are much more computationally efficient. One such popular method is called the LASSO, which estimates $\hat{\boldsymbol{\beta}}$ such that

$$\hat{\boldsymbol{\beta}} = \text{arg}\min_{\boldsymbol{\beta}} ||\mathbf{Y} - \mathbf{X}\boldsymbol{\beta}||_2^2 + \lambda ||\boldsymbol{\beta}||_1$$

where $\lambda \in \mathbb{R}$ is a tunable parameter that controls the level of regularization. The LASSO estimator encourages sparse solutions for $\hat{\boldsymbol{\beta}}$, so that most of the coefficients are $0$.

## LASSO's consistency

There are several ways to discuss the consistency of an estimator. The most common meaning of consistency is that the estimated parameters converge to the true parameters as the sample size increases:

$$\hat{\boldsymbol{\beta}}^{(n)} \to_p \boldsymbol{\beta}_0 \text{ as } n \to \infty$$

where $\hat{\boldsymbol{\beta}}^{(n)}$ is the estimator using $n$ samples, and $\boldsymbol{\beta}_0$ is the true parameter.

Another useful notion of consistency in the context of LASSO is "model selection consistency". This type of consistency requires that all of the variables with true nonzero coefficients are estimated to be nonzero:

$$\mathbb{P}\left[ \{i : \hat{\boldsymbol{\beta}}_i \neq 0\} = \{i : \boldsymbol{\beta}_{0i} \neq 0\} \right] \to 1 \text{ as } n \to \infty.$$

In other words, the set of variables that are estimated to have nonzero coefficients, $\{i : \hat{\boldsymbol{\beta}}\_i \neq 0\}$, is equal to the variables that should have nonzero coefficients under the true model, $\{i : \boldsymbol{\beta}\_{0i} \neq 0\}$. This type of consistency is useful for talking about LASSO because LASSO is constructed to be a good method for model selection.

## The irrepresentable condition

A key question is: under what circumstances does the LASSO estimator have model selection consistency? A 2006 paper by Peng Zhao and Bin Yu answered this question by coining the "irrepresentable condition". This condition, which places restrictions on the data matrix $\mathbf{X}$, is necessary and sufficient for the LASSO estimator to exhibit model selection consistency.

Let's start to define this condition. Recall that we're assuming the data were generated from some true model:

$$\mathbf{Y} = \mathbf{X} \boldsymbol{\beta}_0 + \boldsymbol{\epsilon}.$$

We're also assuming that $\boldsymbol{\beta}$ is sparse, or that most of its components are $0$, and thus that only a few of the variables truly affect the model. Let $\mathbf{X}\_1$ be the subset of the data matrix that contains the relevant variables, let $\mathbf{X}\_2$ be the subset with the data for the irrelevant variables, and let $\boldsymbol{\beta}\_{0(1)}$ be the coefficients for $\mathbf{X}_1$. The irrepresentable condition requires that $\|\|(\mathbf{X}\_2^\top \mathbf{X}\_1)^{-1} (\mathbf{X}\_1 \mathbf{X}\_1) \text{sgn}(\boldsymbol{\beta}\_{0(1)})\|\|\_\infty < 1.$

Basically, this means that the irrelevant variables cannot be too correlated with the relevant variables. One way of seeing this is that the expression $(\mathbf{X}\_2^\top \mathbf{X}\_1)^{-1} (\mathbf{X}\_1 \mathbf{X}\_1)$ is an estimate of the regression coefficients for $\mathbf{X}\_2$ regressed on $\mathbf{X}\_1$ (recall that the standard OLS estimate for $\mathbf{Y}$ regressed on $\mathbf{X}$ is $\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top \mathbf{X})^{-1} (\mathbf{X} \mathbf{Y})$). So in other words, the regression coefficients for $\mathbf{X}\_2$ regressed on $\mathbf{X}\_1$ can be at most $1$. Finally, the term $\text{sgn}(\boldsymbol{\beta}_{0(1)})$ simply checks whether the correlation between $\mathbf{X}\_2$ and $\mathbf{X}\_1$ has the same sign as the true coefficient.

Intuitively, this makes sense because if the irrelevant variables were highly correlated with the relevant variables, it would become impossible to recover which were the "important" variables. The term "representable" means that $\mathbf{X}_2$ can be approximated by a linear combination of $\mathbf{X}_1$. Conversely, "irrepresentable" means that no such sufficiently good linear approximation exists.

## Conclusion

The LASSO estimator is a clever and powerful technique for identifying a sparse set of important covariates. However, its consistency relies on the irrepresentable condition, which may not always hold in practice, and is difficult or impossible to check.

## References

- Lecture notes from Prof. Jianqing Fan's class ORF525.
- Fan, J., Li, R., Zhang, C.-H., and Zou, H. (2020). Statistical Foundations of Data Science.
CRC Press, forthcoming.
- Zhao, Peng, and Bin Yu. "On model selection consistency of Lasso." Journal of Machine learning research 7.Nov (2006): 2541-2563.


