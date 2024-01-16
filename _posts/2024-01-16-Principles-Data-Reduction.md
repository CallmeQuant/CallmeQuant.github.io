---
title: "Principles Data Reduction"
date: 2024-01-16
mathjax: true
toc: true
categories:
  - Study
tags:
  - Statistical Inference
  - Statistics
---

# Sufficient Principle and Minimal Sufficient Statistics
We usually find a sufficient statistic by simple inspection of the pdf or pmf of the sample.

**Theorem (Factorization Theorem)** Let $f(x \lvert \theta)$ denote the joint pdf or pmf of 
a sample X. A statistics T(X) is a sufficient statistics for $\theta$ if and only
if there exist functions $g(t \lvert \theta)$ and $h(x)$ such that, for all sample 
poins $x$ and all parameter points $\theta$,

$$f(x \lvert \theta) = g(T(x) \lvert \theta) h(x)
