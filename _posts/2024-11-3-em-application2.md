---
layout: post
title: "EM applications: Stochastic EM"
author: "Binh Ho"
categories: Statistics
blurb: "Expectation Maximization (EM) is a ubiquitous algorithm for performing maximum likelihood estimation. In this series, I present EM algorithm for GMMs' parameters."
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
$$\newcommand{\converge}[1]{\xrightarrow{\makebox[2em][c]{$$\scriptstyle#1$$}}}$$
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
$$\newcommand{\cA}{\mathcal{A}}$$
$$\newcommand{\cB}{\mathcal{B}}$$
$$\newcommand{\cC}{\mathcal{C}}$$
$$\newcommand{\cD}{\mathcal{D}}$$
$$\newcommand{\cE}{\mathcal{E}}$$
$$\newcommand{\cF}{\mathcal{F}}$$
$$\newcommand{\cG}{\mathcal{G}}$$
$$\newcommand{\cH}{\mathcal{H}}$$
$$\newcommand{\cI}{\mathcal{I}}$$
$$\newcommand{\cJ}{\mathcal{J}}$$
$$\newcommand{\cL}{\mathcal{L}}$$
$$\newcommand{\cM}{\mathcal{M}}$$
$$\newcommand{\cP}{\mathcal{P}}$$
$$\newcommand{\cO}{\mathcal{O}}$$
$$\newcommand{\cQ}{\mathcal{Q}}$$
$$\newcommand{\cU}{\mathcal{U}}$$
$$\newcommand{\cV}{\mathcal{V}}$$
$$\newcommand{\cN}{\mathcal{N}}$$
$$\newcommand{\cT}{\mathcal{T}}$$
$$\newcommand{\cX}{\mathcal{X}}$$
$$\newcommand{\cY}{\mathcal{Y}}$$
$$\newcommand{\cZ}{\mathcal{Z}}$$
$$\newcommand{\cS}{\mathcal{S}}$$
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

# Stochastic EM Algorithm for Gaussian Mixture Models (GMMs)

In this blog post, we'll explore the Stochastic Expectation-Maximization (SEM) algorithm and its application to Gaussian Mixture Models (GMMs). Building upon the traditional EM algorithm, SEM introduces stochasticity into the estimation process, offering potential advantages in terms of escaping local optima and handling large datasets. We'll carefully examine the mathematical derivations and provide a concrete example of GMMs using SEM.

## Introduction to SEM

The Expectation-Maximization (EM) algorithm is a powerful tool for finding maximum likelihood estimates in models with latent variables, such as Gaussian Mixture Models (GMMs). However, EM can sometimes get trapped in local maxima, especially in complex or high-dimensional spaces. 

Moreover, note that the $Q_t(\Theta) := E_{Z \mid X; \Theta_t}\left[ \log p(X, Z; \Theta) \right]$ which appears in the E-step is simply the conditional expectation of the complete-data log-likelihood in terms of observed variable, given $X$ and assuming the true parameter value is $\Theta_t$. This yields some probabilistic insight on this $Q$ function. Since this is expectation, for all $\Theta$, $Q_t(\Theta)$ is an estimate of the complete-data log-likelihood built on the information of the incomplete data and under the assumption that the true parameter is unknown. In some way, it is not far from being the “best” estimate that we can possibly make without knowing $Z$, because conditional expectation is, by definition, the estimator that minimizes the conditional mean squared error:

$$
Q_t(\Theta) = \argmin_{\mu} \int \left[\log p(X, Z; \Theta) -\mu \right]^2 P(Z \mid X; \Theta_t) dZ
$$ 

The **Stochastic EM (SEM)** algorithm introduces randomness into the E-step, which can help the algorithm explore the parameter space more thoroughly and potentially escape local optima. Instead of computing expected values over the latent variables, SEM *samples* them according to their posterior distributions given the current parameter estimates.

**Key Differences Between EM and SEM:**

- **EM Algorithm:**
  - **E-Step:** Compute the expected value of the latent variables given the current parameters.
  - **M-Step:** Maximize the expected complete-data log-likelihood with respect to the parameters.

- **SEM Algorithm:**
  - **S-Step (Stochastic E-Step):** Sample the latent variables from their posterior distributions given the current parameters.
  - **M-Step:** Maximize the complete-data log-likelihood with respect to the parameters using the sampled latent variables.

By introducing stochasticity, SEM can offer advantages in terms of convergence properties and computational efficiency, particularly for large datasets.

## Mathematical Derivation of SEM for GMMs

Let's consider a GMM with $ K $ Gaussian components. Our goal is to estimate the parameters $ \Theta = \{ \pi_k, \vect{\mu}_k, \vect{\Sigma}_k \}_{k=1}^K $, where:

- $ \pi_k $ are the mixture weights (with $ \sum_{k=1}^K \pi_k = 1 $),
- $ \vect{\mu}_k $ are the means,
- $ \vect{\Sigma}_k $ are the covariance matrices.

Given observed data $ X = \{\vect{x}_1, \dots, \vect{x}_n\} $, we introduce latent variables $ Z = \{z_1, \dots, z_n\} $, where $ z_i \in \{1, \dots, K\} $ indicates the component assignment for $ \vect{x}_i $.

### Objective Function

Our objective is to maximize the complete-data log-likelihood:

$$
\log p(X, Z \mid \Theta) = \sum_{i=1}^n \log \left( \pi_{z_i} \mathcal{N}(\vect{x}_i \mid \vect{\mu}_{z_i}, \vect{\Sigma}_{z_i}) \right)
$$

In SEM, we alternate between sampling the latent variables $ Z $ and updating the parameters $ \Theta $.

### E-Step (Stochastic Step)

**S-Step:** For each data point $ \vect{x}_i $, sample $ z_i $ from the posterior distribution:

$$
P(z_i = k \mid \vect{x}_i; \Theta_t) = \gamma_{t, i, k} = \frac{\pi_{t, k} \mathcal{N}(\vect{x}_i \mid \vect{\mu}_{t, k}, \vect{\Sigma}_{t, k})}{\sum_{j=1}^K \pi_{t, j} \mathcal{N}(\vect{x}_i \mid \vect{\mu}_{t, j}, \vect{\Sigma}_{t, j})}
$$

Rather than computing the expected value over $ Z $, we sample each $ z_i $ according to $ \gamma_{t, i, k} $.

### M-Step (Maximization Step)

Given the sampled $ Z $, we maximize the complete-data log-likelihood with respect to $ \Theta $:

$$
\Theta_{t+1} = \arg\max_{\Theta} \sum_{i=1}^n \log \left( \pi_{z_i} \mathcal{N}(\vect{x}_i \mid \vect{\mu}_{z_i}, \vect{\Sigma}_{z_i}) \right)
$$

We can derive the update equations for $ \pi_k $, $ \vect{\mu}_k $, and $ \vect{\Sigma}_k $.

---

### Detailed Derivation

#### E-Step (Stochastic Step)

For each data point $ \vect{x}_i $:

1. **Compute Responsibilities:**

   $$
   \gamma_{t, i, k} = P(z_i = k \mid \vect{x}_i; \Theta_t) = \frac{\pi_{t, k} \mathcal{N}(\vect{x}_i \mid \vect{\mu}_{t, k}, \vect{\Sigma}_{t, k})}{\sum_{j=1}^K \pi_{t, j} \mathcal{N}(\vect{x}_i \mid \vect{\mu}_{t, j}, \vect{\Sigma}_{t, j})}
   $$

2. **Sample $ z_i $:**

   - Sample $ z_i $ from the categorical distribution defined by $ \gamma_{t, i, k} $.

#### M-Step (Maximization Step)

Given the sampled $ Z $, we update the parameters.

##### Update Mixing Coefficients $ \pi_k $

The likelihood w.r.t $ \pi_k $ is:

$$
L(\pi) = \sum_{i=1}^n \log \pi_{z_i}
$$

Subject to the constraint $ \sum_{k=1}^K \pi_k = 1 $. Using the method of Lagrange multipliers:

1. **Formulate Lagrangian:**

   $$
   \mathcal{L}(\pi, \lambda) = \sum_{i=1}^n \log \pi_{z_i} + \lambda \left( \sum_{k=1}^K \pi_k - 1 \right)
   $$

2. **Compute Gradient and Set to Zero:**

   For each $ \pi_k $:

   $$
   \frac{\partial \mathcal{L}}{\partial \pi_k} = \sum_{i: z_i = k} \frac{1}{\pi_k} + \lambda = 0
   $$

3. **Solve for $ \pi_k $:**

   $$
   \pi_k = -\frac{N_k}{\lambda}
   $$

   Where $ N_k = \sum_{i=1}^n \delta(z_i = k) $ is the number of data points assigned to component $ k $.

4. **Apply Constraint:**

   $$
   \sum_{k=1}^K \pi_k = -\frac{1}{\lambda} \sum_{k=1}^K N_k = 1 \implies \lambda = -n
   $$

5. **Final Update:**

   $$
   \pi_k = \frac{N_k}{n}
   $$

##### Update Means $ \vect{\mu}_k $

We need to maximize:

$$
L(\vect{\mu}_k) = \sum_{i: z_i = k} \log \mathcal{N}(\vect{x}_i \mid \vect{\mu}_k, \vect{\Sigma}_k)
$$

This is equivalent to minimizing:

$$
\sum_{i: z_i = k} (\vect{x}_i - \vect{\mu}_k)^T \vect{\Sigma}_k^{-1} (\vect{x}_i - \vect{\mu}_k)
$$

Setting the derivative w.r.t $ \vect{\mu}_k $ to zero:

$$
\frac{\partial L}{\partial \vect{\mu}_k} = \sum_{i: z_i = k} \vect{\Sigma}_k^{-1} (\vect{x}_i - \vect{\mu}_k) = 0
$$

Solving for $ \vect{\mu}_k $:

$$
\vect{\mu}_k = \frac{1}{N_k} \sum_{i: z_i = k} \vect{x}_i
$$

##### Update Covariances $ \vect{\Sigma}_k $

We maximize:

$$
L(\vect{\Sigma}_k) = \sum_{i: z_i = k} \log \mathcal{N}(\vect{x}_i \mid \vect{\mu}_k, \vect{\Sigma}_k)
$$

This involves minimizing:

$$
\sum_{i: z_i = k} \left[ \log \det \vect{\Sigma}_k + (\vect{x}_i - \vect{\mu}_k)^T \vect{\Sigma}_k^{-1} (\vect{x}_i - \vect{\mu}_k) \right]
$$

Setting the derivative w.r.t $ \vect{\Sigma}_k $ to zero:

1. **Compute Gradient:**

   $$
   \frac{\partial L}{\partial \vect{\Sigma}_k} = \frac{N_k}{2} \vect{\Sigma}_k^{-1} - \frac{1}{2} \vect{\Sigma}_k^{-1} \left( \sum_{i: z_i = k} (\vect{x}_i - \vect{\mu}_k)(\vect{x}_i - \vect{\mu}_k)^T \right) \vect{\Sigma}_k^{-1} = 0
   $$

2. **Solve for $ \vect{\Sigma}_k $:**

   $$
   \vect{\Sigma}_k = \frac{1}{N_k} \sum_{i: z_i = k} (\vect{x}_i - \vect{\mu}_k)(\vect{x}_i - \vect{\mu}_k)^T
   $$

---

## Algorithm Summary

The SEM algorithm for GMMs can be summarized as follows:

1. **Initialize** $ \Theta_0 = \{ \pi_{0, k}, \vect{\mu}_{0, k}, \vect{\Sigma}_{0, k} \} $.

2. **For** $ t = 0 $ to convergence:

   - **E-Step (Stochastic Step):**
     - For each $ i $:
       - Compute responsibilities $ \gamma_{t, i, k} $ for $ k = 1, \dots, K $.
       - Sample $ z_i \sim \text{Categorical}(\gamma_{t, i, 1}, \dots, \gamma_{t, i, K}) $.

   - **M-Step:**
     - For each $ k $:
       - Compute $ N_k = \sum_{i=1}^n \delta(z_i = k) $.
       - Update mixing coefficients:
         $$
         \pi_{t+1, k} = \frac{N_k}{n}
         $$
       - Update means:
         $$
         \vect{\mu}_{t+1, k} = \frac{1}{N_k} \sum_{i: z_i = k} \vect{x}_i
         $$
       - Update covariances:
         $$
         \vect{\Sigma}_{t+1, k} = \frac{1}{N_k} \sum_{i: z_i = k} (\vect{x}_i - \vect{\mu}_{t+1, k})(\vect{x}_i - \vect{\mu}_{t+1, k})^T
         $$

3. **Check for Convergence:**

   - Monitor the log-likelihood or parameter changes to determine if convergence criteria are met.

## Final Words

The Stochastic EM algorithm offers an alternative to the traditional EM algorithm by incorporating randomness into the estimation process. This stochasticity can help the algorithm avoid local maxima and explore the parameter space more effectively.

**Summary of SEM Algorithm for GMMs:**

1. **Initialize Parameters:** Set initial values for $ \pi_k $, $ \vect{\mu}_k $, and $ \vect{\Sigma}_k $.
2. **Iterate Until Convergence:**
   - **E-Step (Stochastic Step):**
     - Compute responsibilities $ \gamma_{i, k} $.
     - Sample $ z_i $ from $ \text{Categorical}(\gamma_{i, 1}, \dots, \gamma_{i, K}) $.
   - **M-Step:**
     - Update $ \pi_k = \frac{N_k}{n} $.
     - Update $ \vect{\mu}_k = \frac{1}{N_k} \sum_{i: z_i = k} \vect{x}_i $.
     - Update $ \vect{\Sigma}_k = \frac{1}{N_k} \sum_{i: z_i = k} (\vect{x}_i - \vect{\mu}_k)(\vect{x}_i - \vect{\mu}_k)^T $.
3. **Convergence Check:** Monitor log-likelihood or parameter changes.

**Advantages of SEM:**

- **Escaping Local Optima:** Random sampling can help the algorithm find better solutions.
- **Scalability:** SEM can be more efficient with large datasets, especially when combined with mini-batch techniques.

**Considerations:**

- **Convergence Behavior:** The stochastic nature can lead to fluctuations in log-likelihood. Multiple runs may be necessary.
- **Parameter Initialization:** Good initial parameters can improve convergence.

In conclusion, SEM provides a valuable tool for mixture model estimation, particularly in complex scenarios where traditional EM may struggle. By carefully implementing and analyzing SEM, we can leverage its strengths to achieve robust clustering and parameter estimation.
