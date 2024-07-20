---
layout: post
title: "A quick look at Kalman Filter"
author: "Binh Ho"
categories: Statistics
blurb: "The Kalman filter, a cornerstone in estimation theory, is a powerful algorithm that excels at inferring the hidden state of a system based on noisy measurements. "
img: ""
tags: []
<!-- image: -->
---

$\newcommand{\vect}[1]{{\mathbf{\boldsymbol{{#1}}}}}$
$\newcommand{\abs}[1]{\lvert#1\rvert}$
$\newcommand{\norm}[1]{\lVert#1\rVert}$
$\newcommand{\innerproduct}[2]{\langle#1, #2\rangle}$
$\newcommand{\Tr}[1]{\operatorname{Tr}\mleft(#1\mright)}$

$\DeclareMathOperator*{\argmin}{argmin}$
$\DeclareMathOperator*{\argmax}{argmax}$

$\DeclareMathOperator{\diag}{diag}$

$\newcommand{\converge}[1]{\xrightarrow{\makebox[2em][c]{$\scriptstyle#1$}}}$

$\newcommand{\quotes}[1]{``#1''}$

$\newcommand\ddfrac[2]{\frac{\displaystyle #1}{\displaystyle #2}}$

$\newcommand{\vect}[1]{\boldsymbol{\mathbf{#1}}}$

$\newcommand{\E}{\mathbb{E}}$

$\newcommand{\Var}{\mathrm{Var}}$

$\newcommand{\Cov}{\mathrm{Cov}}$

$\newcommand{\N}{\mathbb{N}}$
$\newcommand{\Z}{\mathbb{Z}}$
$\newcommand{\Q}{\mathbb{Q}}$
$\newcommand{\R}{\mathbb{R}}$
$\newcommand{\C}{\mathbb{C}}$
$\newcommand{\bbP}{\mathbb{P}}$

$\newcommand{\rmF}{\mathrm{F}}$
$\newcommand{\iid}{\mathrm{iid}}$
$\newcommand{\distas}[1]{\overset{#1}{\sim}}$

$\newcommand{\Acal}{\mathcal{A}}$
$\newcommand{\Bcal}{\mathcal{B}}$
$\newcommand{\Ccal}{\mathcal{C}}$
$\newcommand{\Dcal}{\mathcal{D}}$
$\newcommand{\Ecal}{\mathcal{E}}$
$\newcommand{\Fcal}{\mathcal{F}}$
$\newcommand{\Gcal}{\mathcal{G}}$
$\newcommand{\Hcal}{\mathcal{H}}$
$\newcommand{\Ical}{\mathcal{I}}$
$\newcommand{\Jcal}{\mathcal{J}}$
$\newcommand{\Lcal}{\mathcal{L}}$
$\newcommand{\Mcal}{\mathcal{M}}$
$\newcommand{\Pcal}{\mathcal{P}}$
$\newcommand{\Ocal}{\mathcal{O}}$
$\newcommand{\Qcal}{\mathcal{Q}}$
$\newcommand{\Ucal}{\mathcal{U}}$
$\newcommand{\Vcal}{\mathcal{V}}$
$\newcommand{\Ncal}{\mathcal{N}}$
$\newcommand{\Tcal}{\mathcal{T}}$
$\newcommand{\Xcal}{\mathcal{X}}$
$\newcommand{\Ycal}{\mathcal{Y}}$
$\newcommand{\Zcal}{\mathcal{Z}}$
$\newcommand{\Scal}{\mathcal{S}}$

$\newcommand{\shorteqnote}[1]{ & \textcolor{blue}{\text{\small #1}}}$

$\newcommand{\qimplies}{\quad\Longrightarrow\quad}$
$\newcommand{\defeq}{\stackrel{\triangle}{=}}$
$\newcommand{\longdefeq}{\stackrel{\text{def}}{=}}$
$\newcommand{\equivto}{\iff}$

The Kalman filter, a cornerstone in estimation theory, is a powerful algorithm that excels at inferring the hidden state of a system based on noisy measurements. Imagine tracking an object amidst a fog of uncertainty; the Kalman filter is your guiding light, effectively fusing predictions with observations to refine its estimate over the course of time. This post aims to demystify the Kalman filter by providing a quick overview and a step-by-step derivation of this wonderful algorithm. 

## Quick recap from Gaussian algebra
To effectively derive Kalman filter, we need the two following results for Gaussian random variables (some books will pack them inside "lemma" or "theorem"):

### Joint distributions of Gaussian random variables
Suppose $\vect{x} \in \R^{n}$ and $\vect{y} \in \R^{m}$ are Gaussian random variables with the following structure
$$
\begin{align*}
\vect{x} &\sim \Ncal(\vect{\mu}, \vect{\Sigma}) \\
\vect{y} &\lvert x \sim \Ncal(\vect{Ax} + \vect{b}, \vect{R})
\end{align*}
$$
then, the joint distribution for $(x, y)$ is given by 
$$
\begin{pmatrix} \vect{x} \\ \vect{y} \end{pmatrix} \sim \mathcal{N}_{n+m}\left(\begin{pmatrix} \vect{\mu} \\ \vect{A\mu} + b \end{pmatrix}, 
\begin{pmatrix} \vect{\Sigma} & \vect{A \Sigma^\top} \\ \vect{\Sigma A} & \vect{A \Sigma A^\top} + \vect{R} \end{pmatrix} \right).
$$

### Conditional and marginal distribution of Gaussian random variables
Suppose $\vect{x} \in \R^{n}$ and $\vect{y} \in \R^{m}$ with the given joint Gaussian distribution of the form
$$
\begin{pmatrix} \vect{x} \\ \vect{y} \end{pmatrix} \sim \Ncal \left(\begin{pmatrix} \vect{\mu_x} \\ \vect{\mu_y}, 
\begin{pmatrix} \vect{\Sigma_{xx}} & \Sigma_{xy} \\
\vect{\Sigma_{yx}}  & \vect{\Sigma_{yy}} \end{pmatrix} \right)
$$
then marginals and conditional distributions are given by 
$$
\begin{align*}
\vect{x} & \sim \Ncal(\vect{\mu_x}, \vect{\Sigma_{xx}}) \\
\vect{y} & \sim \Ncal(\vect{\mu_y}, \vect{\Sigma_{yy}}) \\
\vect{x} \lvert \vect{y} & \sim \Ncal \left(\vect{\mu_x} + \vect{\Sigma_{xy}} \vect{\Sigma_{yy}}^{-1} (x - \vect{\mu_{y}}) , \vect{\Sigma_{xx}} - \vect{\Sigma_{xy}} \vect{\Sigma_{yy}}^{-1} \vect{\Sigma_{yx}} \right) \\
\vect{y} \lvert \vect{x} & \sim \Ncal \left(\vect{\mu_y} + \vect{\Sigma_{yx}} \vect{\Sigma_{xx}}^{-1} (y - \vect{\mu_{x}}) , \vect{\Sigma_{yy}} - \vect{\Sigma_{yx}} \vect{\Sigma_{xx}}^{-1} \vect{\Sigma_{xy}} \right)
\end{align*}
$$

## Kalman filter: a quick derivation
### A quick look
Kalman filter is an algorithm that provides exact and analytic solution (a more general solution is the Bayesian filtering method) for linear Gaussian state space models. Assume that for all $t =1, \dots, T$ and $\vect{x} \in \R^{n}$, $\vect{y} \in \R^{m}$, linear Gaussian state space models ususally take the following Markovian structure. 
$$
\begin{align*}
p(\vect{z_t} \lvert \vect{z_{t-1}}) &= \Ncal(\vect{F_t} \vect{z_{t-1}}, \vect{Q_t}) \\
p(\vect{y_t} \lvert \vect{z_t}) &= \Ncal(\vect{H_t} \vect{z_t}, \vect{R_t})
\end{align*}
$$
then the Kalman filter is a recursive algorithm which in turn *predict* and *update* the belief state $\vect{z_t}$ in an online fashion. The prediction and update steps are given by the following equations
$$
\begin{align*}
p(\vect{z_{t}} \lvert \vect{y}_{1:t-1}) &= \Ncal(\vect{z_t} \lvert \vect{\mu_{t \lvert t-1}}, \vect{\Sigma_{t \lvert t-1}}) \quad (\text{Predict step}) \\
p(\vect{z_{t}} \lvert \vect{y_t}, \vect{y}_{1:t-1}) &= \Ncal(\vect{z_t} \lvert \vect{\mu_{t \lvert t}}, \vect{\Sigma_{t \lvert t}}) \quad (\text{Update step})
\end{align*}
$$
where the moments at the prediction step are given by 
$$
\begin{align*}
\vect{\mu_{t \lvert t-1}} &= \vect{F_t} \vect{\mu_{t-1 \lvert t-1}} \\
\vect{\Sigma_{t \lvert t-1}} &=\vect{F_t} \vect{\Sigma_{t-1 \lvert t-1}} \vect{F_t}^\top + \vect{Q_t}
\end{align*}
$$
and the moments at the update step are given by 
$$
\begin{align*}
\vect{e_t} &= \vect{y_t} - \vect{H_t} \vect{\mu_{t \lvert t-1}} \\
\vect{S_t} &= \vect{H_t} \vect{\Sigma_{t \lvert t-1}}^{-1} \vect{H_t}^\top + \vect{R_t} \\
\vect{K_t} &=  \vect{\Sigma_{t \lvert t-1}} \vect{H_t}^\top \vect{S_t}^{-1} \\
\vect{\mu_{t \lvert t}} &= \vect{\mu_{t \lvert t-1}} + \vect{K_t} \vect{e_t}\\
\vect{\Sigma_{t \lvert t}} &= \vect{\Sigma_{t \lvert t-1}} - \vect{K_t} \vect{S_t} \vect{K_t}^\top \\
&=  \vect{\Sigma_{t \lvert t-1}} - \vect{K_t} \vect{H_t} \vect{\Sigma_{t \lvert t-1}}
\end{align*}
$$
### Proof Strategy
In the derivation of the algorithm, we notice that due to the nice properties of the Gaussian distribution and the Markovian structure of the hidden state, the inference will be simplified to become analytic. Thus, our strategy will be as follows:
1. Prediction step 
    + Derive the joint distribution $p(\vect{z_t}, \vect{z_{t-1}} \lvert \vect{y_{1:t-1}})$
    + Marginalize over $\vect{z_{t-1}}$ to arrive at $p(\vect{z_t} \lvert \vect{y_{1:t-1}})$. This distribution is equal to $\Ncal(\vect{z_t} \lvert \vect{\mu_{t \lvert t-1}}, \vect{\Sigma_{t \lvert t-1}})$ 
2. Update step 
    + Derive the joint distribution $p(\vect{z_t}, \vect{y_{t}} \lvert \vect{y_{1:t-1}})$ 
    + Condition on $\vect{y_t}$, we arrive at $p(\vect{z_t} \lvert \vect{y_{1:t}})$, which is $\Ncal(\vect{z_t} \lvert \vect{\mu_{t \lvert t}}, \vect{\Sigma_{t \lvert t}})$
The first three substeps will utilize the result on joint distribution of Gaussian random variables, while the last substep will be an application of the conditional distribution result.
### Proof
We will present the step-by-step (here precisely I mean that the derivation will follow the above strategy). 
1. Constructing the joint distribution $p(\vect{z_t}, \vect{z_{t-1}} \lvert \vect{y_{1:t-1}})$

Using the first result, we have 
$$
\begin{align*}
p(\vect{z_t}, \vect{z_{t-1}} \lvert \vect{y_{1:t-1}}) &= p(\vect{z_t} \lvert \vect{z_{t-1}}, \vect{y_{1:t-1}}) p(\vect{z_t-1} \lvert \vect{y_{1:t-1}}) \\
&= p(\vect{z_t} \lvert \vect{z_{t-1}}) p(\vect{z_t-1} \lvert \vect{y_{1:t-1}}) \quad (\text{Markov property}) \\
&=  \Ncal(\vect{z_t} \lvert \vect{F_t} \vect{z_{t-1}}, \vect{Q_t})\Ncal(\vect{z_{t-1}} \lvert \vect{\mu_{t-1 \lvert t-1}}, \vect{\Sigma_{t-1 \lvert t-1}}) \\
&= \Ncal \left(\begin{pmatrix} \vect{z_{t-1}} \\ \vect{z_t} \end{pmatrix} \lvert \vect{\mu}', \vect{\Sigma}' \right)
\end{align*}
$$
where 
$$
\begin{align*}
\vect{\mu}' &= \begin{pmatrix} \vect{\mu_{t-1 \lvert t-1}} \\ \vect{F_t} \vect{\mu_{t-1 \lvert t-1}} \end{pmatrix} \\
\vect{\Sigma}' = \begin{pmatrix} \vect{\Sigma_{t-1 \lvert t-1}} & \vect{\Sigma_{t-1 \lvert t-1}} \vect{F_t}^\top  \\
\vect{F_t} \vect{\Sigma_{t-1 \lvert t-1}} & \vect{F_t} \vect{\Sigma_{t-1 \lvert t-1}} \vect{F_t}^\top \end{pmatrix}
\end{align*}
$$
2. Marginalize $\vect{z_{t-1}}$ to obtain (usually in Bayesian filtering, we call this equation the *Chapman-Kolmogorov* equation)
$$
\begin{align*}
p(\vect{z_{t}} \lvert \vect{y_{1:t-1}}) &= \int p(\vect{z_t}, \vect{z_{t-1}} \lvert \vect{y_{1:t-1}}) d\vect{z_{t-1}} \\
&= \Ncal(\vect{F_t} \vect{\mu_{t-1 \lvert t-1}}, \vect{F_t} \vect{\Sigma_{t-1 \lvert t-1}} \vect{F_t}^\top) \\
&= \Ncal(\vect{\mu}', \vect{\Sigma}')
\end{align*}
where 
$$\vect{\mu}' = \vect{F_t} \vect{\mu_{t-1 \lvert t-1}}, \vect{\Sigma}' = \vect{F_t} \vect{\Sigma_{t-1 \lvert t-1}} \vect{F_t}^\top$$
3. Constructing the joint distribution $p(\vect{y_t}, \vect{x_{t-1}} \lvert \vect{y_{1:t-1}})$
Given the above $p(\vect{z_t} \lvert \vect{y_{1:t-1}})$ and applying the first result again, we attain 
$$
\begin{align*}
p(\vect{y_t}, \vect{z_t} \lvert \vect{y_{1:t-1}}) &= p(\vect{y_t} \lvert \vect{z_t}, \vect{y_{1:t-1}}) p(\vect{z_t} \lvert \vect{y_{1:t-1}}) \\
&= p(\vect{y_t} \lvert \vect{z_t}) p(\vect{z_t} \lvert \vect{y_{1:t-1}}) \quad (\text{Markov property}) \\
&= \Ncal(\vect{y_t} \lvert \vect{H_t} \vect{z_{t}}, \vect{R_t})\Ncal(\vect{z_{t}} \lvert \vect{\mu_{t \lvert t-1}}, \vect{\Sigma_{t \lvert t-1}}) \\
&= \Ncal \left(\begin{pmatrix} \vect{z_{t}} \\ \vect{y_t} \end{pmatrix} \lvert \vect{\mu}'', \vect{\Sigma}'' \right)
\end{align*}
$$
where 
$$
\begin{align*}
\vect{\mu}'' &= \begin{pmatrix} \vect{\mu_{t \lvert t-1}} \\ \vect{H_t} \vect{\mu_{t \lvert t-1}} \end{pmatrix} \\
\vect{\Sigma}'' &= \begin{pmatrix} \vect{\Sigma_{t-1 \lvert t-1}} & \vect{\Sigma_{t \lvert t-1}} \vect{H_t}^\top  \\
\vect{H_t} \vect{\Sigma_{t \lvert t-1}} & \vect{H_t} \vect{\Sigma_{t \lvert t-1}} \vect{H_t}^\top \end{pmatrix}
\end{align*}
$$
4. Convert the joint distribution into conditional one using the second result as follows:
$$
\begin{align*}
p(\vect{z_t} \lvert \vect{y_t}, \vect{y_{1:t-1}}) &= \Ncal(\vect{z_{t}} \lvert \vect{\mu_{t \lvert t}}, \vect{\Sigma_{t \lvert t}}) \\
\vect{\mu_{t \lvert t}} &= \vect{\mu_{t \lvert t-1}} + \vect{\Sigma_{t \lvert t-1}} \vect{H_t}^\top (\vect{H_t} \vect{\Sigma_{t \lvert t-1}} \vect{H_t}^\top)^{-1} (\vect{y_t} - \vect{H_t} \vect{\mu_{t \lvert t-1}} \\
&= \vect{\mu_{t \lvert t-1}} + \vect{K_t} \vect{e_t}\\
\vect{\Sigma_{t \lvert t}} &= \vect{\Sigma_{t \lvert t-1}} - \vect{\Sigma_{t \lvert t-1}} \vect{H_t}^\top (\vect{H_t} \vect{\Sigma_{t \lvert t-1}} \vect{H_t}^\top)^{-1}  \vect{H_t} \vect{\Sigma_{t \lvert t-1}} \\
&=  \vect{\Sigma_{t \lvert t-1}} - \vect{K_t} \vect{H_t} \vect{\Sigma_{t \lvert t-1}} \\
&= \vect{\Sigma_{t \lvert t-1}} - \vect{K_t} \vect{S_t} \vect{K_t}^\top
\end{align*}
$$
where
$$
\begin{align*}
\vect{e_t} &= \vect{y_t} - \vect{H_t} \vect{\mu_{t \lvert t-1}} \\
\vect{S_t} &= \vect{H_t} \vect{\Sigma_{t \lvert t-1}}^{-1} \vect{H_t}^\top + \vect{R_t} \\
\vect{K_t} &=  \vect{\Sigma_{t \lvert t-1}} \vect{H_t}^\top \vect{S_t}^{-1}
\end{align*}
$$

Manipulating Gaussian random variables to derive the results can be quite complex and time-consuming. However, the beauty of Gaussian distributions lies in their algebraic simplicity. Calculations involving Gaussian variables often turn out to be surprisingly straightforward. Furthermore, I plan to share some code demonstrating these techniques in the future if possible.





