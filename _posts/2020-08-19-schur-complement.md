---
layout: post
title: "Schur complements"
author: "Binh Ho"
categories: Computational statistics
blurb: "Schur complements are quantities that arise often in linear algebra in the context of block matrix inversion. Here, we review the basics and show an application in statistics."
img: ""
tags: []
<!-- image: -->
---

Schur complements are quantities that arise often in linear algebra in the context of block matrix inversion. Here, we review the basics and show an application in statistics.


## LDU decomposition

Consider a block matrix $M$:

$$M = \begin{bmatrix} A & B \\ C & D \\ \end{bmatrix}$$

where $A$ is $p \times p$, $B$ is $p \times q$, $C$ is $q \times p$, and $D$ is $q \times q$. Suppose we were interested in inverting $M$. One way would be to try to invert $M$ directly without capitalizing on its block structure. However, as we'll see, we can find a more clever way to find $M^{-1}$.

To start, let's perform an LDU decomposition on $M$. If we right-multiply $M$ by 

$$L = \begin{bmatrix} I_p & 0 \\ -D^{-1} C & I_q \\ \end{bmatrix},$$ we get

\begin{align} ML &=  \begin{bmatrix} A & B \\\ C & D \\\ \end{bmatrix}  \begin{bmatrix} I_p & 0 \\\ -D^{-1} C & I_q \\\ \end{bmatrix} \\\ &= \begin{bmatrix} A - BD^{-1} C & B \\\ 0 & D  \end{bmatrix} \\\ \end{align}

This further decomposes as 

\begin{align} \begin{bmatrix} A - BD^{-1} C & B \\\ 0 & D  \end{bmatrix} &= \begin{bmatrix} I_p & BD^{-1} \\\ 0 & I_q \end{bmatrix} \begin{bmatrix} A - BD^{-1} C & 0 \\\ 0 & D  \end{bmatrix} \\\ \end{align}

Thus, we can rewrite $M$ as

$$M = \begin{bmatrix} I_p & BD^{-1} \\ 0 & I_q \end{bmatrix} \begin{bmatrix} A - BD^{-1} C & 0 \\ 0 & D  \end{bmatrix} \begin{bmatrix} I_p & 0 \\ D^{-1} C & I_q \\ \end{bmatrix}.$$

Notice that at this point, we have decomposed $M$ into to a upper-diagonal matrix, a diagonal matrix, and an lower-diagonal matrix.

## Inverting $M$

Recall that for two matrices $A$ and $B$, the inverse of their product can be written as

$$(AB)^{-1} = B^{-1} A^{-1}.$$

For three matrices, we then have

$$(ABC)^{-1} = ((AB)C)^{-1} = C^{-1} (AB)^{-1} = C^{-1} B^{-1} A^{-1}.$$

Then, the inverse of $M$ can be written as

$$M^{-1} = \begin{bmatrix} I_p & 0 \\ D^{-1} C & I_q \\ \end{bmatrix}^{-1} \begin{bmatrix} A - BD^{-1} C & 0 \\ 0 & D  \end{bmatrix}^{-1} \begin{bmatrix} I_p & BD^{-1} \\ 0 & I_q \end{bmatrix}^{-1}.$$

For the first and third matrices, we have fairly simple inverses (just negate the lower-left and upper-right blocks, respecitively):

$$\begin{bmatrix} I_p & 0 \\ D^{-1} C & I_q \\ \end{bmatrix}^{-1} = \begin{bmatrix} I_p & 0 \\ -D^{-1} C & I_q \\ \end{bmatrix}$$

and

$$\begin{bmatrix} I_p & BD^{-1} \\ 0 & I_q \end{bmatrix}^{-1} = \begin{bmatrix} I_p & -BD^{-1} \\ 0 & I_q \end{bmatrix}.$$

For the middle matrix, the inverse is simply another block diagonal matrix with each block inverted:

$$\begin{bmatrix} A - BD^{-1} C & 0 \\ 0 & D  \end{bmatrix}^{-1} = \begin{bmatrix} (A - BD^{-1} C)^{-1} & 0 \\ 0 & D^{-1} \end{bmatrix}$$

Plugging in and simplifying, we have


\begin{align} M^{-1} &= \begin{bmatrix} I_p & 0 \\\ -D^{-1} C & I_q \\\ \end{bmatrix} \begin{bmatrix} (A - BD^{-1} C)^{-1} & 0 \\ 0 & D^{-1}  \end{bmatrix}  \begin{bmatrix} I_p & BD^{-1} \\\ 0 & I_q \end{bmatrix} \\\ &= \begin{bmatrix} (A - BD^{-1} C)^{-1} & 0 \\\ -D^{-1} C (A - BD^{-1} C)^{-1} & D^{-1} \\\ \end{bmatrix} \begin{bmatrix} I_p & BD^{-1} \\\ 0 & I_q \end{bmatrix} \\\ &= \begin{bmatrix} (A - BD^{-1} C)^{-1} & (A - BD^{-1} C)^{-1} BD^{-1} \\\ -D^{-1} C (A - BD^{-1} C)^{-1} & -D^{-1} C (A - BD^{-1} C)^{-1} BD^{-1} + D^{-1} \\\ \end{bmatrix} \\\ \end{align}

Notice that to get the inverse of $M$, we now only need the the inverse of $D$ and the inverse of another quantity, $(A - BD^{-1} C)^{-1}$. This second quantity is known as the **Schur complement**.




## Multivariate Guassians

To illustrate the usefulness and prevalence of Schur complements, let's take a look at an application of them in statistics. 

Consider two Gaussian random vectors $\mathbf{X}$ and $\mathbf{Y}$ of length $p$ and $q$, respectively, where we assume for the sake of simplicity that their means are 0:

\begin{align} \mathbf{X} &\sim \mathcal{N}\_p(\mathbf{0}, \boldsymbol{\Sigma}\_X) \\\ \mathbf{Y} &\sim \mathcal{N}\_q(\mathbf{0}, \boldsymbol{\Sigma}\_Y). \\\ \end{align}

Their joint distribution is then

$$(\mathbf{X}, \mathbf{Y}) \sim \mathcal{N}_{p + q}(\mathbf{0}, \boldsymbol{\Sigma})$$

where $\boldsymbol{\Sigma}$ has a block structure:

$$\boldsymbol{\Sigma} = \begin{bmatrix} \boldsymbol{\Sigma}_{XX} & \boldsymbol{\Sigma}_{XY}^\top \\ \boldsymbol{\Sigma}_{XY} & \boldsymbol{\Sigma}_{YY} \\ \end{bmatrix}.$$

Denote the precision matrix as $\boldsymbol{\Omega} = \boldsymbol{\Sigma}^{-1}$, and give it a similar block structure:

$$\boldsymbol{\Omega} = \begin{bmatrix} \boldsymbol{\Omega}_{XX} & \boldsymbol{\Omega}_{XY}^\top \\ \boldsymbol{\Omega}_{XY} & \boldsymbol{\Omega}_{YY} \\ \end{bmatrix}.$$

Using the Schur complement result above, we already know that 

$$\boldsymbol{\Omega}_{XX} = (\boldsymbol{\Sigma}_{XX} - \boldsymbol{\Sigma}_{XY}^\top \boldsymbol{\Sigma}_{YY}^{-1} \boldsymbol{\Sigma}_{XY})^{-1}$$


Suppose we're interested in the conditional distribution of $\mathbf{X} \| \mathbf{Y} = \mathbf{y}$. Then we can write the conditional density as

\begin{align} f(\mathbf{x} \| \mathbf{y}) &= 2\pi^{-p/2} \|\boldsymbol{\Omega}\|^{1/2} \exp\left( -\frac12 \begin{bmatrix} \mathbf{x} \\\ \mathbf{y} \end{bmatrix}^\top \begin{bmatrix} \boldsymbol{\Omega}\_{XX} & \boldsymbol{\Omega}\_{XY}^\top \\ \boldsymbol{\Omega}\_{XY} & \boldsymbol{\Omega}\_{YY} \\ \end{bmatrix} \begin{bmatrix} \mathbf{x} \\\ \mathbf{y} \end{bmatrix} \right) \\\ &\propto \exp\left( -\frac12 \begin{bmatrix} \mathbf{x}^\top \boldsymbol{\Omega}\_{XX} + \mathbf{y}^\top \boldsymbol{\Omega}\_{XY} \\\ \mathbf{x}^\top \boldsymbol{\Omega}\_{XY}^\top + \mathbf{y}^\top \boldsymbol{\Omega}\_{YY} \end{bmatrix}^\top \begin{bmatrix} \mathbf{x} \\\ \mathbf{y} \end{bmatrix} \right) \\\ &\propto \exp\left( -\frac12 (\mathbf{x}^\top \boldsymbol{\Omega}\_{XX} \mathbf{x} + \mathbf{y}^\top \boldsymbol{\Omega}\_{XY} \mathbf{x} + \mathbf{x}^\top \boldsymbol{\Omega}\_{XY}^\top \mathbf{y} + \mathbf{y}^\top \boldsymbol{\Omega}\_{YY} \mathbf{y})  \right) \\\ \end{align}

Ignoring the terms that don't depend on $\mathbf{x}$, we have

\begin{align} f(\mathbf{x} \| \mathbf{y}) &\propto \exp\left( -\frac12 (\mathbf{x}^\top \boldsymbol{\Omega}\_{XX} \mathbf{x} + \mathbf{y}^\top \boldsymbol{\Omega}\_{XY} \mathbf{x} + \mathbf{x}^\top \boldsymbol{\Omega}\_{XY}^\top \mathbf{y}) \right) \\\ &\propto \exp\left( -\frac12 \mathbf{x}^\top \boldsymbol{\Omega}\_{XX} \mathbf{x} - \mathbf{x}^\top \boldsymbol{\Omega}\_{XY}^\top \mathbf{y}  \right) \\\ \end{align}

Putting this into a form that allows us to read off the covariance, we have

\begin{align} \exp\left( -\frac12 \mathbf{x}^\top \boldsymbol{\Omega}\_{XX} \mathbf{x} - \mathbf{x}^\top \boldsymbol{\Omega}\_{XY}^\top \mathbf{y}  \right) &= \exp\left( -\frac12 (\mathbf{x} - \boldsymbol{\Omega}\_{XX}^{-1} \boldsymbol{\Omega}\_{XY} \mathbf{y})^\top  \boldsymbol{\Omega}\_{XX}  (\mathbf{x} - \boldsymbol{\Omega}\_{XX}^{-1} \boldsymbol{\Omega}\_{XY} \mathbf{y}) \right). \\\ \end{align}

Now, we can see that the covariance of $\mathbf{X} \| \mathbf{Y} = \mathbf{y}$ is $\boldsymbol{\Omega}\_{XX} = (\boldsymbol{\Sigma}\_{XX} - \boldsymbol{\Sigma}\_{XY}^\top \boldsymbol{\Sigma}\_{YY}^{-1} \boldsymbol{\Sigma}\_{XY})^{-1}$.


## References

- Wikipedia [entry on Schur complements](https://www.wikiwand.com/en/Schur_complement)
- Prof. Jean Gallier's [notes on the Schur complement](https://www.cis.upenn.edu/~jean/schur-comp.pdf)
- Uhler, Caroline. "Gaussian graphical models: an algebraic and geometric perspective." arXiv preprint arXiv:1707.04345 (2017).
- Terry Tao's [post on the Schur complement](https://terrytao.wordpress.com/tag/schur-complement/)
