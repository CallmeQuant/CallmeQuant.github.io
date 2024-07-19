---
layout: post
title: "Boostrapping with Extension to Markov Chain"
author: "Binh Ho"
categories: Statistics
blurb: ""
img: ""
tags: []
<!-- image: -->
---


## Motivation and Algorithm Sketch
In the context of forecasting intermittent demands, many bootstrapping approaches have been proposed to overcome the obstacles with this specific pattern of time series, which is not efficiently manipulated by many parametric models (Shenstone and Hyndman, 2003). The boostrapping approach propose by Willemain et al., (2004) is the most attracted and well-approved in both academic literature and real-world applications. To improve the accuracy of forecasting procedures of products posed highly intermittent patterns, Willemain devised the simple boostrapping method by plug in two features (extensions), the use of discrete-time Markov Chain and the process of "Jittering". 
	
Concerning conventional boostrapping, which was developed by Efron (1979), the method involves repetitively sampling with replacement from the original data set, thus construct independent boostrap replications, in order to estimate the empirical distribution of the demand (or sale volumes under our concern). Nevertheless, two significant drawbacks acknowledged in this method are: First, the possibility of any autocorrelations presenting in the data violates the assumption of independence amongst observations and Second, the generated values of the reconstructed distribution may not be observed, yet still compatible with the earlier distribution.
	
To transcend these limitations, Willemain 's WSS method utilizes the Markov chain to explicitly capture the autocorrelation between occurrences of demand and non-occurrences by evaluating the empirical distribution of the transition matrix. We will first summarize the incorporation of the dependency of current demand state on the previous demand occurrence status as follows:
+ Estimate the transition probability (in this context, it will be a $2 \times 2$ matrix) between two states defined above.
+ Based upon this estimated transition probabilities and the **last** observed demand, generate a sequence of zero/non-zero values for the entire forecast horizon. 
	
Hence, within the process above, we try to encompass the conditional structure of the occurrence (or non-occurrence) with respect to the previous periods. After generating the future sequence of demands' states, we will only perform resampling demand size restricted to the previous periods where demands occurred and random selections will be collected based on these previous non-zero demands. As previously concerned, possible values that have not been observed in the past data set might be appeared in the future; thereby, need to be accounted in the boostrapped resamples. Willemain proposed a modification to this issue by introducing a "jittering" process which permits a greater variation around the selected positive values. The flow of the jittering procedures is as below:
	
+ Replace all values where demand occurred with values from a set of positive previous demand sizes. 
+ Jittering the values with the deviated realization of standard normal $Z$. If we assume that the forecast is $X$ and $X^{\ast}$ then $X$ would be:

$$
X = 
\begin{cases}
& 1 + \text{INT}(X^{\ast} + Z\sqrt{X^{\ast}}), \quad \text{if } X \gt 0 \\
& 0, \quad \text{otherwise} 
\end{cases}
$$
	
We repeat the process multiple times (Willemain et al., 2004 used $1,000$ replications to obtain the empirical distribution). As we clearly inspect, the design of "jittering" process is to shift up or down the primary values by a random quantity which equals to $\sqrt{X^{\ast}}Z$. Since this generated quantity could reduce the original values to be below zero, we restrict them to be $X$ if negative values are returned. Finally, we average the values obtained for each future time step $h$ to obtain is each expected values for future horizon. 

## Implementation
We will use the package [Markovchain](https://cran.r-project.org/web/packages/markovchain/vignettes/an_introduction_to_markovchain_package.pdf) by Spedicato (2022) to estimate the transition probabilities using MLE method. Then, we will follow the step from Willemain's method to forecast the future demand based on resampling and **jittering**. The implementation is given in R language and I have prepared [a sample of data](https://github.com/CallmeQuant/Boostrapping-Markov-Chain/blob/main/boostrap_dat.RData) in the **.Rdata** format which is extracted from the [M5 Forecasting data set](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) on Kaggle.
```r
library(dplyr)
# Loading RData file 
load(file = "boostrap_dat.RData")
# Discrete Time Markov Chain 
observed_df <- data_final %>%
  filter(id == "HOBBIES_1_192_WI_2"  & dates < "2016-04-18") %>%
  mutate(State = case_when(sales == 0 ~ "Zero",
                           sales > 0 ~ "Positive")) %>%
  select(sales, State)
# Estimating the  Markov Chain 
state <- observed_df %>% pull(State) %>%
  as.character()
 
library(markovchain)
mc <- markovchainFit(state, method = "laplace")$estimate
# Examine the property of Markov Chain 
verifyMarkovProperty(state)
assessStationarity(state, 10)
# Predict 7 days ahead 
pred <- predict(object = mc, newdata = "Zero", n.ahead = 7)
# Extracting df with positive sales only
positive_df <- data_final %>%
  filter(id == "HOBBIES_1_192_WI_2" & dates < "2016-04-18") %>%
  filter(sales > 0)
# Creating a sample with historical positive sales 
sample_positive_sale <- positive_df %>% pull(sales) %>%
  unique()
# Retrieving last observation
lastobs <- data_final %>%
  filter(id == "HOBBIES_1_192_WI_2" & dates == "2016-04-17") %>% pull(sales)
lastobs <- if_else(lastobs == 0, "Zero", "Positive")
mc.sim <- function(trans_mat, num.iters = 1000, n_ahead = 7, method = c("modified","simple")){
  type = method[1]
  states <- numeric(n_ahead) # Not Including the last observed demand
  states[1] <- lastobs
  sample_positive_sale <- positive_df %>% pull(sales) %>%
  unique()
  result <- matrix(0, nrow = num.iters, ncol = n_ahead)
  for (i in 1:num.iters){
    fc_seq <- predict(object = trans_mat, newdata = lastobs, n.ahead = 7)
    # if (lastobs == "Positive"){
    #   fc_seq[i] <- ifelse(runif(1) < trans_mat[1], )
    # }
    for (j in 1:n_ahead){
      if (fc_seq[j] == "Zero") {fc_seq[j] = as.numeric(0)}
      if (fc_seq[j] == "Positive"){
        x = sample(sample_positive_sale, 1, replace= TRUE)
        if (type == "Simple"){
          fc_seq[j] = 1 + as.integer(x + sqrt(x) * rnorm(1))
          if (fc_seq[j] < 0) {fc_seq[j] = x}
        else {
          fc_seq[j] = as.integer(0.5 + x + sqrt(x) * rnorm(1))
          if (fc_seq[j] < 0) {fc_seq[j] = 1}
        }
        }
      }
    }
    result[i, ] <- fc_seq
  }
  result <- apply(result, 2, as.numeric)
  result_bs <- colMeans(result)
  df <- tibble(Forecast = result_bs)
  return(df)
}
# Given at 2016-04-17, we have a postive sales => Simulate for 7 day ahead
result <- mc.sim(mc, num.iters = 1000, n_ahead = 7, method = "modified") %>%
  bind_cols(data_final %>%
  filter(id == "HOBBIES_1_192_WI_2" & dates > "2016-04-17") %>% select(sales))
ZAPE <- function(actual, pred){
  loss <- 0
  y <- actual
  yhat <- pred
  for (i in 1:length(y)){
    if (y[i] == 0){
      loss <- loss + yhat[i]
    }
    else{
      loss <- loss + abs((y[i] - yhat[i])/y[i])
    }
  }
  return((loss / length(y)) * 100)
}
print(result)
ZAPE(result$sales, result$Forecast)
```

As we could see from the result below, the method works really well under the circumstance of intermittency since the probability of transition from demand to zero demand and zero-demand to itself is really high (nearly $1$). Hence, given that transition matrix, if the last day we observe a non-occurence demand event, it is highly possible that the next day would be the same and vice versa, if we observe a demand on the last day, we will more likely to end up being zero sale on the next day.

## Appendix: Derivation of MLE of Markov Chain 

This section would be additional for anyone interested in formal derivation of the Markov chain MLE estimation. Here we will follow the proof given in the note of [Cosma Shalizi](https://www.stat.cmu.edu/~cshalizi/) (one of the my most favorite statisticians). This proof assume that you are familiar with basic structure of Markov Chain and its properties. 

Supposed that we have observed a sample from the chain, the realization of the random variable $X_{1}^{n}$, which is denoted by $x_{1}^{n} \neq x_1, x_2, \dots, x_{n}$. Then we would like to obtain the probability that we will observe such data points, or the joint probability of the sequence $x_{1}^{n}$

$$
\begin{align}
P\left(X_{1}^{n} = x_{1}^{n}\right) &= P\left(X_1 = x_1, X_2 = x_2, \dots,X_n = x_n\right) \\
& = P\left(X_1 = x_1\right) \prod_{t  = 2}^{n} P\left(X_t = x_t \mid X_{1}^{t-1} = x_{1}^{t-1}\right) \\
& = P\left(X_1 = x_1\right) \prod_{t  = 2}^{n} P\left(X_{t} = x_{t} \mid X_{t-1} = x_{t-1}\right)
\end{align}
$$

The second expression is the conditional probability of the process being at state $x_t$ given all past observations. On the other hand, the third expression is derived based on one of the most notable properties of a discrete time Markov chain is the first order Markov assumption. In other words, it is assumed that the model hold the *Markov property*:

$$
P\left(X_{t} = j \mid X_{t-1} = i, X_{t-2} = i_{t-1}, \ldots, X_1 = i_1\right) = P\left(X_{t} = j \mid X_{t-1} = i\right)
$$
	
As our target is to determine the MLE estimation for our transition probabilities, then we rewrite the above expression in terms of $p_{ij}$ to arrive at the likelihood function for the 

$$
L(p) = P(X_1 = x_1) \prod_{t  = 2}^{n} p_{x_{t-1} x_{t}}
$$

Assumed that $n_{ij}$ be the number of times observed state transits from $i$ to $j$ in our sample $X_{1}^{n}$, then we rewrite the \ref{eq:likelihood} in terms of $n_{ij}$.

$$
L(p) = P(X_1 = x_1) \prod_{i=1}^{h} \prod_{j=1}^{h} p_{ij}^{n_{ij}}
$$

Maximize the above expression \textit{w.r.t} the $p_{ij}$ by transforming it into logarithms (since it would be easier to deal with sums rather than product), taking derivatives, and setting it equal to $0$. However, it would be crucial to consider one special property of the Markov chain: it is stochastic row unitary. That is,

$$ 
\begin{align}
p \cdot \boldsymbol{1}^\intercal & = \mathbf{1}^\intercal \\
& \text{or} \\
\sum_{j} p_{ij} & = 1, \quad \text{ for each } i
\end{align}
$$

where $\mathbf{1}^\intercal$ denotes the row vector with its entries equal to one.
	
It is clear that we have an equality constraint in our formulation and a formal way to tackle this problem would be reforming it to a standard optimization problem with Lagrange multipliers. However, in our case, there exists a method which is more effortless to handle such issue: Eliminating parameters. 

The first thing to do is taking the logs of the likelihood function

$$
\mathcal{L}(p) =\log\left(P(X_1 = x_1)\right) + \sum_{i,j} n_{ij} \log p_{ij}
$$

then, randomly choose one of the transition probabilities to express the others. For easy manipulation, we would collect the probability of transiting to state $1$. Thus, for each $i$, $p_{i1} = 1 - \sum_{j=2}^{m} p_{ij}$. Subsequently, differentiating the log-likelihood with respect to $p_{ij}$ while ignoring any partial derivatives relating to $\frac{\partial}{\partial P_{i1}}$

$$
\frac{\partial \mathcal{L}}{\partial p_{ij}} = \frac{n_{ij}}{p_{ij}} - \frac{n_{i1}}{p_{i1}} 
$$

Letting this equal to zero, then we have the following relations,

$$
\begin{align}
\frac{n_{ij}}{\hat p_{ij}} &= \frac{n_{i1}}{\hat p_{i1}} \\
& \Updownarrow \\
\frac{n_{ij}}{n_{i1}} &= \frac{\hat p_{ij}}{\hat p_{i1}}
\end{align}
$$

The above relations hold for all $j \neq 1$; thereby, we could conclude that 

$$
\begin{align}
\hat p_{ij} &= \frac{n_{ij}}{\frac{n_{i1}}{\hat p_{i1}}} \\
& = \frac{n_{ij}}{n_{i}} \\
& = \frac{n_{ij}}{\sum_{j=1}^{m} n_{ij}}
\end{align}
$$

where the last relation could be deduced as follows: $n_{i1}$ represent the number of observed consecutive transitions from state $i$ to state $1$ in random variable $X_{n}^{1}$ or expressing in terms of probability  $n_{i1} \propto P(X_{t} = i, X_{t+1} = 1)$, while $\hat p_{i}$ indicates the chance that we move to other states starting from state $1$. Hence, applying the Bayes'rule in terms of proportional expression, then the fraction between the former and latter would result in $P(X_{t} = i)$, implying the number of times that state $i$ is recorded, which is proportional to $n_{i}$ or $\sum_{j=1}^{m} n_{ij}$. It is noteworthy that if $n_{i} = 0$, or the state $i$ is not appeared in the chain except for the last position, then we would formally set all probabilities of the transition from state $i$ to any state $j$ ($j \neq i$) to be zero, $\hat p_{ij} = 0$. Thus, it turns out that $\hat p_{ii} = 1$.

# References
1. Willemain, Thomas \& Smart, Charles & Schwarz, Henry. (2004). A new approach to forecasting intermittent demand for service parts inventories. International Journal of Forecasting. 20. 375-387. 10.1016/S0169-2070(03)00013-X. 

2. Spedicato, Giorgio & Signorelli, Mirko. (2014). The markovchain Package: A Package for Easily Handling Discrete Markov Chains in R. 

3. https://www.stat.cmu.edu/~cshalizi/462/lectures/06/markov-mle.pdf
