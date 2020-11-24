# Econometrics I Cheat Sheet

## Basics

**Covariance**
$$
cov_{x, y}=\frac{\sum\left(x_{i}-\bar{x}\right)\left(y_{i}-\bar{y}\right)}{N-1}
$$

***

**Pearson correlation coefficient**
$$
r_{y, \widehat{y}}:=\frac{\operatorname{cov}(y, \widehat{y})}{\sqrt{\operatorname{var}(y) \operatorname{var}(\hat{y})}}
$$


## Simple linear regression (SLR) model

**Population model**
$$
y=\beta_{0}+\beta_{1} x+u
$$

**The dependent variable expressed in terms of estimates**
$$
y_{i}=\hat{\beta}_{0}+\hat{\beta}_{1} x_{i}+\hat{u}_{i}
$$
**Regression residuals**
$$
\hat{u}_{i}=y_{i}-\hat{y}_{i}
$$
**Ordinary least squares (OLS) Estimator for the SLR model**
$$
\hat{\beta}_{1}=\frac{\sum_{i=1}^{n}\left(y_{i}-\bar{y}\right)\left(x_{i}-\bar{x}\right)}{\sum_{i=1}^{n}\left(x_{i}-\bar{x}\right)^{2}} = \frac{s_{y}}{s_{x}} r_{x y}, \quad 
\hat{\beta}_{0}=\bar{y}-\hat{\beta}_{1} \bar{x}
$$
**Relationship between $\hat{\beta}_1$ and $\beta_1$**
$$
\hat{\beta}_{1}=\beta_{1}+\frac{\sum_{i=1}^{N}\left(X_{i}-\bar{X}\right) u_{i}}{N \cdot s_{X}^{2}}
$$
**Standard error of the slope estimator**
$$
\operatorname{se}(\tilde{\beta}_{1})=\frac{\hat{\sigma}}{\sqrt{n} \operatorname{sd}(x_{j}) } \\ \\
$$


**TSS = SSE + SSR**
$$
\begin{aligned}
\mathrm{SST} &\equiv \sum_{i=1}^{n}\left(y_{i}-\bar{y}\right)^{2} \\   
\mathrm{SSR} &\equiv \sum_{i=1}^{n} \hat{u}_{i}^{2} \\
\mathrm{SSE} &\equiv \sum_{i=1}^{n}\left(\hat{y}_{i}-\bar{y}\right)^{2}
\end{aligned}
$$
**R-squared of the regression**
$$
\mathrm{R}^{2}=\frac{\mathrm{TSS}-\mathrm{SSR}}{\mathrm{TSS}}=1-\frac{\mathrm{SSR}}{\mathrm{TSS}}
$$

***

#### Assumptions of the SLR model

**SLR.1 (linear in parameters)**

In the population model, the dependent variable, *y*, is related to the independent variable, *x*, and the error (or disturbance), *u*, as$y=\beta_{0}+\beta_{1} x+u$ where b0 and b1 are the population intercept and slope parameters, respectively.

**SLR.2 (Random Sampling)**

We have a random sample of size *n*, $\left\{\left(x_{i}, y_{i}\right): i=1,2, \ldots, n\right\}$, following the population model in Assumption SLR.1.

**SLR.3 (Sample Variation in the Explanatory Variable)**

The sample outcomes on *x*, namely, $\left\{\left(x_{i}, y_{i}\right): i=1,2, \ldots, n\right\}$, are not all the same value.

**SLR.4 Zzero Conditional Mean)**

The error *u* has an expected value of zero given any value of the explanatory variable
$$
\mathrm{E}(u \mid x)=0
$$
**SLR.5 (Homoskedasticity)**

The error *u* has the same variance given any value of the explanatory variable
$$
\operatorname{Var}(u \mid x)=\sigma^{2}
$$

## Multiple linear regression (MLR) model

**Population model**
$$
y=\beta_{0}+\beta_{1} x_{1}+\beta_{2} x_{2}+\cdots+\beta_{k} x_{k}+u
$$

***

#### Assumptions of the SLR model

**MLR.1 (linear in parameters)**

**MLR.2 (Random Sampling)**

**MLR.3 (no perfect collinearity**)

In the sample (and therefore in the population), none of the independent variables is constant, and there are no *exact linear* relationships among the independent variables.

**MLR.4 (Zero conditional Mean)**

**MLR.5 (Homoskedasticity)**

## Interpretation der Regressionskoeffizienten

![](https://cdn.mathpix.com/snip/images/do7eSZuOM5i-GQQGSY6lirRpB23m57oGneZsdeY65iQ.original.fullsize.png)

