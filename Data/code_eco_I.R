
# Slide 23 ----------------------------------------------------------------
# Loading and depicting data

# Make sure that the working directory is set to the folder the data is located in
marketing <- read.csv("marketing.csv")

# Look at whole dataset
marketing
View(marketing) #Only works in RStudio

# Only look at first few entries
head(marketing)

# Try the same with the other datasets mentioned on slide 23


# Slide 34 ----------------------------------------------------------------

beta0 <- 0.2
beta1 <- -1.8
N <- 100
sigma <- 1

# Generate random covariates
x <- runif(N) + 1

# Generate error term from normal distribution
u <- rnorm(N, 0, sigma)

# Implement the linear model
y <- beta0 + beta1 * x + u

# Plot the resulting data
plot(x, y, col = 2, pch = 16, xlab = "Price", ylab = "Demand",
     ylim = c(-6.5, 1.5), main = bquote(sigma^2 ~ "=" ~ .(sigma^2)))

# Add the true regression line
abline(beta0, beta1, lwd = 4)

# Add lines representing residuals
yhat <- beta0 + beta1 * x
for (i in 1:N) {
  lines(c(x[i], x[i]), c(yhat[i], y[i]))
}


# Slide 39 ----------------------------------------------------------------


n <- 100
beta0tilde <- .2
beta1 <- -1.8
sigma <- sqrt(.01)

x <- sort(runif(n) + 1)
u <- rnorm(n, 0, sigma)
yhat <- beta0tilde * x ^ beta1
y <- yhat * exp(u)

par(mfrow = c(1,2))
plot(x, y, col = 2, pch = 16, xlab = "Price", ylab = "Demand",
     ylim = c(0, 0.35), main = bquote(sigma^2 ~ "=" ~ .(sigma^2)))
lines(x, yhat, lwd = 4)

for (i in 1:n) {
  lines(c(x[i], x[i]), c(yhat[i], y[i]))
}

plot(log(x), log(y), col = 2, pch = 16, xlab = "log(Price)",
     ylab = "log(Demand)",
     main = bquote(sigma^2 ~ "=" ~ .(sigma^2)))
abline(log(beta0tilde), beta1, lwd = 4)

for (i in 1:n) {
  lines(log(c(x[i], x[i])), log(c(yhat[i], y[i])))
}



# Slide 67 ----------------------------------------------------------------

par(mfrow = c(1,1))

beta0 <- .2
beta1 <- -1.8
N <- 100
sigma <- 1

# Generate random covariates
x <- runif(N) + 1

# Generate error term from normal distribution
u <- rnorm(N, 0, sigma)

# Implement the linear model
y <- beta0 + beta1*x + u

# Estimate a linear model
mylm <- lm(y ~ x)

# Plot resulting data and add estimated parameters as title
plot(x, y, main = bquote(hat(beta)[0] ~ "=" ~ .(round(mylm$coef[1], 2)) ~
                           "                  " ~ hat(beta)[1] ~ "=" ~ .(round(mylm$coef[2], 2))),
     xlim = c(1,2), ylim = c(-5.5, 1.5), col = "black")

# Add true regression line
abline(beta0, beta1, lty=2)

# Add estimated regression line
abline(mylm)

# Add legend
legend("topright", legend=c("estimated slope","true slope"), lty=c(1,2))


# Slide 70 ----------------------------------------------------------------

#### This code demonstrates the difference the number of observations makes
beta0 <- .2
beta1 <- -1.8
sigma <- sqrt(.1)

# Set two different sample sizes
N <- 50
N2 <- 400

# This allows two plots to be displayed next to one another
par(mfrow = c(1,2))

x <- runif(N) + 1
u <- rnorm(N, sd = sigma)
y <- beta0 + beta1*x + u

mylm <- lm(y ~ x)

plot(x, y, main = bquote(N ~ "=" ~ .(N) ~ "    " ~ hat(beta)[0] ~ "=" ~ .(round(mylm$coef[1], 2)) ~
                           "    " ~ hat(beta)[1] ~ "=" ~ .(round(mylm$coef[2], 2))),
     xlim = c(1,2), ylim = c(-4.5, -.5), col = "gray")

abline(beta0, beta1, lty=2)
abline(mylm)

legend("topright", legend=c("estimated slope","true slope"), lty=c(1,2))

# Now generate a second, larger data set (with the same underlying true model)
# and estimate the parameters a second time
x2 <- runif(N2) + 1
u2 <- rnorm(N2, sd = sigma)
y2 <- beta0 + beta1*x2 + u2

mylm2 <- lm(y2 ~ x2)

plot(x2, y2, main = bquote(N ~ "=" ~ .(N2) ~ "    " ~ hat(beta)[0] ~ "=" ~ .(round(mylm2$coef[1], 2)) ~
                             "    " ~ hat(beta)[1] ~ "=" ~ .(round(mylm2$coef[2], 2))),
     xlim = c(1,2), ylim = c(-4.5, -.5), col = "gray", xlab = "x", ylab = "y")

abline(beta0, beta1, lty=2)
abline(mylm2)

legend("topright", legend=c("estimated slope","true slope"), lty=c(1,2))


#### This code demonstrates the effect of differing error variances
beta0 <- .2
beta1 <- -1.8
N <- 50

# Set two different error variances
sigma <- sqrt(.1)
sigma2 <- sqrt(.01)

par(mfrow = c(1,2))

x <- runif(N) + 1
u <- rnorm(N, sd = sigma)
y <- beta0 + beta1*x + u

mylm <- lm(y ~ x)

plot(x, y, main = bquote(sigma^2 ~ "=" ~ .(sigma^2) ~ "    " ~ hat(beta)[0] ~ "=" ~ .(round(mylm$coef[1], 2)) ~
                           "    " ~ hat(beta)[1] ~ "=" ~ .(round(mylm$coef[2], 2))),
     xlim = c(1,2), ylim = c(-4.5, -.5), col = "red")

abline(beta0, beta1, lty=2)
abline(mylm)

legend("topright", legend=c("estimated slope","true slope"), lty=c(1,2))

x2 <- x
# Note that now the error term is generated with a different variance (sigma2)
u2 <- rnorm(N, sd = sigma2)
y2 <- beta0 + beta1*x2 + u2

mylm2 <- lm(y2 ~ x2)

plot(x2, y2, main = bquote(sigma^2 ~ "=" ~ .(sigma2^2) ~ "    " ~ hat(beta)[0] ~ "=" ~ .(round(mylm2$coef[1], 2)) ~
                             "    " ~ hat(beta)[1] ~ "=" ~ .(round(mylm2$coef[2], 2))),
     xlim = c(1,2), ylim = c(-4.5, -.5), col = "red", xlab = "x", ylab = "y")

abline(beta0, beta1, lty=2)
abline(mylm2)

legend("topright", legend=c("estimated slope","true slope"), lty=c(1,2))


#### This code demonstrates the effect of larger variability of x
beta0 <- .2
beta1 <- -1.8
N <- 50
sigma <- sqrt(.1)

# Set two different variances for x
sigmax1 <- sqrt(.1)
sigmax2 <- sqrt(.01)

# Variance of uniform distribution is range^2/12
range1 <- sqrt(12 * sigmax1^2)
range2 <- sqrt(12 * sigmax2^2)

par(mfrow = c(1,2))

# Note that now x comes from a uniform distribution with differing range
x <- runif(N, 1.5 - range1/2, 1.5 + range1/2)
u <- rnorm(N, sd = sigma)
y <- beta0 + beta1*x + u

mylm <- lm(y ~ x)

plot(x, y, main = bquote(sigma[X]^2 ~ "=" ~ .(sigmax1^2) ~ "    " ~ hat(beta)[0] ~ "=" ~ .(round(mylm$coef[1], 2)) ~
                           "    " ~ hat(beta)[1] ~ "=" ~ .(round(mylm$coef[2], 2))),
     xlim = c(.9,2.1), ylim = c(-4.5, -.5), col = "red")

abline(beta0, beta1, lty=2)
abline(mylm)

legend("topright", legend=c("estimated slope","true slope"), lty=c(1,2))

# Note that now x comes from a uniform distribution with differing range
x2 <- runif(N, 1.5 - range2/2, 1.5 + range2/2)
u2 <- rnorm(N, sd = sigma)
y2 <- beta0 + beta1*x2 + u2

mylm2 <- lm(y2 ~ x2)

plot(x2, y2, main = bquote(sigma[X]^2 ~ "=" ~ .(sigmax2^2) ~ "    " ~ hat(beta)[0] ~ "=" ~ .(round(mylm2$coef[1], 2)) ~
                             "    " ~ hat(beta)[1] ~ "=" ~ .(round(mylm2$coef[2], 2))),
     xlim = c(.9,2.1), ylim = c(-4.5, -.5), col = "red", xlab = "x", ylab = "y")

abline(beta0, beta1, lty=2)
abline(mylm2)

legend("topright", legend=c("estimated slope","true slope"), lty=c(1,2))


# Slide 95 ----------------------------------------------------------------

# This is an example of how to load a data set and estimate a linear model
chicken <- read.csv("chicken.csv")

# Estimate the multiple linear regression model
# More covariates are added by combining them with a '+' sign
chick_lm <- lm(consum ~ income + pchick + pbeef + ppork, data = chicken)

# Simply printing chick_lm gives you a brief overview of the estimated parameters
chick_lm

# More information can be gained by using the summary() function
summary(chick_lm)

# If one wants to estimate a multiple log-linear model, the covariates and the response can
# be wrapped in the log() function
chick_log <- lm(log(consum) ~ log(income) + log(pchick) + log(pbeef) + log(ppork), data = chicken)
summary(chick_log)



# Slide 107 ---------------------------------------------------------------

yieldus <- read.csv("yieldus.csv")

# To bring notation in line with slides, create y, x1, x2 and x3
y <- yieldus$Y3
x1 <- yieldus$Y1
x2 <- yieldus$Y60
x3 <- x2 - x1

# Estimate model without spread
yield_lm1 <- lm(y ~ x1 + x2)
summary(yield_lm1)
# All parameters estimated without hiccups!

# Now we add the spread
yield_lm2 <- lm(y ~ x1 + x2 + x3)
summary(yield_lm2)
# The parameter for x3 could not be estimated, due to perfect multicollinearity
# Check what happens if you change the order of the covariates in R (e.g. lm(y ~ x3 + x2 + x1))


# Slide 128 ---------------------------------------------------------------

# This is the same data set and model as in the code for slide 95
chicken <- read.csv("chicken.csv")
chick_lm <- lm(consum ~ income + pchick + pbeef + ppork, data = chicken)

# Residuals are obtained with the residuals() function
resids <- residuals(chick_lm)

# These can then be easily plotted with plot()
dev.off() # This command clears any previous plots
plot(resids)
# Add a horizontal line at zero as a visual guide
abline(h = 0, lty = 2, col = "grey")

# To get the fitted values from R, there are two functions, fitted() and predict()
# predict() can additionally generate predicted from new data for covariates
fit <- predict(chick_lm)
fit2 <- fitted(chick_lm)

# Both produce the same results
all.equal(fit, fit2)

# However, predict can do more. Here is an example:
newdata <- data.frame(income = 1200, pbeef = 80, pchick = 50, ppork = 55)
predict(chick_lm, newdata = newdata)


# Plot the fitted values against the true values
plot(x = fit, y = chicken$consum, xlab = "Fitted values", ylab = "True values")
# Add f(x) = x function, as a visual guide for how accurate our model is
abline(b = 1, a = 0, lty = 2, col = "grey")
# How could you find the residuals from this plot?


# Slide 134 ---------------------------------------------------------------

# Use same model as slide 128 (above)
# summary() gives you the parameter estimates and the standard errors
summary(chick_lm)

# An alternative way to obtain the standard errors via the vcov() function
# vcov() extracts the variance-covariance matrix of the OLS estimators
var_cov <- vcov(chick_lm)
var_cov

# The square root of the diagonal elements of the variance-covariance matrix are the standard errors
# Note: diag() extracts the elements on the diagonal of a matrix
sqrt(diag(var_cov))


# Slide 143 -----------------------------------------------------------

n <- 100
sigma <- sqrt(.1)
sigmax <- sqrt(1)
xbar <- 0
b0 <- .2
b1 <- -1.8
b2 <- 0
rho <- .7  # correlation of predictors

xrange <- sqrt(12*sigmax^2)
x1 <- runif(n, -xrange/2, xrange/2) + xbar
tmp1 <- runif(n, -xrange/2, xrange/2) + xbar
tmp2 <- rbinom(n, 1, rho)
x2 <- x1*tmp2 + tmp1*(1-tmp2)

y <- b0 + b1*x1 + b2*x2 + rnorm(n, 0, sigma)

mod <- lm(y ~ x1 + x2)
summary(mod)

# Observe that the estimator for b2 is generally different from 0!



# Slide 144/145 -----------------------------------------------------------

# This code generates the plots as seen on slides 144/145
set.seed(1)

reps <- 200
n <- 100
sigma <- sqrt(.1)
sigmax <- sqrt(1)
xbar <- 0
b0 <- .2
b1 <- -1.8
b2 <- 0
rho <- .7  # correlation of predictors

xrange <- sqrt(12*sigmax^2)
x1 <- runif(n, -xrange/2, xrange/2) + xbar
tmp1 <- runif(n, -xrange/2, xrange/2) + xbar
tmp2 <- rbinom(n, 1, rho)
x2 <- x1*tmp2 + tmp1*(1-tmp2)

res <- vector("list", 2)
res[[1]] <- matrix(NA_real_, nrow = reps, ncol = 2)
res[[2]] <- matrix(NA_real_, nrow = reps, ncol = 3)

for (i in 1:reps) {
  y <- b0 + b1*x1 + b2*x2 + rnorm(n, 0, sigma)
  mylm2 <- lm(y ~ x1)
  res[[1]][i,] <- coef(mylm2)
  mylm3 <- lm(y ~ x1 + x2)
  res[[2]][i,] <- coef(mylm3)
}

par(mar = c(3.3,4,2,1.5), mgp = c(2.3,.7,0))

plot(res[[2]][,2], res[[2]][,3], xlab = bquote(paste(hat(beta)[1], " (important variable)")), ylab = bquote(paste(hat(beta)[2], " (redundant variable)")))
mtext(bquote(N ~ "=" ~ .(n) ~ "    " ~ sigma^2 ~ "=" ~
               .(sigma^2) ~ "    " ~ sigma[X[1]]^2 ~ "=" ~ sigma[X[2]]^2 ~ "=" ~
               .(sigmax^2) ~ "    " ~ mu[X[1]] ~ "=" ~ mu[X[2]] ~ "=" ~
               .(xbar) ~  "    " ~ sigma[X[1]*X[2]] ~ "=" ~
               .(rho)),
      cex=1.4)
abline(h = b2, col = "gray")
for (j in 1:3) points(b1, b2, cex = j, col = 2)

ylims <- range(res[[1]][,2], res[[2]][,2])

par(mfrow = c(1,2), mar = c(3.3,4,2,1.5), mgp = c(2.3,.7,0))

mains <- c("One predictor", "Two predictors")
for (i in 1:2) {
  plot(res[[i]][,1], res[[i]][,2], xlab = bquote(paste(hat(beta)[0], "")),
       ylab = bquote(paste(hat(beta)[1], "")), ylim = ylims,
       main = mains[i])
  abline(h = b2, col = "gray")
  for (j in 1:3) points(b0, b1, cex = j, col = 2)
}


#### FROM HERE ON OUT IS NEW CODE
# Slide 165 ---------------------------------------------------------------

# This is the same data set and model as in the code for slide 95
chicken <- read.csv("chicken.csv")
chick_lm <- lm(consum ~ income + pchick + pbeef + ppork, data = chicken)

# summary() function automatically outputs t statistic and p-value
sum <- summary(chick_lm)
sum

# They can be extracted with the following code
t_vals <- summary(chick_lm)[["coefficients"]][, "t value"]
p_vals <- summary(chick_lm)[["coefficients"]][, "Pr(>|t|)"]
t_vals
p_vals

# Slide 183/184 -----------------------------------------------------------
marketing <- read.csv("marketing.csv")
head(marketing)
marketing_lm <- lm(rating ~ price + rq + ju + vo + wa + education + gender + income + age, marketing)
summary(marketing_lm)

# To use the command 'linearHypothesis' you must load the package 'car'
library(car)
f_test1 <- linearHypothesis(marketing_lm, c("gender = 0", "age = 0"))
f_test1

f_test2 <- linearHypothesis(marketing_lm, c("gender = 0", "age = 0", "price = 0"))
f_test2

# Now test if the effect of increasing price and increasing income is the same:
f_test3 <- linearHypothesis(marketing_lm, c("price = income"))
f_test3

# You can also play around with this and even test other linear restrictions:
f_test4 <- linearHypothesis(marketing_lm, c("2*price = income"))
f_test4

y <- rnorm(100)+2
x1 <- rnorm(100)
x2 <- rnorm(100)

model <- lm(y~ x1 + x2)
summary(model)

model_reduced <- lm(y~1)
summary(model_reduced)

linearHypothesis(model, c("x1=0", "x2=0"))

# Slide 209 ---------------------------------------------------------------

# Load marketing data
marketing <- read.csv("marketing.csv")
head(marketing)

marketing_lm <- lm(rating ~ rq + ju + vo + wa + price, marketing)
summary(marketing_lm)

linearHypothesis(marketing_lm, c("rq = vo"))

marketing_lm2 <- lm(rating ~ kr + ju + vo + wa + price, marketing)
summary(marketing_lm2)


# What happens if all brands are included?
marketing_lm_all <- lm(rating ~ rq + ju + vo + wa + kr + price, marketing)
summary(marketing_lm_all)
# How can this phenomenon be explained? --> Perfect multicollinearity

# Now a model without an intercept
marketing_lm_no_int <- lm(rating ~ 0 + rq + ju + vo + wa + kr + price, marketing)
summary(marketing_lm_no_int)
# Observe the change in the estimation of the coefficients and the different interpretation!




# Slide 215 ---------------------------------------------------------------

# This is the same data set and model as in the code for slide 95
chicken <- read.csv("chicken.csv")
chick_lm <- lm(consum ~ income + pchick + pbeef + ppork, data = chicken)
summary(chick_lm)

# Residuals are obtained throught the residuals() function
resids <- residuals(chick_lm)
resids
plot(resids)

# They can also be calculated manually
# model.matrix extracts design matrix
x <- model.matrix(chick_lm)
x

# Calculte y - x'beta
resids_man <- chicken$consum - x %*% chick_lm$coefficients

all.equal(resids, c(resids_man), check.attributes = FALSE)



# Slide 219 ---------------------------------------------------------------

u <- rnorm(1000)
u <- runif(1000,-1,1)
hist(u, breaks = 10, xlab = "Normalverteilte Fehler", main = "")

qqnorm(u)
qqline(u)


# Consider the same situation as on slide 107
yieldus <- read.csv("yieldus.csv")
# To bring notation in line with slides, create y, x1, x2 and x3
y <- yieldus$Y3
x1 <- yieldus$Y1
x2 <- yieldus$Y60

# Estimate model without spread
yield_lm1 <- lm(y ~ x1 + x2)
summary(yield_lm1)
# All parameters estimated without hiccups!

# Now have a look at the residuals
resids <- residuals(yield_lm1)

# Let's start with a histrogram
dev.off()
hist(resids, breaks = 30, xlab = "Residuals", main = "")
# What does it look like? Does this resemble the shape of the density of a normal distribution?

# Now a QQ-Plot
qqnorm(resids)
qqline(resids, col="red")
# Do the dots lie on the line? At least approximately?


# The Jarque-Bera test
JB <- tseries::jarque.bera.test(resids)
JB
# Can we reject normality?



# Now consider profits

profit <- read.csv("profit.csv")
profit_lm <- lm(GEW94 ~ GEW93 + UM94, profit[1:20, ])
summary(profit_lm)
resids <- residuals(profit_lm)

# Let's start with a histrogram
dev.off()
hist(resids, breaks = 30, xlab = "Residuals", main = "")
# What does it look like? Does this resemble the shape of the density of a normal distribution?

# Now a QQ-Plot
qqnorm(resids)
qqline(resids, col="red")
# Do the dots lie on the line? At least approximately?


# The Jarque-Bera test
JB <- tseries::jarque.bera.test(resids)
JB
# Can we reject normality?




# Slide 228 ---------------------------------------------------------------


profit <- read.csv("profit.csv")
profit_lm <- lm(GEW94 ~ GEW93, profit)
summary(profit_lm)

par(mfrow = c(1, 2))
# Plot GEW94 and 93 against each other
plot(profit$GEW93, profit$GEW94, ylab = "GEW94", xlab = "GEW93")
# Add regression line
abline(profit_lm, lty=2)
legend("topleft", legend=c("regression line"), lty=2)

# Plot residuals against explanatory variable
plot(profit$GEW93, residuals(profit_lm), ylab = "Residuals", xlab = "GEW93")
abline(0,0, lty=2)





# Slide 232 ---------------------------------------------------------------

# Start with basic model
profit <- read.csv("profit.csv")
profit_lm <- lm(GEW94 ~ GEW93, profit)

# summary() also reports R^2 (under "Multiple R-squared")
summary(profit_lm)

# Can also be calculated manually
SSR <- sum(residuals(profit_lm)^2)
SSR
TSS <- sum((profit$GEW94 - mean(profit$GEW94))^2)
TSS

1 - SSR/TSS

# Alternative way to calculate TSS
profit_lm_0 <- lm(GEW94 ~ 1, profit)
summary(profit_lm_0)
sum(residuals(profit_lm_0)^2)



# R^2 always goes down, when we add another predictor
profit_lm2 <- lm(GEW94 ~ GEW93 + UM94, profit)
summary(profit_lm2)

# Even if that predictor is not significant
profit_lm3 <- lm(GEW94 ~ GEW93 + UM94 + MA94, profit)
summary(profit_lm3)


# Slide 233 ---------------------------------------------------------------

chicken <- read.csv("chicken.csv")

# We consider the log linear model.
chick_lm1 <- lm(log(consum) ~ log(pchick), chicken)
chick_lm2 <- lm(log(consum) ~ log(income), chicken)
chick_lm3 <- lm(log(consum) ~ log(income) + log(pchick), chicken)
chick_lm4 <- lm(log(consum) ~ log(income) + log(pchick) + log(ppork), chicken)
chick_lm5 <- lm(log(consum) ~ log(income) + log(pchick) + log(ppork) + log(pbeef), chicken)



Rs <- c(
  summary(chick_lm1)$r.squared,
  summary(chick_lm2)$r.squared,
  summary(chick_lm3)$r.squared,
  summary(chick_lm4)$r.squared,
  summary(chick_lm5)$r.squared
)

SSRs <- c(
  sum(residuals(chick_lm1)^2),
  sum(residuals(chick_lm2)^2),
  sum(residuals(chick_lm3)^2),
  sum(residuals(chick_lm4)^2),
  sum(residuals(chick_lm5)^2)
)

cbind(SSRs, Rs)



# Slide 239 ---------------------------------------------------------------

AICs <- c(
  AIC(chick_lm1),
  AIC(chick_lm2),
  AIC(chick_lm3),
  AIC(chick_lm4),
  AIC(chick_lm5)
)

BICs <- c(
  BIC(chick_lm1),
  BIC(chick_lm2),
  BIC(chick_lm3),
  BIC(chick_lm4),
  BIC(chick_lm5)
)

cbind(SSRs, Rs, AICs, BICs)

# Slide 241 -----------------------------------------------------------

# Now we estimate another linear model

chick_lm6 <- lm(consum ~ income + pchick, chicken)
summary(chick_lm3)
summary(chick_lm6)
cbind(sum(residuals(chick_lm6)^2), summary(chick_lm6)$r.squared, AIC(chick_lm6), BIC(chick_lm6))

# Transform AIC and BIC from log-linear scale to linear scale
AIC(chick_lm3) + 2*sum(log(chicken$consum))
BIC(chick_lm3) + 2*sum(log(chicken$consum))


# Slide 255 -----------------------------------------------------------

# I() tells the lm command to perform this calculation before estimation
chicken_lm_quadratic <- lm(consum ~ income + pchick + I(pchick^2) + ppork + I(ppork^2), chicken)
summary(chicken_lm_quadratic)

AIC(chicken_lm_quadratic)
BIC(chicken_lm_quadratic)

chicken_lm_quadratic$coefficients

-chicken_lm_quadratic$coefficients["pchick"]/(2*chicken_lm_quadratic$coefficients["I(pchick^2)"])
-chicken_lm_quadratic$coefficients["ppork"]/(2*chicken_lm_quadratic$coefficients["I(ppork^2)"])



# Slide 262/263 -----------------------------------------------------------

chick_lm_interaction <- lm(consum ~ income + pchick + ppork + income:ppork, chicken)
summary(chick_lm_interaction)
coefs <- chick_lm_interaction$coefficients

coefs["pchick"] + mean(chicken$income) * coefs["income:ppork"]

# I() tells the lm command to perform this calculation before estimation
# This is equivalent to the following:
chick_lm_interaction2 <- lm(consum ~ pchick + ppork + income + I(income - mean(income)) : I(ppork - mean(ppork)), chicken)
summary(chick_lm_interaction2)
chick_lm_interaction2$coefficients["ppork"]


# Slide 269 ---------------------------------------------------------------

marketing <- read.csv("marketing.csv")
marketing_lm <- lm(rating ~ kr:price, marketing)

summary(marketing_lm)
marketing_lm$coefficients["price"] + marketing_lm$coefficients["kr:price"]
