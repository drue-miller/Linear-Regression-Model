insurance <- read.csv("Desktop/Spring 2020/Regression/insurance.csv")
str(insurance)
summary(insurance)

########################### Convert Categorical Variables into Indicator Variables ######################
insurance$sex.male <- ifelse(insurance$sex == "male", 1, 0)
insurance$smoker.yes <- ifelse(insurance$smoker == "yes", 1, 0)

insurance$region.northwest <- ifelse(insurance$region == "northwest", 1, 0)
insurance$region.northeast <- ifelse(insurance$region == "northeast", 1, 0)
insurance$region.southwest <- ifelse(insurance$region == "southwest", 1, 0)

############################ TEST FOR MULTICOLINEARITY #################

data <- insurance[,c(1,3,4,8,9,10,11,12)]
summary(data)
data <- scale(data, center = TRUE)
data <- as.matrix(data)
center <- t(data) %*% data
determinant <- center / (length(insurance$charges) - 1)
det(determinant)
cor(determinant)
library(corrplot)
corrplot(determinant)
library("PerformanceAnalytics")
my_data <- data
chart.Correlation(data, histogram=TRUE, pch=19)

data <- as.data.frame(data)

full.model <- lm(insurance$charges ~ data$age + data$bmi + data$children + data$sex.male + data$smoker.yes + data$region.northeast + data$region.northwest + data$region.southwest)
summary(full.model)

### Look at VIF of full model ###
## VIF of total was 4.0144 so that indicates that multicollinearity may be a problem, 
## but since it is so close to the cutoff point, we are going to assume there is not a 
## severe multicolinearity problem
vif <- 1/(1- .7509)
vif

# Center is full rank, so 8 real eigenvalues exist
qr(center)$rank

# None of the eigenvalues are near zero, therefore we cannot conclude
# that there is an existance of a multicolinearity problem. 
lambda <- eigen(center)$values
lambda
## Condition number
## None of the condition numbers are close to 1000, therefore we can conclude that there 
## is not a multicolinearity problem in our data. 
conditionnum1 <- max(lambda) / lambda[1]
conditionnum1
conditionnum2 <- max(lambda) / lambda[2]
conditionnum2
conditionnum3 <- max(lambda) / lambda[3]
conditionnum3
conditionnum4 <- max(lambda) / lambda[4]
conditionnum4
conditionnum5 <- max(lambda) / lambda[5]
conditionnum5
conditionnum6 <- max(lambda) / lambda[6]
conditionnum6
conditionnum7 <- max(lambda) / lambda[7]
conditionnum7
conditionnum8 <- max(lambda) / lambda[8]
conditionnum8

library(mctest)
imcdiag(x = data, y = insurance$charges)

################################# No multicolinearity #####################
############# MODEL SELECTION #############################
full.model <- lm(insurance$charges ~ insurance$age + insurance$bmi + insurance$children + insurance$sex.male + insurance$smoker.yes + insurance$region.northeast + insurance$region.northwest + insurance$region.southwest)
summary(full.model)
mse <- mean(residuals(fit.forward)^2)
mse
null.model <- lm(insurance$charges ~ 1, data = insurance)
summary(null.model)

## Forward
fit.forward<-step(null.model,scope=list(lower=null.model, upper=full.model),direction='forward')
summary(fit.forward)

## Backward
fit.backward<-step(full.model,scope=list(lower=null.model, upper=full.model),direction='backward')
summary(fit.backward)

## Step-wise
fit.both<-step(null.model,scope=list(lower=null.model, upper=full.model),direction='both')
summary(fit.both)

head(insurance)
x.values <- insurance[,c(1,3,4,8,9,10,11,12)]
library(leaps)
# R-squared
model.r2 <- regsubsets(x.values, insurance$charges ,nbest = 2, really.big = T)
plot(model.r2, scale = "r2")

# Adjusted R-squared
model.adjr2 <- regsubsets(x.values, insurance$charges ,nbest = 2, really.big = T)
plot(model.adjr2, scale = "adjr2")

# BIC
model.BIC <- regsubsets(x.values, insurance$charges ,nbest = 2, really.big = T)
plot(model.BIC, scale = "bic")

#AIC
n<- dim(insurance)[2]
fit.both2<-step(null.model,scope=list(lower=null.model, upper=full.model),direction='both', criterion = "BIC", k=log(n))
summary(fit.both2)

## Best model from selection ##
model2 <- lm(insurance$charges~ insurance$age + insurance$bmi + insurance$smoker.yes)
summary(model2)



library(caret)
library(psych)
library(boot)
library(tidyverse)
library(qpcR)
x.values <- insurance[,c(1,3,4,8,9,10,11,12)]
# Define training control
set.seed(123)
train.control <- trainControl(method = "repeatedcv", 
                              number = 10, repeats = 3)
# Train the model
model <- train(charges ~ age + bmi + smoker.yes + children + region.southwest + region.northeast + region.northwest, data = insurance, method = "lm",
               trControl = train.control)
# Summarize the results
print(model)
model$finalModel

# PRESS
getpress <- function(ix,y,x){
  if(any(ix)){
    linmod=lm(y~.,data=as.data.frame(x[,ix]))
  } else {
    linmod=lm(y~1)
  }
  sum((linmod$residuals/(1-hatvalues(linmod)))^2)
}
presslm <- function(x,y){
  x=as.data.frame(x)
  np=ncol(x)
  xlist=vector("list",np)
  for(j in 1:np){xlist[[j]]=c(TRUE,FALSE)}
  xall=expand.grid(xlist)
  allpress=apply(xall,1,getpress,y=y,x=x)
  list(which=as.matrix(xall),press=allpress)
}
prmod = presslm(x=x.values, y=insurance$charges)
widx = which.min(prmod$press)
xidx = (1:ncol(x.values))[prmod$which[widx,]]
Xin = as.data.frame(x.values[,xidx])
prmod = lm(insurance$charges ~  ., data=Xin)
summary(prmod)

# CP
library(leaps)
cpmod = leaps(x=x.values, y=insurance$charges, method="Cp")
widx = which.min(cpmod$Cp)
xidx = (1:ncol(x.values))[cpmod$which[widx,]]
Xin = data.frame(x.values[,xidx])
cpmod = lm(insurance$charges ~  ., data=Xin)
summary(cpmod)

###### BEst model from that section ############
print(model)
model$finalModel
summary(prmod)
summary(cpmod)

## Use cpmod


#LASSO
library(glmnet)
# Perform 10-fold cross-validation to select lambda ---------------------------
lambdas_to_try <- 10^seq(-3, 5, length.out = 100)
# Setting alpha = 1 implements lasso regression
lasso_cv <- cv.glmnet(as.matrix(x.values), insurance$charges, alpha = 1, lambda = lambdas_to_try,
                      standardize = TRUE, nfolds = 10)
# Plot cross-validation results
plot(lasso_cv)
# Best cross-validated lambda
lambda_cv <- lasso_cv$lambda.min
# Fit final model, get its sum of squared residuals and multiple R-squared
model_cv <- glmnet(as.matrix(x.values), insurance$charges, alpha = 1, lambda = lambda_cv, standardize = TRUE)
model_cv$beta
model_cv$a0
y_hat_cv <- predict(model_cv, as.matrix(x.values))
ssr_cv <- t(insurance$charges - y_hat_cv) %*% (insurance$charges - y_hat_cv)
ssr_cv
rsq_lasso_cv <- cor(insurance$charges, y_hat_cv)^2
rsq_lasso_cv

## Ridge
library(glmnet)
# Perform 10-fold cross-validation to select lambda ---------------------------
lambdas_to_try <- 10^seq(-3, 5, length.out = 100)
# Setting alpha = 0 implements ridge regression
ridge_cv <- cv.glmnet(as.matrix(x.values), insurance$charges, alpha = 0, lambda = lambdas_to_try,
                      standardize = TRUE, nfolds = 10)
# Plot cross-validation results
plot(ridge_cv)
# Best cross-validated lambda
lambda_cv <- ridge_cv$lambda.min
# Fit final model, get its sum of squared residuals and multiple R-squared
model_cv <- glmnet(as.matrix(x.values), insurance$charges, alpha = 0, lambda = lambda_cv, standardize = TRUE)
summary(model_cv$beta)
y_hat_cv <- predict(model_cv, as.matrix(x.values))
ssr_cv <- t(insurance$charges - y_hat_cv) %*% (insurance$charges - y_hat_cv)
ssr_cv
rsq_ridge_cv <- cor(insurance$charges, y_hat_cv)^2
rsq_ridge_cv



# Use the Box-Tidwell one step procedure to transform the data. 
summary(full.model)

z1 = insurance$age*log(insurance$age)
z2 = insurance$bmi*log(insurance$bmi)


trreg <- lm(insurance$charges ~ insurance$age + insurance$bmi + insurance$children + insurance$sex.male + insurance$smoker.yes + insurance$region.northwest + insurance$region.northeast + insurance$region.southwest + z1 + z2)
summary(trreg)

alpha1 = (585.01)/(256.86) +1
alpha2 = (-626.84)/(339.19) +1
 

x1tr = insurance$age^alpha1
x2tr = insurance$bmi^alpha2


trreg1 <- lm(insurance$charges~x1tr + x2tr + insurance$children + insurance$sex.male + insurance$smoker.yes + insurance$region.northwest + insurance$region.northeast + insurance$region.southwest)
summary(trreg1)


## BoxCox
library(MASS)
Box <- boxcox(insurance$charges ~ insurance$age + insurance$bmi + insurance$children + insurance$sex.male + insurance$smoker.yes + insurance$region.northwest + insurance$region.northeast + insurance$region.southwest,lambda = seq(0,2,0.01))

lam <- Box$x[which.max(Box$y)]
lam

model3 <- lm(insurance$charges^lam ~ insurance$age + insurance$bmi + insurance$children + insurance$sex.male + insurance$smoker.yes + insurance$region.northwest + insurance$region.northeast + insurance$region.southwest, data = insurance)
summary(model3)

range(insurance$charges^lam)

## Comparing my best models
# full model
summary(full.model)
# best model from forward, backward, step-wise
summary(fit.both)
# best model from r2, adj r2, bic, aic
summary(model2)
# best model from CV, PRESS, CP
summary(cpmod)
# best model LASSO, Ridge
summary(model_cv$beta)
rsq_ridge_cv
# box-tidwell
summary(trreg1)
plot(trreg1$fitted.values, trreg1$residuals)
# box-cox
summary(model3)

## Comparing models
# no difference between full model and fit.both
anova(full.model, fit.both)
# model improved by using model2
anova(full.model, model2)
# No difference between full model and cpmod
anova(full.model,cpmod)
# these are the same
anova(full.model, trreg1)
# doesn't run
anova(full.model, model3)

## Checking residuals on model3
library(car)
library(olsrr)
library(snpar)
library(lawstat)
library(stats)
## Residuals are not normally distributed
shapiro.test(model3$residuals)
ols_plot_resid_qq(model3)
ols_test_normality(model3)
durbinWatsonTest(model3)## Residuals do not have constant variance
# non-constant error variance test
ncvTest(model3)
# plot studentized residuals vs. fitted values
spreadLevelPlot(model3)
## Residuals are independent
runs.test(model3$residuals)
plot(model3$fitted.values, model3$residuals)
abline(h=0)
x <- as.matrix(insurance[, c(1,3,4,8,9, 10, 11, 12)])
hat <- x %*% solve(t(x)%*%x) %*% t(x)
plot(diag$`diag(hat)`)
diag <- as.data.frame(diag(hat))
## data points 4, 10, 35, 63, 93, 103, 116, 141, 144, 162, 220, 224, 243, 246, 290, 292, 306, 307, 
## 355, 356, 388, 398, 430, 431, 469, 504, 517, 521, 526, 527, 534, 540, 555, 584, 600, 619, 638, 
## 659, 689, 697, 755, 804, 807, 820, 859, 877, 926, 937, 958, 960, 981, 984, 988, 1004, 1009, 
## 1020, 1028, 1040, 1058, 1081, 1105, 1124, 1129, 1135, 1140, 1143, 1157, 1158, 1163, 1190, 1196,
## 1207, 1212, 1216, 1292, 1301, 1316, 1318, 1329, 1332
rstandard <- as.data.frame(rstandard(model3))
##
rstudent <- as.data.frame(rstudent(model3))

##
dffits(model3)[1000:1338]



dffits <- as.data.frame(dffits(model3))
dfbetas <- as.data.frame(dfbetas(model3))
cooks <- as.data.frame(cooks.distance(model3))

mix <- cbind(rstandard, rstudent, diag)

range(insurance$charges)
ols_plot_resid_stud_fit(model3)
ols_plot_resid_fit(model3)
ols_plot_resid_qq(model3)
ols_test_normality(model3)
ols_coll_diag(model3)
ols_plot_resid_lev(model3)
ols_plot_cooksd_bar(model3)
ols_plot_cooksd_chart(model3)
ols_plot_resid_pot(model3)
durbinWatsonTest(model3)

hist(insurance$charges)
glm1 <- glm(charges ~ insurance$age + insurance$bmi + insurance$sex.male + insurance$smoker.yes + insurance$region.northwest + insurance$region.northeast + insurance$region.southwest, data = insurance, family = Gamma(link = "inverse"))
summary(glm1)

glm2 <- glm(charges ~ insurance$sex.male + insurance$smoker.yes + insurance$region.northwest + insurance$region.northeast + insurance$region.southwest, data = insurance, family = Gamma(link = "inverse"))
summary(glm2)

glm3 <- glm(charges ~ insurance$age + insurance$bmi , data = insurance, family = Gamma(link = "inverse"))
summary(glm3)
insurance$bmi_squared <- (insurance$bmi)^2
glm4 <- lm(log(charges) ~ insurance$age + insurance$bmi + insurance$bmi_squared + insurance$sex.male + insurance$smoker.yes + insurance$region.northwest + insurance$region.northeast + insurance$region.southwest, data = insurance)
summary(glm4)
AIC(glm4)
AIC(model3)
plot(glm4$fitted.values, glm4$residuals)
hist(log(insurance$charges))
hist(insurance$bmi)
hist(insurance$age)


# box-tidwell
summary(trreg1)
plot(trreg1$fitted.values, trreg1$residuals)

childrenmodel <- lm(log(charges) ~ children + age + bmi + smoker.yes + sex.male, data = insurance)
summary(childrenmodel)
plot(childrenmodel$fitted.values, childrenmodel$residuals)
hist(insurance$smoker.yes)
plot(log(insurance$charges), insurance$children)
str(insurance$children)
insurance$children <- as.numeric(insurance$children)
insurance$children.yes <- ifelse(insurance$children > 0, 1, 0)

plot(insurance)
