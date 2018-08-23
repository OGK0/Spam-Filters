# Lab 2 Spam Filter Code

# Change the working directory

# Source the necessary files
spam <- read.table("Spam.txt", header = F)

Spam <- test.set(spam, .33)
response.train <- as.factor(Spam$train[,58])
response.test <- as.factor(Spam$test[,58])

source("SPM_Panel.R")
source("FactorPlots.R")
source("PCAplots.R")
source("ROC.R")
source("Testset.R")
source("pc.glm.R")
source('princompreg.R')

summary(spam)

barplot(table(spam[,58]), xlab = "Mail Classfication", main = "Spam in Data Set", names.arg = c("Ham", "Spam"), col = "navy blue")

# Scatter Plot Matrices
pairs(spam[,c(1:10, 58)], upper.panel = panel.cor, diag.panel = panel.hist, panel = panel.smooth)
pairs(spam[,c(11:20, 58)], upper.panel = panel.cor, diag.panel = panel.hist, panel = panel.smooth)
pairs(spam[,c(21:30, 58)], upper.panel = panel.cor, diag.panel = panel.hist, panel = panel.smooth)
pairs(spam[,c(31:40, 58)], upper.panel = panel.cor, diag.panel = panel.hist, panel = panel.smooth)
pairs(spam[,c(41:57, 58)], upper.panel = panel.cor, diag.panel = panel.hist, panel = panel.smooth)


# Principal Components - 98% Confidence
#Principal Components

spam.pc = princomp(spam[,1:57], cor = T)
biplot(spam.pc)

# Biplot with the spam and ham classifications color coded.
biplot.fact(spam.pc, spam[,58])
legend(20, 15, legend = c("spam", "Ham"), pch = c(18, 19), col = c("red", "blue"))
summary(spam.pc)

screeplot(spam.pc)

# loading plot & values
loadingsplot(spam.pc, k=3)
loadings(spam.pc)

pc.train <- princomp(Spam$train[,-58], cor = T)
cumplot(pc.train)
var.comp(pc.train, 98)
spampca1 <- pc.glm(pc.train, 98, response.train)
summary(spampca1)

# Model utility test
pc.n <- pc.null(pc.train, 98, response.train)
anova(pc.n, spampca1, test = "Chi")

# predicted or fitted values for the pc model
spampca1.pred <- predict.pc.glm(spampca1, pc.train, ndata = Spam$test[,-58])


# GLM Model with all 58 Variables

spam.glm <- glm(response~., data = data.frame(spam[,-58], response = as.factor(spam[,58])), family = binomial)
summary(spam.glm, cor = F)
AIC(spam.glm)
plot( jitter(spam.glm$fitted.values),spam[,58], xlab = "Fitted", ylab = "Actual")

# Stepwise GLM Model
spam.step <- step(spam.glm, data = data.frame(spam[,-58], response = as.factor(spam[,58])), family = binomial, trace=0)
summary(spam.step)

# Model Utility Comparisons (Chi-Square Test)
spam.null <- glm(response~1, data = data.frame(spam[,-58], response = as.factor(spam[,58])), family = binomial)
anova(spam.null, spam.glm, test = "Chi")
anova(spam.null, spam.step,test = "Chi")
anova(spam.step, spam.glm, test = "Chi")

# Test Set Method for Evaluation
response.train <- as.factor(Spam$train[,58])
response.test <- as.factor(Spam$test[,58])

# All variable model
glm1.train <- glm(response.train ~., data = Spam$train[,-58], family = binomial)
null.train <- glm(response.train ~1, data = Spam$train[,-58], family = binomial)
glm1.pred <- predict(glm1.train, type = "response", newdata = Spam$test)

# A Stepwise Model
glm2.train  <- step(glm1.train, trace = 0)

# Partial Chi2 test	
anova(glm2.train, glm1.train, test = "Chi")

# Test set predictions for the stepwise model
glm2.pred <- predict(glm2.train, type = "response", newdata = Spam$test)

# ROC Curves & Score Table
plot.roc(glm1.pred, Spam$test[,58], main = "ROC Curve - SPAM Filter")
lines.roc(glm2.pred, Spam$test[,58], col = "orange")
legend(.6, .4, legend = c("All Var Model", "Step Model"), col = c("blue", "orange"), lwd = 1)
score.table(glm1.pred, Spam$test[,58], .5)
score.table(glm2.pred, Spam$test[,58], .5)

# Log-Model

Lspam.train <- log(Spam$train[,-58] + .01)
Lspam.test <- log(Spam$test[,-58] + .01)
Lspam1.train <- glm(response.train~., data = Lspam.train, family = binomial)
Lspam.null <- glm(response.train~1, data = Lspam.train, family = binomial)
Lspam1.pred <- predict(Lspam1.train, type = "response", newdata = Lspam.test)  

Lspam2.train <- step(Lspam1.train, trace = 0)
Lspam2.pred <- predict(Lspam2.train, type = "response", newdata = Lspam.test)

anova(Lspam.null, Lspam1.train, test = "Chi")
anova(Lspam2.train, Lspam1.train, test = "Chi")


# ROC's for Comparing Stepwise Models
plot.new()
plot.roc(glm2.pred, Spam$test[,58], main = "ROC Curve - SPAM Filter for Log Stepwise Models")
lines.roc(Lspam2.pred, Spam$test[,58], col = "red")
legend(.6, .4, legend = c("Stepwise T&T Model", "Log Step T&T Model"), col = c("blue", "red"), lwd = 1)


# TIME SERIES ANALYSIS CODE

library(forecast)

ham <- read.table('ham_ts-1.csv',header=T,sep=',')
summary(ham)

spam <-read.table('spam_ts.csv',header=T,sep=',')
summary(spam)

ham.ts  <- ts(ham$count)
spam.ts <- ts(spam$count)


# HAM PART 1

plot(ham.ts, main = "Ham Time Series", ylab = "Amount of Ham")
acf(ham.ts, main = "ACF for Ham") #stationary?

#Modeling of trend

time <- c(1:length(ham.ts))
ham.trend<-lm(ham.ts~time)
summary(ham.trend)

#Plot the trend with ts:
plot(ham.ts, main = "Ham Trend", ylab = "Amount of Ham")
abline(ham.trend, col='blue')

#What about the period?
pgham <- spec.pgram(ham.ts,spans=9,demean=T,log='no', main = "Ham Periodogram")
max.omega.ham <- pgham$freq[which(pgham$spec==max(pgham$spec))]
1/max.omega.ham

#modeling of the seasonality
season <- rep(NA, NROW(ham.ts))

season[which(time %% 7 == "1")] <- "1Thursday"
season[which(time %% 7 == "2")] <- "2Friday"
season[which(time %% 7 == "3")] <- "3Saturday"
season[which(time %% 7 == "4")] <- "4Sunday"
season[which(time %% 7 == "5")] <- "5Monday"
season[which(time %% 7 == "6")] <- "6Tuesday"
season[which(time %% 7 == "1")] <- "0Wednesday"

season <- as.factor(season)
 
#think: what's the meanings of '1,2,3,4,5,6,0' here?
#It's better to change the level values into 'Th(1), F(2), Sa(3) , S(4) , M(5), T(6), W(0)'.

ham.season <- lm(ham.ts ~ time + season) #here both trend and seasonality are modeled together
summary(ham.season) #Wednesday is the base case

#plot the trend and seasonality with ts:
plot(ham.ts, main = "Ham and Trend and Seasonality", ylab = "Amount of Ham")
abline(ham.trend,col='red')
lines(predict(ham.season),col='blue')

#get the residuals from the above model:
ham.e.ts <- ts(ham.season$residuals)
plot(ham.e.ts, main = "Ham Residuals", ylab = "Residuals")

#ACF 
acf(ham.e.ts)
acf(diff(ham.e.ts))


#PACF
pacf(ham.e.ts)
pacf(diff(ham.e.ts))

#the first order difference:

par(mfrow = c(3,1))
plot(ham.e.ts, main = "Ham Residuals")
acf(ham.e.ts)
pacf(ham.e.ts)
par(mfrow = c(1,1))

par(mfrow = c(3,1))
plot(diff(ham.e.ts), main = "Ham Residuals - First Difference")
acf(diff(ham.e.ts))
pacf(diff(ham.e.ts))
par(mfrow = c(1,1))

# ARIMA Models
ham.ar1 <- arima(diff(ham.e.ts),order=c(1,0,0))
ham.ar1
tsdiag(ham.ar1,gof.lag=20)

# Auto-ARIMA Model
ham.auto <- auto.arima(diff(ham.e.ts))
ham.auto
tsdiag(ham.auto,gof.lag=20)


ham.arma.32 <- arima(diff(ham.e.ts),order=c(3,0,2))
ham.arma.32
tsdiag(ham.arma.32,gof.lag=20)

#Forecasting:
ham.auto.forecast <- forecast(ham.auto, h = 100)
plot(ham.auto.forecast, ylab = "Amount of Ham", xlab = "Time")



# SPAM PART 2
plot(spam.ts, main = "Spam Time Series", ylab = "Amount of Spam")

#Modeling of trend

time2 <- c(1:length(spam.ts))
spam.trend <- lm(spam.ts~time2)
summary(spam.trend)

#Plot the trend with ts:
plot(spam.ts, main = "Spam Trend", ylab = "Amount of Spam")
abline(spam.trend, col = 'red')

#What about the period?
pgspam <- spec.pgram(spam.ts, spans=9,demean=T, log='no', main = "Spam Periodogram")
max.omega.spam <- pgspam$freq[which(pgspam$spec==max(pgspam$spec))] #where is the peak?
1/max.omega.spam

# Plots
par(mfrow = c(3,1))
plot(spam.ts, main = "Spam Time Series - Plot, ACF, PACF")
acf(spam.ts)
pacf(spam.ts)
par(mfrow = c(1,1))

#First Order Difference Plots
par(mfrow = c(3,1))
plot(diff(spam.ts), main = "Spam First Difference - Plot, ACF, PACF")
acf(diff(spam.ts))
pacf(diff(spam.ts))
par(mfrow = c(1,1))

# ARIMA Models
spam.ar1 <- arima(spam.ts,order=c(1,0,0))
spam.ar1
tsdiag(spam.ar1,gof.lag=20)

# Auto-ARIMA Model
spam.auto <- auto.arima(diff(spam.ts))
spam.auto
tsdiag(spam.auto, gof.lag=20)

spam.arma.32 <- arima(diff(spam.ts),order=c(3,0,2))
spam.arma.32
tsdiag(spam.arma.32,gof.lag=20)

#Forecasting:
spam.auto.forecast <- forecast(spam.auto, h = 100)
plot(spam.auto.forecast, ylab = "Amount of Spam", xlab = "Time")
