# Cases data Prediction

# Section 0: Global settings and import packages 
getwd()
setwd("D:/RStudio/Time_Series_Project/Part1/")
options(rlib_downstream_check = FALSE)
options(digits=7)

library(tseries)
library(fGarch)
library(forecast)
library(lifecycle)
library(xts)
library(ggplot2)
library(MTS)
#Section 0 define evaluation function
rmse_scaled=function(Y_true,Y_pre)
{
  rmse=sqrt(mean((Y_true-Y_pre)^2))
  scale=sqrt(mean(Y_true^2))
  return(rmse/scale)
}
MAPE=function(Y_true,Y_pre)
{
  mae = mean(abs((Y_true-Y_pre)/Y_true))
  mape = mae*100
  return(mape)
}

# Section 1: Data pre-processing

data=read.table('dat_ca.csv', sep=",", header=TRUE)
data

dast=read.table('dat_st.csv', sep=",", header=TRUE)
dast
??mape
#According to the paper provided by professor, we build a model to fit the
#log-growth rate of (case+1) series.

SG = ts(data$SG,start = c(2020,1),end=c(2022,40),frequency = 48)
lsg = log(SG+1)
diffsg = diff(lsg)

# The train set is from the second week of 2020 to the 20th week of 2022.
# The validation set is from the 21th week to the 32th week of 2022.

sgtrain = window(diffsg,start = c(2020,2),end=c(2022,20),frequency = 48)
sgvalid = window(diffsg,start = c(2022,21),end=c(2022,32),frequency = 48)
sgforpre = window(diffsg,start = c(2020,2),end=c(2022,32),frequency = 48)

UK<-ts(data$UK,start = c(2020,1),end=c(2022,40),frequency = 48)
luk<-log(UK+1)
diffuk = diff(luk)
uktrain = window(diffuk,start = c(2020,2),end=c(2022,20),frequency = 48)
ukvalid=window(diffuk,start = c(2022,21),end=c(2022,32),frequency = 48)
ukforpre = window(diffuk,start = c(2020,2),end=c(2022,32),frequency = 48)

US<-ts(data$US,start = c(2020,1),end=c(2022,40),frequency = 48)
lus<-log(US+1)
diffus = diff(lus)
ustrain = window(diffus,start = c(2020,2),end=c(2022,20),frequency = 48)
usvalid=window(diffus,start = c(2022,21),end=c(2022,32),frequency = 48)
usforpre = window(diffus,start = c(2020,2),end=c(2022,32),frequency = 48)
#Stock data

SG_stock = ts(dast$SG,start = c(2020,1),end=c(2022,40),frequency = 48)
d_SG_stock = diff(log(SG_stock))

sg_ret_train = window(d_SG_stock,start = c(2020,2),end=c(2022,20),frequency = 48)
sg_ret_valid = window(d_SG_stock,start = c(2022,21),end=c(2022,32),frequency = 48)
sg_ret_forpre = window(d_SG_stock,start = c(2020,2),end=c(2022,32),frequency = 48)
sg_ret_predict = window(d_SG_stock,start = c(2022,33),end=c(2022,40),frequency = 48)
US_stock = ts(dast$US,start = c(2020,1),end=c(2022,40),frequency = 48)
d_US_stock = diff(log(US_stock))

us_ret_train = window(d_US_stock,start = c(2020,2),end=c(2022,20),frequency = 48)
us_ret_valid = window(d_US_stock,start = c(2022,21),end=c(2022,32),frequency = 48)
us_ret_forpre = window(d_US_stock,start = c(2020,2),end=c(2022,32),frequency = 48)

UK_stock = ts(dast$UK,start = c(2020,1),end=c(2022,40),frequency = 48)
d_UK_stock = diff(log(UK_stock))

uk_ret_train = window(d_UK_stock,start = c(2020,2),end=c(2022,20),frequency = 48)
uk_ret_valid = window(d_UK_stock,start = c(2022,21),end=c(2022,32),frequency = 48)
uk_ret_forpre = window(d_UK_stock,start = c(2020,2),end=c(2022,32),frequency = 48)

lgrtrain=data.frame(SG_case=sgtrain,UK_case=uktrain,US_case=ustrain,SG_ret=sg_ret_train,UK_ret=uk_ret_train,US_ret=us_ret_train)
lgrvalid=data.frame(SG_case=sgvalid,UK_case=ukvalid,US_case=usvalid,SG_ret=sg_ret_valid,UK_ret=uk_ret_valid,US_ret=us_ret_valid)
lgr_forpre=data.frame(SG_case=sgforpre,UK_case=ukforpre,US_case=usforpre,SG_ret=sg_ret_forpre,UK_ret=uk_ret_forpre,US_ret=us_ret_forpre)

# Section 2: Basic diagnosis and description
# Plot the growth rate of weekly cases.

par(mfrow=c(3,1))
plot(sgtrain)
plot(uktrain)
plot(ustrain)

par(mfrow=c(3,1))
acf(sgtrain^2)
# Data of Singapore seems to have no need for GARCH
acf(ustrain^2)
# Data of United States seems to have a need for GARCH
acf(uktrain^2)
# Data of United Kingdom seems to have a need for GARCH

cor(uktrain,ustrain)
cor(sgtrain,ustrain)
cor(uktrain,sgtrain)
# Result 1 : The series of US and UK have a strong correlation,
# whose parameter is 0.627.

adf.test(sgtrain);kpss.test(sgtrain);
adf.test(uktrain);kpss.test(uktrain);
adf.test(ustrain);kpss.test(ustrain);

# Result 2: All the series of three countries are stationary according to adf-test and pp-test.

par(mfrow=c(1,1))
hist(sgtrain,xlab='Ratio of weekly increae',ylab='Frequency',main='Histogram of the speed in SG',breaks=100)
hist(ustrain,xlab='Ratio of weekly increae',ylab='Frequency',main='Histogram of the speed in US',breaks=100)
hist(uktrain,xlab='Ratio of weekly increae',ylab='Frequency',main='Histogram of the speed in UK',breaks=100)

# Result 3: The growth rate of cases are right-skewed

# Section 3: ARIMA model : Determine order
auto.arima(sgtrain,ic="bic")
auto.arima(ustrain,ic="bic")
auto.arima(uktrain,ic="bic")

# Because of 'Result 2', the series are stationary, so
# we consider limit the times of difference d=0.

auto.arima(sgtrain,ic="bic",d=0)
auto.arima(ustrain,ic="bic",d=0)
auto.arima(uktrain,ic="bic",d=0)
# All three series conform AR(1) process.

# Section 4: SG modelling

fit1 = arima(sgtrain,order=c(1,0,0))
res1 = residuals(fit1)

Box.test(res1, lag = 10, type="Ljung")
Box.test(res1^2, lag = 10, type="Ljung")
# The residuals and squared residuals are both white noise for a AR(1) process,
# which indicates that the model is good enough.

# Section 5: US modelling
fit2 = arima(ustrain,order=c(1,0,0))
res2 = residuals(fit2)
Box.test(res2, lag = 10, type="Ljung")
Box.test(res2^2, lag = 10, type="Ljung")
# The residual is white noise but the squared residuals imply further use of GARCH.

garch.model.us.1 = garchFit(formula= ~arma(1,0) + garch(1,1),data= ustrain, cond.dist="std")
summary(garch.model.us.1)
garch.model.us.2 = garchFit(formula= ~arma(1,0) + garch(1,0),data= ustrain, cond.dist="std")
summary(garch.model.us.2)
garch.model.us.3 = garchFit(formula= ~arma(1,0) + garch(1,2),data= ustrain, cond.dist="std")
summary(garch.model.us.3)
garch.model.us.4 = garchFit(formula= ~arma(1,0) + garch(2,1),data= ustrain, cond.dist="std")
summary(garch.model.us.4)
# By using AIC criteria, we choose ARMA(1,0)+GARCH(1,1)model to fit the US series.

# Section 6: UK modelling
fit3 = Arima(uktrain,order=c(1,0,0))
res3 = residuals(fit3)
Box.test(res3, lag = 10, type="Ljung")
Box.test(res3^2, lag = 10, type="Ljung")

# The residual is white noise but the squared residuals imply further use of GARCH.

garch.model.uk.1 = garchFit(formula= ~arma(1,0) + garch(1,1),data= uktrain, cond.dist="std")
summary(garch.model.uk.1)

garch.model.uk.2 = garchFit(formula= ~arma(1,0) + garch(1,0),data= uktrain, cond.dist="std")
summary(garch.model.uk.2)

help(garchFit)
garch.model.uk.2@fitted
# By using AIC criteria, we choose ARMA(1,0)+ARCH(1)model to fit the UK series.

# Conclusion
# SG:AR(1); US:AR(1)+GARCH(1,1);  UK:AR(1)+ARCH(1)

# Section 7: ARIMA+GARCH Prediction

# Singapore
forecast_sg = predict(fit1,12)
plot(diffsg,xlim=c(2020,2023),ylim=c(-1.5,2.5))
lines(seq(from=2022.41667,by=1/48,length=12), forecast_sg$pred,col="red")
lines(seq(from=2022.41667,by=1/48,length=12), forecast_sg$pred + 1.96*forecast_sg$se,
      col="blue")
lines(seq(from=2022.41667,by=1/48,length=12), forecast_sg$pred - 1.96*forecast_sg$se,
      col="blue")
legend(x = 'topright', 
       legend = c(expression('Predict numbers'),expression('Confidential Interval')),
       lty = 1,
       col = c('red','blue'),
       bty = 'n',
       horiz = T)

# US
forecast_us = predict(garch.model.us.1,n.ahead=12)
forecast_us$interval_1=forecast_us$meanForecast+qt(0.025,3.313066)*forecast_us$standardDeviation
forecast_us$interval_2=forecast_us$meanForecast+qt(0.975,3.313066)*forecast_us$standardDeviation
forecast_us

plot(diffus,xlim=c(2020,2023),ylim=c(-1,4))
lines(seq(from=2022.41667,by=1/48,length=12), forecast_us$meanForecast,col="red")
lines(seq(from=2022.41667,by=1/48,length=12), forecast_us$interval_1, col="blue")
lines(seq(from=2022.41667,by=1/48,length=12), forecast_us$interval_2, col="blue")
legend(x = 'topright', 
       legend = c(expression('Predict numbers'),expression('Confidential Interval')),
       lty = 1,
       col = c('red','blue'),
       bty = 'n',
       horiz = T)
#UK 
forecast_uk = predict(garch.model.uk.2,n.ahead=12)
forecast_uk$interval_1=forecast_uk$meanForecast+qt(0.025,4.44841)*forecast_uk$standardDeviation
forecast_uk$interval_2=forecast_uk$meanForecast+qt(0.975,4.44841)*forecast_uk$standardDeviation
forecast_uk

plot(diffuk,xlim=c(2020,2023),ylim=c(-2,3))
lines(seq(from=2022.41667,by=1/48,length=12), forecast_uk$meanForecast,col="red")
lines(seq(from=2022.41667,by=1/48,length=12), forecast_uk$interval_1, col="blue")
lines(seq(from=2022.41667,by=1/48,length=12), forecast_uk$interval_2, col="blue")
legend(x = 'topright', 
       legend = c(expression('Predict numbers'),expression('Confidential Interval')),
       lty = 1,
       col = c('red','blue'),
       bty = 'n',
       horiz = T)
#Section 8: Obtain predictions on case series from growth rate series

#SG
case_SG = as.data.frame(SG[1:128])
colnames(case_SG)[1] = 'true_value'

growth_SG <- c(as.vector(sgtrain),as.vector(forecast_sg$pre))
initial_SG <- as.numeric(case_SG[1,1])
case_pre_SG <- round(exp(log(initial_SG+1)+cumsum(growth_SG)))-1
case_SG$pre <- c(initial_SG,case_pre_SG)

# Prediction on verification set
index = 21:32
plot(case_SG$true_value[117:128]~index, 
     main = 'SG New Cases Prediction from 2022-06 to 2022-08',
     type = 'o',
     pch = 16,
     col = 'black',
     xlab = "Time index",
     ylab = "Number of new cases",
     ylim = c(10000,80000))
lines(case_SG$pre[117:128]~index, type = 'o', pch = 16, col = 'red')
legend(x = 'topright', 
       legend = c(expression('True numbers'), expression('Predict numbers')),
       lty = 1,
       pch = 16,
       lwd = 2,
       col = c('black', 'red'),
       bty = 'n',
       horiz = T)

SG_AA_RMSE=rmse_scaled(SG[117:128],case_SG$pre[117:128])
SG_AA_MAPE=MAPE(SG[117:128],case_SG$pre[117:128])
SG_AA_RMSE
SG_AA_MAPE
??mape
#US
case_US = as.data.frame(US[1:128])
colnames(case_US)[1] = 'true_value'

growth_US <- c(as.vector(ustrain),as.vector(forecast_us$meanForecast))
initial_US <- as.numeric(case_US[1,1])
case_pre_US <- round(exp(log(initial_US+1)+cumsum(growth_US)))-1
case_US$pre <- c(initial_US,case_pre_US)

# Prediction on verification set
index = 21:32
plot(as.vector(case_US$true_value)[117:128]~index, 
     main = 'US New Cases Prediction from 2022-06 to 2022-08',
     type = 'o',
     pch = 16,
     col = 'black',
     xlab = "Time index",
     ylab = "Number of new cases")
lines(case_US$pre[117:128]~index, type = 'o', pch = 16, col = 'red')
legend(x = 'topright', 
       legend = c(expression('True numbers'), expression('Predict numbers')),
       lty = 1,
       pch = 16,
       lwd = 2,
       col = c('black', 'red'),
       bty = 'n',
       horiz = T)

US_AA_RMSE=rmse_scaled(US[117:128],case_US$pre[117:128])
US_AA_RMSE
US_AA_MAPE=MAPE(US[117:128],case_US$pre[117:128])
US_AA_MAPE

# UK
case_UK = as.data.frame(UK[1:128])
colnames(case_UK)[1] = 'true_value'

growth_UK <- c(as.vector(uktrain),as.vector(forecast_uk$meanForecast))
initial_UK <- as.numeric(case_UK[1,1])
case_pre_UK <- round(exp(log(initial_UK+1)+cumsum(growth_UK)))-1
case_UK$pre <- c(initial_UK,case_pre_UK)

# Prediction on verification set
index = 21:32
plot(as.vector(case_UK$true_value)[117:128]~index, 
     main = 'UK New Cases Prediction from 2022-06 to 2022-08',
     type = 'o',
     pch = 16,
     col = 'black',
     xlab = "Time index",
     ylab = "Number of new cases")
lines(case_UK$pre[117:128]~index, type = 'o', pch = 16, col = 'red')

legend(x = 'topright', 
       legend = c(expression('True numbers'), expression('Predict numbers')),
       lty = 1,
       pch = 16,
       lwd = 2,
       col = c('black', 'red'),
       bty = 'n',
       horiz = T)

UK_AA_RMSE=rmse_scaled(UK[117:128],case_UK$pre[117:128])
UK_AA_RMSE
UK_AA_MAPE=MAPE(UK[117:128],case_UK$pre[117:128])
UK_AA_MAPE

#Section 8.2: Calculate MAPE for Training set
# We calculate the MAPE from the fifth data point, which is

sg_train_fit = fitted(fit1)
us_train_fit = garch.model.us.1@fitted
uk_train_fit = garch.model.uk.2@fitted

sg_train_fit = ts(sg_train_fit,start = c(2020,2),end=c(2022,20),frequency = 48)
us_train_fit = ts(us_train_fit,start = c(2020,2),end=c(2022,20),frequency = 48)
uk_train_fit = ts(uk_train_fit,start = c(2020,2),end=c(2022,20),frequency = 48)


plot(sg_train_fit,type='l')
plot(sgtrain)
plot(us_train_fit,type='l')
plot(ustrain)
plot(uk_train_fit,type='l')
plot(uktrain)
sg_train_fit_case=round(exp(cumsum(sg_train_fit)))-1
us_train_fit_case=round(exp(cumsum(us_train_fit)))-1
uk_train_fit_case=round(exp(cumsum(uk_train_fit)))-1


SG_train_MAPE=MAPE(SG[5:116],sg_train_fit_case[4:115])
US_train_MAPE=MAPE(US[5:116],us_train_fit_case[4:115])
UK_train_MAPE=MAPE(UK[5:116],uk_train_fit_case[4:115])

SG_train_MAPE
US_train_MAPE
UK_train_MAPE

#Section 9: VAR for six series
cor(uk_ret_train,us_ret_train)
cor(sg_ret_train,us_ret_train)
cor(uk_ret_train,sg_ret_train)
lgrtrain
lgrvalid
dim(lgrtrain)
MTSplot(lgrtrain)
ccm(lgrtrain)
m0=VARorder(lgrtrain)
m0$Mstat
m1=VAR(lgrtrain,2)
m2=refVAR(m1,thres=1.96)
m1_resi= m1$residuals
m1_fitted = m1$data[3:115,]-m1_resi
MTSplot(m1_resi)
acf(m1_resi)
MTSdiag(m2,adj=12)


a=VARpred(m1,12)
c=VARpred(m2,12)
b=a$pred
b=data.frame(b)
d=c$pred
d=data.frame(d)
d

#Section 10£ºVAR in each countries
lgrtrain=data.frame(SG_case=sgtrain,UK_case=uktrain,US_case=ustrain,SG_ret=sg_ret_train,UK_ret=uk_ret_train,US_ret=us_ret_train)
lgrvalid=data.frame(SG_case=sgvalid,UK_case=ukvalid,US_case=usvalid,SG_ret=sg_ret_valid,UK_ret=uk_ret_valid,US_ret=us_ret_valid)


#Singapore
sg_data <- data.frame(
  sgca=sgtrain, 
  sgst=sg_ret_train,
  stringsAsFactors=FALSE)

MTSplot(sg_data)
m0_sg=VARorder(sg_data)
#VAR(2)or VAR(4)
m1_sg=VAR(sg_data,2) 
m2_sg=refVAR(m1_sg,thres=1.96)
a_sg=VARpred(m1_sg,12)
c_sg=VARpred(m2_sg,12)
b_sg=a_sg$pred
b_sg=data.frame(b_sg)
d_sg=c_sg$pred
d_sg=data.frame(d_sg)

m3_sg=VAR(sg_data,4) 
m4_sg=refVAR(m3_sg,thres=1.96)
a2_sg=VARpred(m3_sg,12)
c2_sg=VARpred(m4_sg,12)
b2_sg=a2_sg$pred
b2_sg=data.frame(b2_sg)
d2_sg=c2_sg$pred
d2_sg=data.frame(d2_sg)

#UK
uk_data <- data.frame(
  ukca=uktrain, 
  ukst=uk_ret_train,
  stringsAsFactors=FALSE)

MTSplot(uk_data)
m0_uk=VARorder(uk_data)
#VAR(2)
m1_uk=VAR(uk_data,2) 
m2_uk=refVAR(m1_uk,thres=1.96)
a_uk=VARpred(m1_uk,12)
c_uk=VARpred(m2_uk,12)
b_uk=a_uk$pred
b_uk=data.frame(b_uk)
d_uk=c_uk$pred
d_uk=data.frame(d_uk)
d_uk$ukst

#US
us_data <- data.frame(
  usca=ustrain, 
  usst=us_ret_train,
  stringsAsFactors=FALSE)

MTSplot(us_data)
m0_us=VARorder(us_data)
#VAR(1)
m1_us=VAR(us_data,1) 
m2_us=refVAR(m1_us,thres=1.96)
a_us=VARpred(m1_us,12)
c_us=VARpred(m2_us,12)
b_us=a_us$pred
b_us=data.frame(b_us)
d_us=c_us$pred
d_us=data.frame(d_us)
b_us
d_us
#Section 11 Three stock returns VAR
ret_data <- data.frame(
  sg=sg_ret_train,
  us=us_ret_train, 
  uk=uk_ret_train,
  stringsAsFactors=FALSE)
m0_ret=VARorder(ret_data)
m1_ret=VAR(ret_data,2) 
m2_ret=refVAR(m1_ret,thres=1.96)
a_ret=VARpred(m1_ret,12)
c_ret=VARpred(m2_ret,12)
b_ret=a_ret$pred
b_ret=data.frame(b_ret)
d_ret=c_ret$pred
d_ret=data.frame(d_ret)
b_ret
d_ret

#Section11: MAPE of VAR model

m2_fitted = m2$data[3:115,]-m1$residuals
sgcase_var_train_fit = m2_fitted[,1]
ukcase_var_train_fit = m2_fitted[,2]
uscase_var_train_fit = m2_fitted[,3]
sgret_var_train_fit = m2_fitted[,4]
ukret_var_train_fit = m2_fitted[,5]
usret_var_train_fit = m2_fitted[,6]

sgret_var_valid = d$SG_ret
ukret_var_valid = d$UK_ret
usret_var_valid = d$US_ret

SG_stock[1:128]
US_stock[1:128]
UK_stock[1:128]

ini_sg_1 = SG_stock[1]
ini_sg_2 = SG_stock[116]
ini_uk_1 = UK_stock[1]
ini_uk_2 = UK_stock[116]
ini_us_1 = US_stock[1]
ini_us_2 = US_stock[116]

stk_fit_sg = exp(log(ini_sg_1)+cumsum(sgret_var_train_fit))
sg_stk_train_MAPE = MAPE(SG_stock[5:116],stk_fit_sg[2:113])
sg_stk_train_MAPE

stk_fit_uk = exp(log(ini_uk_1)+cumsum(ukret_var_train_fit))
uk_stk_train_MAPE = MAPE(UK_stock[5:116],stk_fit_uk[2:113])
uk_stk_train_MAPE

stk_fit_us = exp(log(ini_us_1)+cumsum(usret_var_train_fit))
us_stk_train_MAPE = MAPE(US_stock[5:116],stk_fit_us[2:113])
us_stk_train_MAPE


stk_valid_sg = exp(log(ini_sg_2)+cumsum(sgret_var_valid))
stk_valid_uk = exp(log(ini_uk_2)+cumsum(ukret_var_valid))
stk_valid_us = exp(log(ini_us_2)+cumsum(usret_var_valid))

sg_stk_valid_MAPE = MAPE(SG_stock[117:128],stk_valid_sg)
us_stk_valid_MAPE = MAPE(US_stock[117:128],stk_valid_us)
uk_stk_valid_MAPE = MAPE(UK_stock[117:128],stk_valid_uk)

sg_stk_valid_MAPE;us_stk_valid_MAPE;uk_stk_valid_MAPE;

par(mfrow=c(1,1),pin=c(5,4))
index = 21:32
plot(SG_stock[117:128]~index, 
     main = 'SG Stock Index Prediction from 2022-06 to 2022-08',
     type = 'o',
     pch = 16,
     col = 'black',
     xlab = "Time index",
     ylab = "Stock Index(GSPC)")
lines(stk_valid_sg~index, type = 'o', pch = 16, col = 'red')

legend(x = 'topleft', 
       legend = c(expression('True numbers'), expression('Predict numbers')),
       lty = 1,
       pch = 16,
       lwd = 2,
       col = c('black', 'red'),
       bty = 'n',
       horiz = T)

par(mfrow=c(1,1),pin=c(5,4))
index = 21:32
plot(UK_stock[117:128]~index, 
     main = 'UK Stock Index Prediction from 2022-06 to 2022-08',
     type = 'o',
     pch = 16,
     col = 'black',
     xlab = "Time index",
     ylab = "Stock Index(FTSE)",
     ylim = c(7100,7650))
lines(stk_valid_uk~index, type = 'o', pch = 16, col = 'red')

legend(x = 'topright', 
       legend = c(expression('True numbers'), expression('Predict numbers')),
       lty = 1,
       pch = 16,
       lwd = 2,
       col = c('black', 'red'),
       bty = 'n',
       horiz = T)

par(mfrow=c(1,1),pin=c(5,4))
index = 21:32
plot(US_stock[117:128]~index, 
     main = 'US Stock Index Prediction from 2022-06 to 2022-08',
     type = 'o',
     pch = 16,
     col = 'black',
     xlab = "Time index",
     ylab = "Stock Index(S&P500)")
lines(stk_valid_us~index, type = 'o', pch = 16, col = 'red')

legend(x = 'topright', 
       legend = c(expression('True numbers'), expression('Predict numbers')),
       lty = 1,
       pch = 16,
       lwd = 2,
       col = c('black', 'red'),
       bty = 'n',
       horiz = T)

#Section12:MAPE for stock
garch_SG_1 = garchFit(formula= ~arma(0,1) + garch(1,1),data= sg_ret_train, cond.dist="std")
summary(garch_SG_1)
garch_US_4 = garchFit(formula= ~arma(0,1) + garch(2,1),data= us_ret_train, cond.dist="std")
summary(garch_US_4)

SG_stk_pred = predict(garch_SG_1,n.ahead=12)
US_stk_pred = predict(garch_US_4,n.ahead=12)

sg_stk_valid_arima=SG_stk_pred$meanForecast
us_stk_valid_arima=US_stk_pred$meanForecast

sg_stk_train_fit = garch_SG_1@fitted
us_stk_train_fit = garch_US_4@fitted


SG_stock[1:128]
US_stock[1:128]

ini_sg_1 = SG_stock[1]
ini_sg_2 = SG_stock[116]

ini_us_1 = US_stock[1]
ini_us_2 = US_stock[116]


stk_fit_sg = exp(log(ini_sg_1)+cumsum(sg_stk_train_fit))
sg_stk_train_MAPE = MAPE(SG_stock[5:116],stk_fit_sg[2:113])
sg_stk_train_MAPE


stk_fit_us = exp(log(ini_us_1)+cumsum(us_stk_train_fit))
us_stk_train_MAPE = MAPE(US_stock[5:116],stk_fit_us[2:113])
us_stk_train_MAPE
stk_valid_sg = exp(log(ini_sg_2)+cumsum(sg_stk_valid_arima))
stk_valid_us = exp(log(ini_us_2)+cumsum(us_stk_valid_arima))

sg_stk_valid_MAPE = MAPE(SG_stock[117:128],stk_valid_sg)
us_stk_valid_MAPE = MAPE(US_stock[117:128],stk_valid_us)

stk_valid_uk=UK_stock[116]
uk_stk_valid_MAPE = MAPE(UK_stock[117:128],stk_valid_uk)
uk_stk_valid_MAPE

#Section13:Prediction model
# US cases:AR(1)+GARCH(1,1)
US<-ts(data$US,start = c(2020,1),end=c(2022,40),frequency = 48)
lus<-log(US+1)
diffus = diff(lus)
usfit = window(diffus,start = c(2020,2),end=c(2022,32),frequency = 48)
uspre=window(diffus,start = c(2022,33),end=c(2022,40),frequency = 48)
garch.model.us.1 = garchFit(formula= ~arma(1,0) + garch(1,1),data= usfit, cond.dist="std")
summary(garch.model.us.1)
us_predict=predict(garch.model.us.1,8)

case_US = as.data.frame(US[129:136])
colnames(case_US)[1] = 'true_value'

growth_US <- us_predict$meanForecast
initial_US <- as.numeric(US[129])
case_pre_US <- round(exp(log(initial_US+1)+cumsum(growth_US)))-1
MAPE(US[129:136],case_pre_US)

#SG stock&US stock
sg_ret_forpre = window(d_SG_stock,start = c(2020,2),end=c(2022,32),frequency = 48)
sg_ret_predict = window(d_SG_stock,start = c(2022,33),end=c(2022,40),frequency = 48)
garch_SG_1 = garchFit(formula= ~arma(0,1) + garch(1,1),data= sg_ret_forpre, cond.dist="std")
summary(garch_SG_1)
us_ret_forpre = window(d_US_stock,start = c(2020,2),end=c(2022,32),frequency = 48)
us_ret_predict = window(d_US_stock,start = c(2022,33),end=c(2022,40),frequency = 48)
garch_US_4 = garchFit(formula= ~arma(0,1) + garch(2,1),data= us_ret_forpre, cond.dist="std")
summary(garch_US_4)

SG_stk_pred = predict(garch_SG_1,n.ahead=8)
US_stk_pred = predict(garch_US_4,n.ahead=8)

sg_stk_test_arima=SG_stk_pred$meanForecast
us_stk_test_arima=US_stk_pred$meanForecast

ini_sg_2 = SG_stock[128]
ini_us_2 = US_stock[128]

stk_test_sg = exp(log(ini_sg_2)+cumsum(sg_stk_test_arima))
stk_test_us = exp(log(ini_us_2)+cumsum(us_stk_test_arima))

#UK stock
lgr_forpre
m0=VARorder(lgr_forpre)
m0$Mstat
m1=VAR(lgr_forpre,2)
m2=refVAR(m1,thres=1.96)
m1_resi= m1$residuals
a=VARpred(m1,8)
c=VARpred(m2,8)
b=a$pred
b=data.frame(b)
d=c$pred
d=data.frame(d)
d$UK_ret
d
stk_test_uk = exp(log(UK_stock[128])+cumsum(d$UK_ret))