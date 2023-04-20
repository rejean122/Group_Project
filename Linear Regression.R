#Set Working directory 
setwd("/Users/raphyjean/Desktop/GTX 6501/hw5-SP22") 


#Upload the dataset
crime_data = read.table("uscrime.txt", stringsAsFactors= FALSE, header=TRUE) 

#To ensure codes are reproducible, set seed
set.seed(1) 

#See what data looks like
head(crime_data) 

tail(crime_data) 

#Summary for crime column
summary(crime_data[16])


#Create a test dataframe using the sample data given above
test_df=data.frame(M = 14.0,So = 0, Ed = 10.0, Po1 = 12.0, 
                   Po2 = 15.5,LF = 0.640, M.F = 94.0, 
                   Pop = 150, NW = 1.1, U1 = 0.120, 
                   U2 = 3.6, Wealth = 3200, Ineq = 20.1, 
                   Prob = 0.04,Time = 39.0)

#Display test data frame to ensure its correct
test_df


#Fit regression model to data
crime_data_lm=lm(Crime~M+So+Ed+Po1+Po2+LF+M.F+Pop+NW+U1+U2+Wealth+Ineq+Prob+Time,crime_data)


#Summary of model
summary(crime_data_lm)


#Predict crime of sample data using model
crime_pred_1 = predict(crime_data_lm, test_df)

#Display prediction
crime_pred_1


#Create new model where only variables with a p-value less than .05 are included. So Crime, M, Ed, Ineq, Prob
crime_data_p_05_lm =lm(Crime~M+Ed+Ineq+Prob,crime_data)


#Summary of new model
summary(crime_data_p_05_lm)


#Predict crime of sample data using new adjusted model
crime_pred = predict(crime_data_p_05_lm, test_df)

#Display prediction
crime_pred


#Install metrics package
install.packages('Metrics')


#Load metrics library
library(Metrics)


#Find predicted values for original 15 variable model
crime_data_predicted = c(predict(crime_data_lm, crime_data))


#Display predicted values
crime_data_predicted


#Calculate root mean square error on actual vs predicted crime values
rmse(crime_data$Crime, crime_data_predicted)


#Find predicted values for new adjusted model
crime_data_predicted = c(predict(crime_data_p_05_lm, crime_data))


#Display predicted values
crime_data_predicted


#Calculate root mean square error on actual vs predicted crime values
rmse(crime_data$Crime, crime_data_predicted)


#Install DAAG package for cross validation
install.packages("DAAG")

#Load in DAAG library
library(DAAG)

#Calculate total sum of squared difference between data and mean
sum_squared_difference = sum((crime_data$Crime - mean(crime_data$Crime))^2)

sum_squared_difference


#Perform 5 fold cross validation on original 15 variable model
five_fold_cv_model1 = cv.lm(crime_data, crime_data_lm, m=5)


#Perform 5 fold cross validation on adjusted 5 variable model
five_fold_cv_model2 = cv.lm(crime_data, crime_data_p_05_lm, m=5)


#Calculate total sum of squared difference between data and mean for model 1
Sum_squared_model1 = attr(five_fold_cv_model1,"ms")*nrow(crime_data)

Sum_squared_model1

#Cross validation for model 1
1- Sum_squared_model1/sum_squared_difference

#Calculate total sum of squared difference between data and mean for model 2
Sum_squared_model2 = attr(five_fold_cv_model2,"ms")*nrow(crime_data)

Sum_squared_model2

#Cross validation for model 2
1- Sum_squared_model2/sum_squared_difference













