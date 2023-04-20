### Principal Component Analysis ###


#Load in DAAG and tidyverse library 
library(DAAG) 
library(tidyverse) 


#Upload the dataset
crime_data = read.table("uscrime.txt", stringsAsFactors= FALSE, header=TRUE) 

#To ensure codes are reproducible, set seed
set.seed(1) 

#See what data looks like (tail data for column r1 looks that way because I was split screen on Mac) head(crime_data) 


#Summary of data 
summary(crime_data)


#Fit PCA model to data not using crime column
pca_model = prcomp(crime_data[,1:15], scale = TRUE)

#Summary of model
summary(pca_model)


#Plot Model
screeplot(pca_model, type="line")

#Set number of principal coordinates I will use.
pc = pca_model$x[,1:5]

#Summary of pcs
summary(pc)

#Add Pcs to original crime data
crime_data_pca = cbind(pca_model$x[,1:5],crime_data[,16])


#Fit Linear Regression model to new combined data
us_crime_pca_LR = lm(V6~., data=as.data.frame(crime_data_pca))


#Summary of model
summary(us_crime_pca_LR)


#Display intercept and coefficients from model
us_crime_pca_LR$coefficients


#Assign intercept
Intercept = us_crime_pca_LR$coefficients[1]

#Assign coefficients to create beta vector
betas = us_crime_pca_LR$coefficients[2:6]

#Create alpha vector (rotated matrix *beta vector)
alphas = pca_model$rotation[,1:5]%*%betas

#Get original alpha values
original_alpha = alphas/sapply(crime_data[,1:15], sd)

#Get original alpha values
original_beta = Intercept - sum(alphas*sapply(crime_data[,1:15],mean)/sapply(crime_data[,1:15], sd))


#Find estimates
Estimates = as.matrix(crime_data[,1:15]) %*% original_alpha + original_beta

#Calculate adjusted R-squared and R-squared

SSE = sum((Estimates-crime_data[,16])^2)
SS_total = sum((crime_data[,16]-mean(crime_data[,16]))^2)
R_squared = 1- (SSE/SS_total)
R_squared_adjusted= R_squared - (1-R_squared)*5/(nrow(crime_data)-5-1)

#Display R-squared
R_squared

#Display R-squared adjusted
R_squared_adjusted

#Create a test dataframe using the sample data given last week 
test_df=data.frame(M = 14.0,So = 0, Ed = 10.0, Po1 = 12.0, Po2 = 15.5,
                   LF = 0.640, M.F = 94.0, Pop = 150, 
                   NW = 1.1, U1 = 0.120, U2 = 3.6,
                   Wealth = 3200, Ineq = 20.1, 
                   Prob = 0.04,Time = 39.0)

#Perform prediction of last week test data using new model
prediction_test_df = data.frame(predict(pca_model, test_df))
prediction_test_df_model = predict(us_crime_pca_LR, prediction_test_df)

#Display model prediction
prediction_test_df_model
