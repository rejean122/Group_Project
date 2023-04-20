###Exponential Smoothing Model

#Set Working directory 
setwd("/Users/raphyjean/Desktop/GTX 6501/hw4-SP22") 


#Upload the dataset
temps_data = read.table("~/Desktop/GTX 6501/hw4-SP22/temps.txt", header=TRUE) 

#To ensure codes are reproducible, set seed

set.seed(1) 

#See what data looks like (tail data for column r1 looks that way because I was split screen on Mac) 
head(temps_data) 


#tail(temps_data) 


#Install smooth package
install.packages("smooth")

#Install forecast package  
install.packages("forecast")

#Create vector of the temperature data
temps_data = as.vector(unlist(temps_data[,2:21]))

#Transform data to time series
temp_time_series = ts(temps_data, frequency=123, start=1996)

#Plot time series data
plot.ts(temp_time_series)

#Perform exponential smoothing using Holt Winter Method
temp_time_series_smoothing = HoltWinters(temp_time_series, beta = NULL, gamma = NULL, seasonal = "additive")


#Plot the model
plot(temp_time_series_smoothing)

#Examine each smoothing parameter model

#Alpha
temp_time_series_smoothing $alpha


#Beta
temp_time_series_smoothing $beta

#Gamma
temp_time_series_smoothing $gamma


#Plot the Holt Winters decomposition of time series
plot(fitted(temp_time_series_smoothing))