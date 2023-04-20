### Regression Tree Model & Random Forest Model ###


#Load in necessary libraries
library(ggplot2)
library(tidyverse) 
library(randomForest)
library(tree)
library(caret)


#Upload the dataset
crime_data = read.table("uscrime.txt", stringsAsFactors= FALSE, header=TRUE) 

#To ensure codes are reproducible, set seed
set.seed(1) 

#See what data looks like 
head(crime_data) 

tail(crime_data) 

#Summary of data 
summary(crime_data)


#Fit regression tree model
tree_model = tree(Crime~., data = crime_data)

#Summary of model
summary(tree_model)


#Look at tree split
tree_model$frame


#Plot Model
plot(tree_model)
text(tree_model)


#Prune tree
nodes = 5
prune_tree_model = prune.tree(tree_model, best = nodes)

#Plot Model
plot(prune_tree_model)
text(prune_tree_model)

#Summary of prune tree model
summary(prune_tree_model)


#Use 7 node for prune model
nodes = 7

#7 node prune tree
prune_tree_model_2 = prune.tree(tree_model, best = nodes)


#Plot model
plot(prune_tree_model_2)
text(prune_tree_model_2)

#Summary of model
summary(prune_tree_model_2)

#Predict crime using 7 node model 
tree_pred = predict(prune_tree_model_2, data = crime_data[,1:15])

#Calculate residual sum of squares
RSS = sum((tree_pred - crime_data[,16])^2)

#Calculate total sum of squares
TSS = sum((crime_data[,16] - mean(crime_data[,16]))^2)

#Calculate R-SQUARED

R2 = 1 - RSS/TSS

#Display R-SQUARED
R2

#Fit random forest model
random_forest = randomForest(Crime ~ ., data=crime_data, importance = TRUE, nodesize = 5)


#Predict crime using model 
random_forest_pred = predict(random_forest, data=crime_data[,-16])


#Calculate residual sum of squares
RSS = sum((random_forest_pred - crime_data[,16])^2)

#Calculate total sum of squares
TSS = sum((crime_data[,16] - mean(crime_data[,16]))^2)

#Calculate R-SQUARED
R2 = 1 - RSS/TSS

#Display R-SQUARED
R2

#High r2 at 0.72. Possibly, overfitting.