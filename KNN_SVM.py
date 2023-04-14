# ----- MGT 6203 Analytics Project KNN & SVM Models -----


#Import packages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn
import sklearn
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm
from sklearn.metrics import f1_score



### Employee Churn Dataset ###

#Read in employee churn dataset
df = pd.read_csv("employee_churn_data.csv")

#Display first five rows of dataset
print(df.head())


### Initial Data Visualization ###



# 1. Department Turnover Count Plot

seaborn.countplot(y='department', data=df, hue="left",palette = 'Set2', edgecolor = 'b')
plt.title('Department Turnover')
plt.xlabel('Count')
plt.ylabel('Department')
plt.legend(loc = 'upper right', fontsize = 12)
plt.show()


# 2. Review Satisfaction scatter visualization

seaborn.scatterplot(x='review', y='satisfaction', data = df, hue = 'left', palette = 'Set2', edgecolor = 'b', s = 150,
               alpha = 0.7)
plt.title('REVIEW / SATISFACTION')
plt.xlabel('Review')
plt.ylabel('Satisfaction')
plt.legend(loc = 'upper right', fontsize = 12)
plt.show()



#----- KNN Classifcation With Scikit-Learn -----



### Preprocessing Data for KNN Regression ###

#Change left column to binary data type
df.loc[df.left == 'no', 'left'] = 0
df.loc[df.left == 'yes', 'left'] = 1


#Change salary column to numeric, mutate each salary type
df.loc[df.salary == 'low', 'salary'] = 1
df.loc[df.salary == 'medium', 'salary'] = 2
df.loc[df.salary == 'high', 'salary'] = 3

#Change department column to numeric, mutate each department type
df.loc[df.department == 'admin', 'department'] = 1
df.loc[df.department == 'engineering', 'department'] = 2
df.loc[df.department == 'finance', 'department'] = 3
df.loc[df.department == 'IT', 'department'] = 4
df.loc[df.department == 'logistics', 'department'] = 5
df.loc[df.department == 'marketing', 'department'] = 6
df.loc[df.department == 'operations', 'department'] = 7
df.loc[df.department == 'retail', 'department'] = 8
df.loc[df.department == 'sales', 'department'] = 9
df.loc[df.department == 'support', 'department'] = 10

#Assign response column to Y
y = df['left']

#Drop response column and assign rest to x
x = df.drop(['left'], axis = 1)

#Transpose results, transforming rows into columns and describe data
print(x.describe().T)


#### Split data into Training & Test Sets ###

#Use train_test_split method from scikit-Learn to create train and test splits, randomly. Set seed to make results reproducible
SEED = 42
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=SEED)

#Diplay length of full sample, training set, test set
print(len(x))
print(len(X_train))
print(len(X_test))


### Perform Feature Scaling KNN Regression ###

#Create scaler
scaler = StandardScaler()

#Fit scaler to x training data
scaler.fit(X_train)

#Scale x training set and x test set
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

### Training and Predicting KNN Classification ###

#Set default for K, deault is k=5
classifier = KNeighborsClassifier()

#Fit knn regression model to training data
classifier.fit(X_train.astype(int), y_train.astype(int))

#Prediction on test data
y_pred = classifier.predict(X_test)


### Evaluate KNN Classification Model ###

#Display classification report with k=5
print(classification_report(y_test.astype(int), y_pred.astype(int)))


### K-Fold Cross Validation to find best K ###

#Create loop to run models for 1 to 40 neighbors
f1s = []

# Calculating f1 score for K values between 1 and 40
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train.astype(int), y_train.astype(int))
    pred_i = knn.predict(X_test)
    # using average='weighted' to calculate a weighted average for the yes & no
    f1s.append(f1_score(y_test.astype(int), pred_i.astype(int), average='weighted'))

#Plot mean absolute error results
plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), f1s, color='blue', linestyle='solid')
plt.title('K Value / Accuracy ')
plt.xlabel('K Value')
plt.ylabel('Accuracy')
plt.show()

#Highest f1-score of K=21

#Retrain classifier with K=21
classifier20 = KNeighborsClassifier(n_neighbors=21)
classifier20.fit(X_train.astype(int), y_train.astype(int))
y_pred20 = classifier20.predict(X_test.astype(int))

#Model Accuracy
print("Accuracy:",metrics.accuracy_score(y_test.astype(int), y_pred.astype(int)))

#Model Precision
print("Precision:",metrics.precision_score(y_test.astype(int), y_pred.astype(int)))

#Model Recall
print("Recall:",metrics.recall_score(y_test.astype(int), y_pred.astype(int)))

#Model F-1 Score:
print("F-1 Score:", metrics.f1_score(y_test.astype(int), y_pred.astype(int), average=None))


#Display classification report with k=21
print(classification_report(y_test.astype(int), y_pred20.astype(int)))

#Create Confusion Matrix
column_names = ["no", "yes"]
cm = pd.DataFrame(confusion_matrix(y_test.astype(int), y_pred.astype(int)), columns = column_names, index = column_names)

# Seaborn's heatmap to better visualize the confusion matrix
seaborn.heatmap(cm, annot=True, fmt='d', cmap = "Blues");
plt.title("KNN Confusion Matrix", fontsize =12)
plt.show()

#----- Support Vector Machine (SVM) With Scikit-Learn -----


### Employee Churn Dataset ###


#Read in employee churn dataset
df = pd.read_csv("/Users/raphyjean/Desktop/GTX 6203/Group Project/employee_churn_data.csv")

#Display first five rows of dataset
print(df.head())


### Preprocessing Data for SVM ###

#Change left column to binary data type
df.loc[df.left == 'no', 'left'] = 0
df.loc[df.left == 'yes', 'left'] = 1


#Change salary column to numeric, mutate each salary type
df.loc[df.salary == 'low', 'salary'] = 1
df.loc[df.salary == 'medium', 'salary'] = 2
df.loc[df.salary == 'high', 'salary'] = 3

#Change department column to numeric, mutate each department type
df.loc[df.department == 'admin', 'department'] = 1
df.loc[df.department == 'engineering', 'department'] = 2
df.loc[df.department == 'finance', 'department'] = 3
df.loc[df.department == 'IT', 'department'] = 4
df.loc[df.department == 'logistics', 'department'] = 5
df.loc[df.department == 'marketing', 'department'] = 6
df.loc[df.department == 'operations', 'department'] = 7
df.loc[df.department == 'retail', 'department'] = 8
df.loc[df.department == 'sales', 'department'] = 9
df.loc[df.department == 'support', 'department'] = 10

#Assign response column to Y
y = df['left']

#Drop response column and assign rest to x
x = df.drop(['left'], axis = 1)

#Transpose results, transforming rows into columns and describe data
print(x.describe().T)


#### Split data into Training & Test Sets ###

#Use train_test_split method from scikit-Learn to create train and test splits, randomly. Set seed to make results reproducible
SEED = 42
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=SEED)

#Diplay length of full sample, training set, test set
print(len(x))
print(len(X_train))
print(len(X_test))


### Perform Feature Scaling SVM ###

#Create scaler
scaler = StandardScaler()

#Fit scaler to x training data
scaler.fit(X_train)

#Scale x training set and x test set
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


### Training and Predicting SVM Model ###


#Import svm model
from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel='rbf') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train.astype(int), y_train.astype(int))

#Predict the response for test dataset
y_pred = clf.predict(X_test.astype(int))

#Model Accuracy
print("Accuracy:",metrics.accuracy_score(y_test.astype(int), y_pred.astype(int)))

#Model Precision
print("Precision:",metrics.precision_score(y_test.astype(int), y_pred.astype(int)))

#Model Recall
print("Recall:",metrics.recall_score(y_test.astype(int), y_pred.astype(int)))

#Model F-1 Score:
print("F-1 Score:", metrics.f1_score(y_test.astype(int), y_pred.astype(int), average=None))


#Display classification report
print(classification_report(y_test.astype(int), y_pred.astype(int)))

#Create Confusion Matrix
column_names = ["no", "yes"]
cm = pd.DataFrame(confusion_matrix(y_test.astype(int), y_pred.astype(int)), columns = column_names, index = column_names)

# Seaborn's heatmap to better visualize the confusion matrix
seaborn.heatmap(cm, annot=True, fmt='d', cmap="Greens");
plt.title("SVM Confusion Matrix", fontsize =12)
plt.show()

