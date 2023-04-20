### Optimization Model ###

#install & load the necessary libraries
!pip install pandas
!pip install pulp
!pip install xlrd

from pulp import *
import pandas as pd


#load the diet data
diet =pd.read_excel("diet.xls")

# print first 5 rows of diet data 
diet.head()

# remove bottom intake row
cleaned_data = diet[0:64]

#convert to list
cleaned_data = cleaned_data.values.tolist()


#Make master food dictionary
foods = [x[0] for x in cleaned_data]
calories = dict([(x[0], float(x[3])) for x in cleaned_data])
cholesterol = dict([(x[0], float(x[4])) for x in cleaned_data])
totalFat = dict([(x[0], float(x[5])) for x in cleaned_data])
sodium = dict([(x[0], float(x[6])) for x in cleaned_data])
carbs = dict([(x[0], float(x[7])) for x in cleaned_data])
fiber = dict([(x[0], float(x[8])) for x in cleaned_data])
protien = dict([(x[0], float(x[9])) for x in cleaned_data])
vitaminA = dict([(x[0], float(x[10])) for x in cleaned_data])
vitaminC = dict([(x[0], float(x[11])) for x in cleaned_data])
calcium = dict([(x[0], float(x[12])) for x in cleaned_data])
iron = dict([(x[0], float(x[13])) for x in cleaned_data])



#Make min & max list for the foods
min_food_list = [1500, 30, 20, 800, 130, 125, 60, 1000, 400, 700, 10]
max_food_list = [2500, 240, 70, 2000, 450, 250, 100, 10000, 5000, 1500, 40]


#add contraints for each column 
constraints= []
for h in range(0,11):
    constraints.append(dict([(x[0], float(x[h+3])) for x in cleaned_data]))


#Create cost dictionary
cost_dictionary = dict([(x[0], float(x[1])) for x in cleaned_data])


#Fit optimization problem framework (Minimization)
optimzation_problem = LpProblem('PuLPTutorial', LpMinimize)


#Define continous variables
continous_variables = LpVariable.dicts("foods", foods,0)


#Define binary variables
binary_var = LpVariable.dicts("Chosen",foods,0,1,"Binary")


#Dictionary of optimization variables 
opt_dict= LpVariable.dicts("x", foods, 0)


#Define Objective function
optimzation_problem += lpSum([cost_dictionary[x] * continous_variables[x] for x in foods])


#Constraints for  foods
for x in range(0,11):
    c1 = pulp.lpSum([constraints[x][y] * continous_variables[y] for y in foods])
    c2 = min_food_list[x] <= + c1
    optimzation_problem += c2
    
for i in range(0,11):
    c1 = pulp.lpSum([constraints[x][y] * continous_variables[y] for y in foods])
    c3 = max_food_list[x] >= + c1
    optimzation_problem += c3



#solve the optimization problem!
optimzation_problem.solve()


#print the foods of the optimal diet
print('Solution:')
for variable in optimzation_problem.variables():
    if variable.varValue > 0:
        if str(variable).find('Chosen'):
            print(str(variable.varValue) + " units of " + str(variable))
            
#print the costs of the optimal diet             
print("Total cost of food = $%.2f" % value(optimzation_problem.objective))
