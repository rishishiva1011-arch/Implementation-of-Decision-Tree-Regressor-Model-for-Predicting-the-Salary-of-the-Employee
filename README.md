# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: Prepare the Dataset
Define employee data with Position, Level (1–10), and Salary columns. Convert it into a DataFrame where Level represents job hierarchy and Salary is the corresponding compensation.

Step 2: Split Features & Target
Extract Level as the input feature matrix X and Salary as the target vector y for training the regression model.

Step 3: Train Decision Tree Regressor
Initialize DecisionTreeRegressor with random_state=42 and fit it on X and y. The model learns to map each level to a salary range by recursively splitting the data to minimize prediction error.

Step 4: Predict Salary
Predict salaries for all existing levels in the dataset and print results. Also predict salary for a new input level (6.5) by passing it as a 2D array and print the estimated salary.

Step 5: Visualize the Results
Create a high-resolution range of level values using np.arange with step 0.01. Plot actual salaries as red scatter points and the Decision Tree predictions as a blue step curve to visualize how the model maps levels to salary ranges.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Rishikesh S
RegisterNumber:  212225240118
*/
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree

df = pd.read_csv("Salary.csv")

X = df[['Level']]

y = df['Salary']

model = DecisionTreeRegressor(random_state=42)

model.fit(X, y)

prediction = model.predict([[6.5]])

print("Predicted Salary:", prediction[0])

plt.scatter(df['Level'], df['Salary'])

plt.plot(df['Level'], model.predict(X))

plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.title("Decision Tree Regression")

plt.show()

plt.figure(figsize=(12,8))

plot_tree(model,
          feature_names=['Level'],
          filled=True)

plt.title("Decision Tree Regressor Tree")

plt.show()
```

## Output:


<img width="295" height="31" alt="image" src="https://github.com/user-attachments/assets/83dc05dc-e953-48db-bc33-bff649f425c7" />

<br>

<img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/cd66d90f-2749-42ce-8746-f93e6eecdbf7" />


<br>

<img width="950" height="658" alt="image" src="https://github.com/user-attachments/assets/1b74af0e-099f-4c78-b4d3-0b2e5383f99d" />


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
