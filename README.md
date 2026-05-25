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
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# ------------------------------
# Step 1: Sample dataset
# ------------------------------
data = {
    'Position': ['Business Analyst', 'Junior Consultant', 'Senior Consultant',
                 'Manager', 'Country Manager', 'Region Manager',
                 'Partner', 'Senior Partner', 'C-level', 'CEO'],
    'Level': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Salary': [45000, 50000, 60000, 80000, 110000, 150000, 200000, 300000, 500000, 1000000]
}

df = pd.DataFrame(data)

# ------------------------------
# Step 2: Split features and target
# ------------------------------
X = df[['Level']]     # Feature (Level)
y = df['Salary']      # Target (Salary)

# ------------------------------
# Step 3: Create Decision Tree Regressor
# ------------------------------
regressor = DecisionTreeRegressor(random_state=42)
regressor.fit(X, y)

# ------------------------------
# Step 4: Predict salary for the dataset or new levels
# ------------------------------
y_pred = regressor.predict(X)
print("Predicted salaries:", y_pred)

# Example: predict salary for a new employee at level 6.5
level = np.array([[6.5]])
predicted_salary = regressor.predict(level)
print(f"Predicted Salary for level {level[0][0]}: {predicted_salary[0]}")

# ------------------------------
# Step 5: Visualize the results (High-resolution curve)
# ------------------------------
X_grid = np.arange(min(X.values), max(X.values)+0.01, 0.01)  # High-resolution for smoother curve
X_grid = X_grid.reshape(-1, 1)

plt.scatter(X, y, color='red', label='Actual Salary')
plt.plot(X_grid, regressor.predict(X_grid), color='blue', label='Decision Tree Prediction')
plt.title('Decision Tree Regression: Level vs Salary')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.legend()
plt.show()
```

## Output:

<img width="984" height="81" alt="image" src="https://github.com/user-attachments/assets/50b447a6-3fdf-4989-a583-b60a91f43cec" />

<br>

<img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/04da79c6-39c5-4bd9-94ae-5ca14bfde7be" />

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
