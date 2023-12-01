# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the Random Forest Regression model on the whole dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)               # 의사 결정 회귀 트리와는 다르게 트리의 수를 입력해줘야한다.
regressor.fit(x, y)

# Predicting a new result
y_pred = regressor.predict([[6.5]])


# Visualising the Random Forest Regression results (higher resolution)
X_grid = np.arange(min(x), max(x), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Randomforest tree Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()