# Decision Tree Regression    :    의사 결정 트리는 연속적인 노드를 통해 데이터를 분할한다.

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the Decision Tree Regression model on the whole dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(x, y)



# Predicting a new result
y_pred = regressor.predict([[6.5]])


# Visualising the Decision Tree Regression results (higher resolution)     :   특성의 범위가 달라도 예측은 같을 수 있다.
X_grid = np.arange(min(x), max(x), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Decision Tree Regressor')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()