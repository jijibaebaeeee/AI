# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the Linear Regression model on the whole dataset
from sklearn.linear_model import LinearRegression          # 단순 선형 회귀와 다항 선형 회귀의 성능을 비교하고자 단순 선형 회귀 모델도 만든다.
lin_reg = LinearRegression()
lin_reg.fit(x, y)               # 원래는 훈련세트와 테스트세트로 나누고 훈련 세트를 매개변수로 넣어주는데, 최대한 데이터를 활용해서 모델을 훈련시키기 위해 전체를 넣어준다.


# Training the Polynomial Regression model on the whole dataset       다중 선형 회귀는 y = b0 + b1x1 + b2x2 + b3x3 ... 이런식으로 나가는데 다항 선형 회귀는 x2 = x1^2 , x3 = x1^3 .. 의 형태이다.
from sklearn.preprocessing import PolynomialFeatures                  #전처리에 있는 이유는 x1이라는 특성을 x1^2, x1^3, x1^n으로 전처리 할 것이기 때문이다. 다항 선형 회귀를 만드는 방법, 다양한 제곱된 특성들의 행렬을 만들고(특성 행렬), 단순 선형 회귀를 만들어서 둘의 특성을 통합하면 된다.
poly_reg = PolynomialFeatures(degree = 4)                             # n = 2 라는 의미
x_poly = poly_reg.fit_transform(x)                                    # 경력에 해당하는 부분의 특성을 제곱한 행렬을 만들었다.
lin_reg_2 = LinearRegression()                                        # 새로운 단순 선형 회귀 모델을 만들고
lin_reg_2.fit(x_poly, y)                                              # 이 모델에 위에서 만든 행렬과, 종속 변수 벡터를 넣어준다.

# Visualising the Linear Regression results
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg.predict(x), color = 'blue')
plt.title('Truth or Bluff(Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg_2.predict(x_poly), color = 'blue')     # lin_reg_2를 이용할 때는 다양한 제곱으로 변환이 완료된 특성 x의 행렬에 적용해야한다. 
plt.title('Truth or Bluff(polynomial Regression)')               # x는 단일 특성이므로 그대로 사용할 수가 없는 것이다.
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(x), max(x), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
print(lin_reg.predict([[6.5]]))    # 원하는 예상값을 배열로 입력해야한다.  [[6.5,5]] => 행 1개와 열 2개의 배열  [[6.5,5],[2,3]] => 행 2개와 열 2개의 배열
                                   # 단순 선형 회귀는 예측값이 과대되어 나온다.

# Predicting a new result with Polynomial Regression
print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))