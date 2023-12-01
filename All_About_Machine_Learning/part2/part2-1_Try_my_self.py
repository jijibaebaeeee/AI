# Simple Linear Regression

# Importing the libraries      : 라이브러리 임포트 하고~
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset                 : 데이터 셋 불러 오고~ 특징 행렬과 종속 변수 벡터 만들어주고~
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set     :       train_test_split 함수를 불러와서 훈련세트와 테스트세트로 분리한다~
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Training the Simple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()                   # 단순 선형 모델의 인스턴스이다. 이것을 훈련세트에 연결해야함
regressor.fit(x_train, y_train)                  # fit 메소드를 사용하여 regressor과 훈련세트를 연결함

# Predicting the Test set results
y_pred = regressor.predict(x_test)     #메소드를 호출하려면 객체 그 자체부터 호출해야 한다.


# Visualising the Training set results                # 실제 급여는 빨간색, 예측 급여는 파란색
plt.scatter(x_train, y_train, color = 'red')                   # scatter함수는 2D 구성에 점을 찍는다 색을 정할 수 있다.
plt.plot(x_train, regressor.predict(x_train), color = 'blue')    # 회귀선이므로 scatter말고 plot 메서드를 사용 plot(x좌표,y좌표)  y좌표 = 훈련 세트의 예측 급여로 했음
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')    # 훈련 세트의 회귀선과 같은 회귀선이 나타난다.
plt.title('Salary vs Experience (Test set)')                     # 단순 선형 회귀 모델의 회귀선은 고유한 식에서 도출되므로
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()