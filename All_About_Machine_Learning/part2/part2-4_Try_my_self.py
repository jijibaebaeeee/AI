# Support Vector Regression (SVR)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

print(x)
print(y)
y = y.reshape(len(y),1)       # 특성 스케일링의 Standard 클래스가 사용할 수 있는 2차원 배열로 입력으로 만든다.
print(y) 
# 데이터를 훈련세트와 테스트세트로 나누지 않는다. 직위 수준과 연봉의 상관관계를 학습 하기 위해 모두다 필요하다.

# Feature Scaling         SVR 모델에는 특성에 대한 명시적인 종속 변수의 방정식이 없기때문에 스케일링을 해야한다.
from sklearn.preprocessing import StandardScaler                             # 종속변수 y와 특성 x에 대한 특정한 방정식이 있을 때 특성 스케일링을 적용해야한다.
sc_x = StandardScaler()          # 하나의 StandardScaler 로 x 와 y 의 데이터를 fit 해버리면 x의 평균과 표준편차 값이 먼저 들어가고, y는 그 평균과 표준편차는 다르므로 2개의 객체를 각각 fit해야한다. *****중요
sc_y = StandardScaler()
x = sc_x.fit_transform(x)               # 각각의 sc 객체가 각각의 데이터의 평균과 표준편차를 계산한다.
y = sc_y.fit_transform(y)

print(x)
print(y)     #-1에서 3의 값

# Training the SVR model on the whole dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')    # 커널 rbf인 SVR 모델을 만들었어요.
regressor.fit(x,y)

# Predicting a new result       # 스케일링 하기 전의 값으로 돌아갈 수 있다.
result_y = sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])).reshape(-1,1))      # predict 메소드는 2차원 배열을 입력으로 요구한다. 예측한 값을 y의 형식으로 다시 역변환을 위해서 sc_y의 inverse_transform을 한다.
print(result_y)                                                                                  #reshape은 형식 오류를 피하기 위해서 작성

# Visualising the SVR results
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color = 'red')            #시각화를 위해서는 다시 원래 데이터 값으로 돌아와야 합니다.
plt.plot(sc_x.inverse_transform(x), sc_y.inverse_transform(regressor.predict(x).reshape(-1,1)), color = 'blue')        # predict 메소드는 스케일링된 값에 적용해야한다.
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(sc_x.inverse_transform(x)), max(sc_x.inverse_transform(x)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color = 'red')
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_x.transform(X_grid)).reshape(-1,1)), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()




# 스케일링 한 값을 필요로 하는가. 아니면 스케일링 전의 값을 필요로 하는가를 정확하게 알아서 입, 출력 해야한다.