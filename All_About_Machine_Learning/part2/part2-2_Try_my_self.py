# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[3])] , remainder='passthrough')       # 열의 인덱스를 데이터에 맞게 바꿔줘야함.
x = np.array(ct.fit_transform(x))                                                                       # 인코딩된 결과는 처음열로 나온다. 어떤 범주가 어떻게 인코딩되었는지 확인해봐야함

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()                       # 회귀 클래스 안에 디폴트 값을 가지고 있어서 우리는 매개변수를 넣을 필요가 없다.
regressor.fit(x_train, y_train)


# Predicting the Test set results
y_pred = regressor.predict(x_test)           ###### 이 부분이 이해가 안돼     -> 이해 완료
np.set_printoptions(precision=2)             # 소수점 아래 2 자리까지
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))   # 첫 번째 인자는 내가 합치고 싶은 배열이나 벡터의 튜플형(같음 모양이어야함), 두 번째 인자는 세로 합치기 or 가로 합치기(원본 배열에서 생각)
