# Artificial Neural Network

# Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf

# Part 1 - Data Preprocessing

# Importing the dataset
dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# Encoding categorical data
# Label Encoding the "Gender" column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()                            # Yes/No는 하나뿐인 싱글 벡터이다. 어떤 것을 인코딩해야하는지 명확하므로 괄호를 비워도 된다.   No = 0, Yes = 1
X[:, 2] = le.fit_transform(X[:, 2]) 

# One Hot Encoding the "Geography" column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers= [('encoder', OneHotEncoder(), [1])], remainder='passthrough')    # [N] N = 변환하고 싶은 열의 인덱스 X로 받아온 데이터를 기준으로 작성 , ct객체 생성완료
X = np.array(ct.fit_transform(X))
print(X)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split                                                
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# Feature Scaling   : 딥러닝에서는 필수, 모든 부분에 대하여 feature scaling 적용 !
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)    #fit은 각 평균과 표준편차를 구하는 것이고, transform은 이 공식을 적용해서 표준화나 정규화를 하는 것이다.
X_test = sc.fit_transform(X_test)

# Part 2 - Building the ANN

# Initializing the ANN
ann = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
ann.add(tf.kears.layers.Dense(units = 6, activation='relu'))

# Adding the second hidden layer
ann.add(tf.kears.layers.Dense(units = 6, activation='relu'))

# Adding the output layer
ann.add(tf.kears.layers.Dense(units = 1, activation='sigmoid'))   #출력할 값이 0 or 1 이므로 뉴런 1개로 커버 가능 , N 분형 예측일 경우 sigmoid 함수는 사용할 수 없음

# Part 3 - Training the ANN

# Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])   #2분형 예측 = binary_crossentropy   N분형 예측 = categorical_crossentropy

# Training the ANN on the Training set
ann.fit(X_train, y_train, batch_size=32, epochs = 100)   #배치 학습 확률적 G. D 방법에서 몇 개씩 비교할 것이냐를 정하는 것,   신경망은 에포크 수를 거쳐 학습한다

# Part 4 - Making the predictions and evaluating the model

# Predicting the result of a single observation
# 1. 쌍 대괄호, 2. 인코딩 된 값 넣어주기, 3. feature scaling 해줘서 입력해주기
y_pred = ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]) > 0.5)

# Predicting the Test set results
y__pred = ann.predict(X_test)
y__pred = (y__pred > 0.5)
print(np.concatenate((y__pred.reshape(len(y__pred),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y__pred)
print(cm)
score = accuracy_score(y_test, y__pred)
print(score)