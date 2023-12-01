# Data Preprocessing Tools    : 데이터 전처리 도구들

# Importing the libraries    :      라이브러리 임포트  ->  아래의 데이터 전처리 도구들을 사용할 수 있게 된다.
import numpy as np                  # 배열을 가지고 작업할 수 있게 한다. 머신러닝 모델도 입력으로 배열을 요구할 수 있음
import matplotlib.pyplot as plt     # 멋진 차트를 만들 수 있는 라이브러리(모듈의 예시) matplotlib + pyplot(특정 모듈) 모두 불러온다.   => matplotlib 라이브러리의 다른 모듈인 pyplot에도 접근할 수 있다.
import pandas as pd                 # 데이터 세트를 불러오고, 특징 행렬과 종속변수 벡터를 생성하기도 한다.


# Importing the dataset   :       <데이터 세트 불러오기>
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values    # x는 특징 행렬인데, 맨 마지막의 예측열을 제외한 모든 데이터값을 가진다. 코드의 뜻은 모든 행과 , 맨 마지막을 제외한 모든 열을 포함하겠다!이다
y = dataset.iloc[:, -1].values     # y는 종속변수 벡터인데 우리가 할 예측결과를 저장하는 변수이다. 코드의 뜻은 모든 행과, 맨 마지막 열만을 포함하겠다!이다
                                   # x와 y 두개의 엔티티를 선언하는 이유는 모델들을 만들 때 몇몇 클래스를 사용할 건데 이 클래스들은 데이터 세트 전체를 원하지 않고, 두 개의 분리된 엔티티를 원하기 때문입니다.

print(x)
print(y)

# Taking care of missing data    : <누락된 데이터 처리하기>
# 누락된 데이터는 모델을 훈련시킬 때 오류가 생길 수 있으므로 반드시 처리해야한다.
# 첫 번째 방법으로는 그냥 무시하는 것이고, 두 번째 방법으로는 누락된 데이터 집합의 평균을 내서 누락된 데이터를 처리하는 방식이다.
# 최고의 과학 라이브러리 사이킷런
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy = 'mean')    # SimpleImputer의 인스턴스 imputer를 생성하고 누락된 값에대해 평균을 적용하는 객체 생성
imputer.fit(x[:,1:3])                                                # SimpleImputer 클래스의 fit 메소드가 객체 imputer와 특징행렬(x)를 연결해준다. fit 함수가 결측치를 보고 Salary에 평균값을 계산한다는 것
x[:,1:3] = imputer.transform(x[:,1:3])                               #transform 메소드는 누락된 Salary를 Salary의 평균값으로 대체해준다.
print(x)

# Encoding categorical data :    <범주형 데이터를 인코딩하기>
#                                문자를 숫자로 바꿔야하는데 독일, 스페인, 프랑스를 각각 0,1,2로 바꿔버리면 머신러닝 모델이 숫자사이의 관계가 있다고 여기기때문에 one-hot 인코딩 방식을 이용한다.
#                                독일 (1 0 0), 스페인 (0 1 0), 프랑스 (0 0 1)로 이진 벡터화해서 인코딩하면 세 개의 열로 바꾸는 것이며, 숫자의 순서가 사라진다.
    # Encoding the Independent Variable    : [독립 변수 인코딩하기] (2개의 클래스 사용)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers= [('encoder', OneHotEncoder(), [0])], remainder='passthrough')    # 두 개의 인자로는 (변환하고싶은열, 유지하고싶은열) , ct객체 생성완료
x = np.array(ct.fit_transform(x))
print(x)

    # Encoding the Dependent Variable    :     [종속 변수 인코딩하기]
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()                            # Yes/No는 하나뿐인 싱글 벡터이다. 어떤 것을 인코딩해야하는지 명확하므로 괄호를 비워도 된다.   No = 0, Yes = 1
y = le.fit_transform(y)                        # 종속 변수 벡터라서 np.array가 필요 없다. 즉, 나중에 np.array가 될 필요가 없다.  
print(y)


# Splitting the dataset into the Training set and Test set  :        <데이터셋을 훈련 세트와 테스트 세트로 나누기>
from sklearn.model_selection import train_test_split                                                # 훈련 세트에 특징 행렬과 종속 변수 벡터 한 쌍, 테스트 세트에 똑같은 한 쌍을 만들 것이다. model_selection의 접근권한을 얻고, train_test_split 함수를 불러왔다.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)        # 이 함수는 4개의 인자를 요구한다. 데이터 세트 전체를 요구하지 않고, 특징 행렬 x와 종속 변수 벡터 y의 조합을 요구한다. 
                                                                                                    # 인자 4개로는 (특징행렬, 종속변수벡터, 분할크기, 무작위 요소) 분할크기 권장값은 훈련 세트 80퍼센트 관측값, 테스트 세트 20퍼센트 관측값
print(x_train)
print(y_train)
print(x_test)
print(y_test)


# Feature Scaling    : 특징을 추출하기       ->  모든 특성을 동일한 크기로 조정해 주는 도구 : 일부 기계 모델에서 일부 특성이 다른 특성에 의해 지배되어 기계 모델에 인식되지 않은 상황을 방지하기위해
                                                # 모든 기계 모델에 F.S를 적용할 필요는 없다. 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])    #fit은 각 평균과 표준편차를 구하는 것이고, transform은 이 공식을 적용해서 표준화나 정규화를 하는 것이다.
x_test[:, 3:] = sc.fit_transform(x_test[:, 3:])


# 데이터셋을 훈련 세트와 테스트 세트로 나누고, 특징을 추출해야하는가. 아니면 그 반대인가?
# A => 데이터셋을 두 세트로 나누고 특징을 추출해야한다.
# 이유는 훈련이 끝날 때까지 가져오면 안되는 테스트 세트의 정보가 유출되는 것을 막기 위해서이다. (데이터세트를 훈련 세트와 테스트 세트로 분할하기 ~ 04:00)

#데이터 전처리를 할 때 변함없이 진행되는 부분이 있다.
# 1. 라이브러리를 임포팅하고
# 2. Dataset을 임포팅해서 특징행렬과 종속변수벡터를 만들고
# 3. 훈련 세트와 테스트 세트를 만드는 것
# 위 3과정은 이미 템플릿이 있을 정도로 변함없이 자주 사용하는 부분이다. 따라서 이미 있는 템플릿을 사용하면 빠르고 쉽게 데이터 전처리가 가능하다. 