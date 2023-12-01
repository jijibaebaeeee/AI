# Convolutional Neural Network

# Importing the libraries
import numpy as np
import tensorflow as tf
#from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator   # Part 1 에서 이미지 전처리를 할 때 무조건 필요하다.

print(tf.__version__)

# Part 1 - Data Preprocessing

# Preprocessing the Training set
# 1. 훈련 세트에 있는 모든 이미지에만 변환을 적용 , 테스트 세트는 X  -> 과적합을 피하기 위해
train_datagen = ImageDataGenerator(
        rescale=1./255,    # 255로 나눠서 픽셀 하나 하나에 특징 스케일링을 적용할 것이다.  신경망에서 feature scaling은 필수다!
        shear_range=0.2,    # 어떠한 변환 하는 것
        zoom_range=0.2,     # 크기 변환
        horizontal_flip=True)   # 가로로 뒤집는 변환

# 디렉토리에서 훈련 세트에 접근해서 훈련 세트를 가지고오고, 동시에 이미지 배치를 만들고, 머신의 계산이 덜 힘들도록 계산을 줄이기 위해 크기를 재설정 함
training_set = train_datagen.flow_from_directory(
        'C:/Users/linha/Downloads/dataset/dataset/training_set',    # 데이터세트의 경로 지정
        target_size=(64, 64),    # 컨볼루션 신경망에 입력될 이미지 크기
        batch_size=32,           # 배치의 크기로 각 배치에 들어가는 이미지의 장수
        class_mode='binary')     # binary or categorical(셋 이상)

# Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale=1./255)  # 훈련 세트와 똑같은 변환을 하지 않는다
test_set = test_datagen.flow_from_directory(
        'C:/Users/linha/Downloads/dataset/dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

# Part 2 - Building the CNN

# Initialising the CNN
cnn = tf.keras.models.Sequential()    # Sequential()을 사용함으로 인공신경망을 일련의 계층들로 생성할 수 있게 한다.

# Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation='relu', input_shape=(64,64,3)))    # class이므로 ()가 들어간다. filters = 몇 개의 feature Detector가 필요한지, kernal_size = feature Detector의 크기(제곱전), activation = 활성함수
                                                # 첫 번째 계층을 추가할 때는 input_shape을 반드시 명시해야한다. input_shape=(64,64,3)   (target_size1, target_size2, rgb= 3/ BW = 1)
# Step 2 - Pooling (최대풀링)
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))    # pool_size = 정사각형 한 변 길이, strides = 몇 픽셀을 건너뛰어서 오른쪽으로 갈지, padding = valid : 예외처리 부분 무시 same : 0으로 대체

# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))
#최대 풀링이 적용된 두 번째 컨볼루션 계층 추가 완료

# Step 3 - Flattening (미래 신경망의 입력)
cnn.add(tf.keras.layers.Flatten())    #  파라미터 없음

# Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units = 128, activation = 'relu'))  #출력이 아니면 모든 활성화함수 relu

# Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units = 1, activation='sigmoid'))   # 강아지 = 0 , 고양이 = 1로     그림은 2개였지만

# Part 3 - Training the CNN

# Compiling the CNN (ANN과 동일)
#optimizer : 예측과 타겟 사이의 손실 오차를 줄이기 위해 확률적 경사하강법을 선택
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])   #2분형 예측 = binary_crossentropy   N분형 예측 = categorical_crossentropy

# Training the CNN on the Training set and evaluating it on the Test set
# 와 훈련과 평가를 동시에 한다! fit : 훈련을 위한 method
cnn.fit(x = training_set, validation_data = test_set, epochs = 25)    # x = 위에서 만든 훈련 세트, validation_data = 평가할 테스트 세트  epoch = 25면 10~15분

# Part 4 - Making a single prediction
from keras.preprocessing.image import image   #단일 이미지르 만들기 위해 사용
test_image =  image.load_img('C:/Users/linha/OneDrive/Desktop/Machine Learning A-Z (Codes and Datasets)/Part 8 - Deep Learning/sigle_predict1.jpg', target_size = (64, 64))   # 경로, 훈련에서 사용된 이미지 크기가 같아야함
test_image = image.img_to_array(test_image)    #predict 함수가 사진을 받아들이려면 PIL 형식으로 바꿔야함. PIL = 이미지를 배열에 놓는 형식
test_image = np.expand_dims(test_image, axis = 0)    # 훈련세트를 진행할 때, 우리는 배치를 만들었다.(차원추가), 여기에서도 배치를 만들어주어야 형식이 맞다.
result = cnn.predict(test_image)    # result는 단순히 0 or 1 만을 결과로 가진다. 따라서 우리가 1 = 강아지, 0 = 고양이 라는 것을 알려줘야한다.
training_set.class_indices
if result[0][0]:   #배치의 첫 번째이자 유일한 요소  두번째 0 = 인덱스 0의 배치에서 단일 예측(이미지가 하나이므로)
    prediction = "dog"
else:
    prediction = "cat"

print(prediction)