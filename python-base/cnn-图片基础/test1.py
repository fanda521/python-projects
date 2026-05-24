"""
卷积神经网络 CNN 入门（图像识别基础）
核心作用
专门处理图片数据，通过卷积核提取纹理、边缘、轮廓等空间特征
基础层介绍
Conv2D 卷积层：滑动窗口提取图像局部特征
MaxPooling2D 池化层：压缩图像尺寸，减少计算量，保留核心特征
Flatten 展平层：二维图像特征转为一维数据，接入全连接层
MNIST 手写数字识别极简实战
"""
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# 1.加载手写数字数据集 0-9共10分类
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2.维度调整+归一化，像素值缩至0-1
x_train = x_train.reshape(-1,28,28,1) / 255.0
x_test = x_test.reshape(-1,28,28,1) / 255.0

# 3.标签编码
y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)

# 4.搭建CNN网络
model = Sequential()
# 卷积层：32个3*3卷积核
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
# 池化层：2*2窗口压缩
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))

# 第二层卷积
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))

# 展平二维特征
model.add(Flatten())
# 全连接层
model.add(Dense(64,activation='relu'))
# 输出10个数字类别
model.add(Dense(10,activation='softmax'))

# 编译训练
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=8,batch_size=32,validation_split=0.1)

# 评估
score = model.evaluate(x_test,y_test)
print(f"手写数字识别准确率：{score[1]:.4f}")