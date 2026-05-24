#多分类神经网络实战
#适配 3 类及以上分类任务，更换输出层激活函数与损失函数

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# 1.生成3分类数据集
X, y = make_blobs(n_samples=400, n_features=8, centers=3, random_state=42)

# 2.划分数据集
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# 3.数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4.标签转为独热编码，适配多分类
from tensorflow.keras.utils import to_categorical
y_train_onehot = to_categorical(y_train)
y_test_onehot = to_categorical(y_test)

# 5.搭建带防过拟合的网络
model = Sequential()
# 输入+隐藏层1
model.add(Dense(32, activation='relu', input_dim=8))
model.add(Dropout(0.2))
# 隐藏层2
model.add(Dense(16, activation='relu'))
# 输出层：3个神经元对应3个类别，Softmax输出各类概率
model.add(Dense(3, activation='softmax'))

# 6.编译模型
# 多分类损失用交叉熵，优化器依旧Adam
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 7.训练模型
history = model.fit(
    X_train, y_train_onehot,
    epochs=60,
    batch_size=10,
    validation_split=0.1
)

# 8.测试评估
test_loss, test_acc = model.evaluate(X_test, y_test_onehot)
print(f"多分类测试准确率：{test_acc:.4f}")

# 9.样本预测
pred_prob = model.predict(X_test[:1])
pred_label = np.argmax(pred_prob)
print(f"预测类别：{pred_label}，真实类别：{y_test[0]}")
#关键参数说明
#activation='softmax'：多分类专属激活，输出概率总和为 1
#to_categorical：数字标签转独热编码，模型可识别分类维度
#categorical_crossentropy：多分类标准损失函数