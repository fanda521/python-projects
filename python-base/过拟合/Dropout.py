# ==============================
# 解决方案 1：Dropout（随机失活神经元）
# ==============================
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

X, y = make_classification(n_samples=200, n_features=20, n_informative=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()
model.add(Dense(256, activation='relu', input_dim=20))
model.add(Dropout(0.3))  # 随机丢弃30%神经元
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("===== 训练【Dropout 解决过拟合】=====")
model.fit(X_train, y_train, epochs=100, batch_size=4, validation_data=(X_test, y_test))

train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print("\n===== 解决方案 1：Dropout 结果 =====")
print(f"训练准确率: {train_acc:.4f}")
print(f"测试准确率: {test_acc:.4f}")