# ==============================
# 【原版】纯过拟合模型
# 训练精度极高，测试精度很低 = 典型过拟合
# ==============================
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 造数据
X, y = make_classification(n_samples=200, n_features=20, n_informative=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. 搭建【超级大、超级复杂】神经网络 → 故意过拟合
model = Sequential()
model.add(Dense(256, activation='relu', input_dim=20))  # 神经元超多
model.add(Dense(256, activation='relu'))                 # 层数很深
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 4. 编译
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 5. 训练（训练久一点，故意过拟合）
print("===== 训练【过拟合模型】=====")
model.fit(X_train, y_train, epochs=100, batch_size=4, validation_data=(X_test, y_test))

# 6. 最终结果
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print("\n===== 过拟合结果（原版）=====")
print(f"训练准确率: {train_acc:.4f}")
print(f"测试准确率: {test_acc:.4f}")