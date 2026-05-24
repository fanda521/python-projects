# ==============================
# 解决方案 3：早停法（不再训练到100轮）
# ==============================
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

X, y = make_classification(n_samples=200, n_features=20, n_informative=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()
model.add(Dense(256, activation='relu', input_dim=20))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 早停：验证损失5轮不下降就停止
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

print("===== 训练【早停法 解决过拟合】=====")
model.fit(X_train, y_train, epochs=100, batch_size=4, validation_data=(X_test, y_test), callbacks=[early_stop])

train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print("\n===== 解决方案 3：早停法 结果 =====")
print(f"训练准确率: {train_acc:.4f}")
print(f"测试准确率: {test_acc:.4f}")