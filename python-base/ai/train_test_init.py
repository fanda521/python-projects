from sklearn.model_selection import train_test_split
import numpy as np

# 假设 X 特征，y 标签
X = np.random.rand(100, 2)
y = np.random.randint(0, 2, 100)

# test_size=0.2 ：20%当测试集，80%训练集
# random_state 固定随机种子，结果可复现
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("训练集大小：", X_train.shape)
print("测试集大小：", X_test.shape)