import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 生成小数据集（方便你看清数值）
# 生成 6 个样本，2 个特征，2分类
X, y = make_classification(
    n_samples=6,    # 6个样本
    n_features=2,   # 每个样本2个特征
    n_informative=2,
    n_redundant=0,
    random_state=42 # 固定随机数，数值可复现
)

print("===== 原始数据 X（特征）=====")
print(X)
print("X 的形状：", X.shape)
print("\n===== 原始标签 y（0/1）=====")
print(y)
print("y 的形状：", y.shape)

# 2. 划分训练集 + 测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,  # 30% 做测试集 → 6个样本 → 4个训练，2个测试
    random_state=42
)

print("\n===== X_train（训练集特征）=====")
print(X_train)
print("形状：", X_train.shape)

print("\n===== X_test（测试集特征）=====")
print(X_test)
print("形状：", X_test.shape)

print("\n===== y_train（训练集标签）=====")
print(y_train)
print("形状：", y_train.shape)

print("\n===== y_test（测试集标签）=====")
print(y_test)
print("形状：", y_test.shape)