import numpy as np

# 目标函数：y = 2x  模拟真实规律
# 我们要用梯度下降，一步步算出权重 w 趋近于 2

# 1. 超参数
lr = 0.01       # 学习率：每一步更新的幅度
epochs = 50     # 迭代次数：循环多少轮
w = 0.0         # 初始权重（随便给0）

# 训练数据
x = np.array([1, 2, 3, 4, 5])
y_true = np.array([2, 4, 6, 8, 10])

# 2. 梯度下降迭代训练
for i in range(epochs):
    # 预测值
    y_pred = w * x
    
    # 损失函数：均方误差 MSE
    loss = np.mean((y_pred - y_true) ** 2)
    
    # 梯度计算
    gradient = np.mean(2 * (y_pred - y_true) * x)
    
    # 权重更新：梯度下降核心公式
    w = w - lr * gradient
    
    # 每10轮打印一次
    if i % 10 == 0:
        print(f"迭代轮数:{i:2d} | 权重w:{w:.4f} | 损失loss:{loss:.4f}")

print("\n最终训练得到权重 w =", round(w, 2))