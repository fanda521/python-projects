"""
一、什么是神经元（仿生人脑）
生物神经元：接收信号 → 处理 → 输出信号人工神经元（感知机）：接收多个输入特征 → 加权求和 + 偏置 → 过激活函数 → 输出
公式：\(z = w_1x_1 + w_2x_2 + \dots + b\)\(\hat y = \sigma(z)\)

w：权重（重要程度）
b：偏置（基线偏移）
\(\sigma\)：激活函数（常用 Sigmoid、ReLU）

二、为什么要神经网络？

逻辑回归只能做线性分割
加激活函数、叠多层网络 → 能拟合任意非线性规律图像、语音、大模型 全靠深层神经网络。

三、最常用两个激活函数

Sigmoid\(\sigma(z) = \dfrac{1}{1+e^{-z}}\)输出 0~1，适合二分类输出概率

ReLU（深度学习最常用）\(ReLU(z) = \max(0, z)\)负数直接置 0，正数原样输出，训练收敛快、防梯度消失。

"""

# 导入numpy数值计算库
import numpy as np

# ======================
# 1. 定义激活函数
# ======================
def sigmoid(z):
    """
    Sigmoid激活函数
    作用：把任意实数压缩到 0~1 之间，输出概率
    """
    return 1 / (1 + np.exp(-z))

# ======================
# 2. 构造单层神经元类
# ======================
class SingleNeuron:
    # 初始化方法
    def __init__(self, input_dim):
        """
        input_dim：输入特征的个数
        初始化权重w、偏置b
        权重初始化为很小的随机数，偏置初始化为0
        """
        # 权重：(输入特征数, 1)
        self.w = np.random.randn(input_dim, 1) * 0.01
        # 偏置：标量
        self.b = 0.0

    # 前向传播：计算预测值
    def forward(self, X):
        """
        X：输入特征矩阵，形状 (样本数, 特征数)
        计算 z = X·w + b
        再经过sigmoid得到预测概率
        """
        # np.dot 矩阵点乘
        z = np.dot(X, self.w) + self.b
        # 过激活函数
        y_hat = sigmoid(z)
        return y_hat

    # 计算损失：均方误差
    def loss(self, y_hat, y_true):
        """
        y_hat：模型预测值
        y_true：真实标签
        均方误差损失 MSE
        """
        return np.mean((y_hat - y_true) ** 2)

    # 反向传播 + 梯度下降更新参数
    def backward(self, X, y_hat, y_true, lr):
        """
        反向传播求梯度 + 更新w和b
        lr：学习率，控制每一步更新幅度
        """
        # 样本数量
        m = X.shape[0]
        
        # 求梯度
        dz = (y_hat - y_true) * y_hat * (1 - y_hat)
        dw = (1 / m) * np.dot(X.T, dz)
        db = (1 / m) * np.sum(dz)
        
        # 梯度下降更新权重和偏置
        self.w = self.w - lr * dw
        self.b = self.b - lr * db

# ======================
# 3. 构造模拟二分类数据
# ======================
# 5条样本，每条2个特征
X = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6]
])
# 真实标签 0/1
y_true = np.array([[0], [0], [0], [1], [1]])

# ======================
# 4. 初始化神经元 + 训练
# ======================
# input_dim=2：每个样本2个特征
neuron = SingleNeuron(input_dim=2)

# 超参数
lr = 0.1        # 学习率
epochs = 1000   # 迭代训练轮数

# 循环训练
for i in range(epochs):
    # 前向传播，得到预测
    y_hat = neuron.forward(X)
    # 计算当前损失
    loss_val = neuron.loss(y_hat, y_true)
    # 反向传播更新参数
    neuron.backward(X, y_hat, y_true, lr)
    
    # 每200轮打印一次损失
    if i % 200 == 0:
        print(f"迭代轮数:{i:4d} | 损失值:{loss_val:.4f}")

# ======================
# 5. 训练完成后预测
# ======================
print("\n训练完成最终预测概率：")
print(neuron.forward(X))


"""
五、核心知识点总结
单个神经元 = 加权求和 + 偏置 + 激活函数
训练流程：
前向传播 → 计算损失 → 反向传播求梯度 → 梯度下降更新权重
激活函数是引入非线性的关键，没有激活函数，再多层也只是线性回归。
大模型、深度学习底层：无数个神经元层层堆叠。
"""