# 多层神经网络（隐藏层）原理与实战
# 这份代码演示机器学习到深度学习的过渡：
# 1) 为什么需要隐藏层
# 2) Sklearn 中的 MLPClassifier 快速搭建
# 3) Keras 中的简单神经网络分类训练

# --------------------------
# 0. 先解释什么是“多层神经网络”
# --------------------------
"""
多层神经网络（Multilayer Perceptron, MLP）是多层感知机的简称。
它不仅包含输入层和输出层，还增加了一个或多个隐藏层。
隐藏层中的每个神经元都会进行线性变换 -> 激活函数 -> 输出。
引入隐藏层后，模型获得了“非线性表达能力”，能够拟合更复杂的分类边界。

与简单的逻辑回归相比：
- 逻辑回归只有一个线性层 + sigmoid
- MLP 可以叠加多个线性层 + 激活函数

隐藏层的核心作用：
- 通过神经元组合学习特征变换
- 使得网络可以近似任意复杂函数
- 激活函数（如 ReLU）提供非线性

训练过程简要：
1. 前向传播：输入经过每一层，计算输出
2. 计算损失：比较输出与真实标签
3. 反向传播：梯度传回每一层，更新权重
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# 这个例子需要 sklearn
from sklearn.neural_network import MLPClassifier

# 用于 Keras 示例
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.optimizers import Adam
    keras_available = True
except ImportError:
    keras_available = False

# --------------------------
# 1. 生成模拟二分类数据
# --------------------------
"""
我们仍然使用 make_classification 生成一个简单的分类数据集。
特征数量不需要太多，方便观察隐藏层的作用。
"""
X, y = make_classification(
    n_samples=300,
    n_features=5,
    n_informative=3,
    n_redundant=1,
    n_classes=2,
    random_state=42
)

# --------------------------
# 2. 划分训练集和测试集
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --------------------------
# 3. 标准化
# --------------------------
"""
神经网络对特征尺度敏感，标准化有助于训练更稳定。
"""
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------
# 4. Sklearn MLPClassifier 快速训练
# --------------------------
"""
MLPClassifier 是 sklearn 中的多层感知机实现。
hidden_layer_sizes=(10,) 表示一个隐藏层，10 个神经元。
activation='relu' 使用 ReLU 作为激活函数。
solver='adam' 使用 Adam 优化器。
max_iter=1000 保证迭代次数足够。
"""
mlp = MLPClassifier(
    hidden_layer_sizes=(10,),
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42
)

mlp.fit(X_train_scaled, y_train)

# 预测与评估
y_pred = mlp.predict(X_test_scaled)

print('===== Sklearn MLPClassifier 分类结果 =====')
print('测试集准确率：', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

print('隐藏层权重形状：', [coef.shape for coef in mlp.coefs_])
print('隐藏层偏置形状：', [bias.shape for bias in mlp.intercepts_])

# --------------------------
# 5. Keras 简单神经网络搭建
# --------------------------
if keras_available:
    print('\n===== Keras 简单神经网络分类 =====')
    model = Sequential([
        Dense(10, activation='relu', input_dim=X_train_scaled.shape[1]),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.01),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # 查看模型结构
    model.summary()

    # 训练模型
    history = model.fit(
        X_train_scaled,
        y_train,
        epochs=50,
        batch_size=16,
        validation_split=0.2,
        verbose=0
    )

    # 评估
    loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print('测试集损失：', loss)
    print('测试集准确率：', accuracy)

    # 预测概率与类别
    y_prob = model.predict(X_test_scaled[:5])
    y_class = (y_prob >= 0.5).astype(int).flatten()
    print('\n前5条样本预测概率：')
    print(y_prob.flatten())
    print('前5条样本预测类别：', y_class)

    # 查看第一个隐藏层的权重和偏置
    weights, biases = model.layers[0].get_weights()
    print('\n第一个隐藏层权重形状：', weights.shape)
    print('第一个隐藏层偏置形状：', biases.shape)
else:
    print('\n===== Keras 未安装，跳过 Keras 示例 =====')
    print('如果想运行 Keras 部分，请先安装 TensorFlow: pip install tensorflow')

# --------------------------
# 6. 深度学习过渡总结
# --------------------------
"""
这里的实践体现了从机器学习到深度学习的核心过渡：
- 逻辑回归 / 线性模型：只能学习线性边界
- MLP（隐藏层）：引入非线性，能够学习更复杂的决策面
- Keras：从模型构建、编译、训练到评估，流程与传统机器学习类似，但更灵活

下一步可以继续学习：
- 增加隐藏层深度和宽度
- 使用不同激活函数（ReLU、tanh、sigmoid）
- 使用 dropout、batch normalization 等深度学习技巧
- 用真实数据集（如 Iris、MNIST）验证模型
"""
