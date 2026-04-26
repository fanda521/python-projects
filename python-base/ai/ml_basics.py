# 机器学习基础教程：AI核心概念 + Scikit-learn 入门实战
# 这个文件涵盖AI基础逻辑、Scikit-learn库的使用，以及分类、回归、聚类的实战示例

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, mean_squared_error, silhouette_score
from sklearn.datasets import make_classification, make_regression, make_blobs

print("=== 机器学习基础教程：AI核心概念 + Scikit-learn 入门 ===")

# 1. AI基础逻辑简介
print("\n1. AI基础逻辑简介:")
print("""
人工智能 (AI) 是让机器模拟人类智能的技术。
机器学习 (ML) 是AI的核心方法，通过数据学习模式，而非显式编程。

主要学习类型：
- 监督学习 (Supervised Learning): 有标签数据，学习输入→输出的映射
  * 分类 (Classification): 输出是离散类别 (如猫/狗)
  * 回归 (Regression): 输出是连续数值 (如房价预测)

- 无监督学习 (Unsupervised Learning): 无标签数据，发现数据中的模式
  * 聚类 (Clustering): 将相似数据分组

- 强化学习 (Reinforcement Learning): 通过试错学习最优策略

Scikit-learn 是Python最流行的机器学习库，提供：
- 数据预处理
- 模型训练和预测
- 模型评估
- 多种算法实现
""")

# 2. 数据准备和探索
print("\n2. 数据准备和探索:")

# 创建示例数据集
print("创建示例数据集...")

# 回归数据集 (房价预测)
X_reg, y_reg = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
print(f"回归数据集: {X_reg.shape[0]} 样本, {X_reg.shape[1]} 特征")

# 分类数据集 (二分类问题)
X_clf, y_clf = make_classification(n_samples=100, n_features=2, n_classes=2, n_redundant=0, random_state=42)
print(f"分类数据集: {X_clf.shape[0]} 样本, {X_clf.shape[1]} 特征, {len(np.unique(y_clf))} 类")

# 聚类数据集
X_clu, _ = make_blobs(n_samples=100, centers=3, cluster_std=1.0, random_state=42)
print(f"聚类数据集: {X_clu.shape[0]} 样本, {X_clu.shape[1]} 特征")

# 3. 回归实战 (Linear Regression)
print("\n3. 回归实战 - 线性回归:")

# 划分训练集和测试集
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# 创建和训练模型
reg_model = LinearRegression()
reg_model.fit(X_train_reg, y_train_reg)

# 预测
y_pred_reg = reg_model.predict(X_test_reg)

# 评估
mse = mean_squared_error(y_test_reg, y_pred_reg)
print(".2f")
print(".2f")

# 可视化
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.scatter(X_reg, y_reg, alpha=0.6, label='Actual data')
plt.plot(X_reg, reg_model.predict(X_reg), color='red', label='Prediction line')
plt.title('Linear Regression')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.legend()

# 4. 分类实战 (Logistic Regression & Decision Tree)
print("\n4. 分类实战 - 逻辑回归和决策树:")

# 划分训练集和测试集
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

# 逻辑回归
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_clf, y_train_clf)
y_pred_log = log_reg.predict(X_test_clf)
acc_log = accuracy_score(y_test_clf, y_pred_log)
print(".2f")

# 决策树
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train_clf, y_train_clf)
y_pred_tree = tree_clf.predict(X_test_clf)
acc_tree = accuracy_score(y_test_clf, y_pred_tree)
print(".2f")

# 可视化分类结果
plt.subplot(1, 3, 2)
x_min, x_max = X_clf[:, 0].min() - 1, X_clf[:, 0].max() + 1
y_min, y_max = X_clf[:, 1].min() - 1, X_clf[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

Z_log = log_reg.predict(np.c_[xx.ravel(), yy.ravel()])
Z_log = Z_log.reshape(xx.shape)

plt.contourf(xx, yy, Z_log, alpha=0.3)
plt.scatter(X_clf[:, 0], X_clf[:, 1], c=y_clf, edgecolors='k', marker='o')
plt.title('Logistic Regression Classification')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# 5. 聚类实战 (K-Means)
print("\n5. 聚类实战 - K-Means:")

# 训练K-Means模型
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_clu)
y_pred_clu = kmeans.labels_

# 评估聚类质量
sil_score = silhouette_score(X_clu, y_pred_clu)
print(".2f")

# 可视化聚类结果
plt.subplot(1, 3, 3)
plt.scatter(X_clu[:, 0], X_clu[:, 1], c=y_pred_clu, cmap='viridis', edgecolors='k', marker='o')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='X', label='Cluster centers')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

plt.tight_layout()
plt.savefig('ml_plots.png')
print("Visualization plots saved as ml_plots.png")

# 6. 总结和下一步
print("\n6. 总结和下一步:")
print("""
恭喜！你已经掌握了机器学习的基础概念和Scikit-learn的使用：

✅ AI基础逻辑：理解了监督学习、无监督学习的核心概念
✅ Scikit-learn掌握：学会了数据划分、模型训练、预测和评估
✅ 分类实战：使用逻辑回归和决策树解决分类问题
✅ 回归实战：使用线性回归进行连续值预测
✅ 聚类实战：使用K-Means发现数据中的自然分组

下一步学习建议：
1. 深入学习更多算法：随机森林、支持向量机、神经网络
2. 掌握数据预处理：特征工程、数据标准化、处理缺失值
3. 学习模型调优：交叉验证、网格搜索、超参数优化
4. 实践真实数据集：Kaggle竞赛、UCI机器学习库
5. 探索深度学习：TensorFlow、PyTorch入门

记住：机器学习是一门实践性学科，多动手实验才能真正掌握！
""")