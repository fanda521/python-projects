# 导入
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 1. 加载官方数据集
iris = load_iris()
X = iris.data   # 特征
y = iris.target # 标签

# 2. 划分训练集 / 测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

# 3. 创建模型 + 训练
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# 4. 模型评估（准确率）
acc = knn.score(X_test, y_test)
print(f"模型准确率：{acc:.2f}")