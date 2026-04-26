# --------------------------
# 1. 导入需要的工具包
# --------------------------
# 导入决策树分类器
from sklearn.tree import DecisionTreeClassifier
# 导入数据集分割工具：拆分训练集、测试集
from sklearn.model_selection import train_test_split
# 导入官方测试数据集：红酒多分类数据
from sklearn.datasets import load_wine

# --------------------------
# 2. 加载数据集
# --------------------------
# load_wine()：加载红酒数据集，经典多分类数据集
# 包含13个特征，3种红酒分类标签
wine = load_wine()

# X：所有输入特征（二维数组，模型学习的原材料）
X = wine.data
# y：真实标签（0/1/2 三种红酒类别）
y = wine.target

# --------------------------
# 3. 划分 训练集 / 测试集
# --------------------------
"""
train_test_split 参数详解：
X：特征数据
y：标签数据
test_size=0.2：20%数据当做测试集，80%训练集
random_state=42：固定随机种子
   - 不设置每次拆分数据不一样
   - 设置后结果固定，方便学习、复现代码
"""
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# --------------------------
# 4. 初始化决策树模型 + 参数详解
# --------------------------
"""
DecisionTreeClassifier 关键参数：
criterion="gini"：划分依据，默认基尼系数
   - gini：基尼系数，计算简单，默认
   - entropy：信息熵，计算慢一点，划分更精细
max_depth=None：树的最大深度
   - None：不限深度，容易过拟合
   - 实际项目一般手动限制，比如 max_depth=5 防过拟合
random_state=42：固定模型随机行为，结果可复现
"""
dt_model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=None,
    random_state=42
)

# --------------------------
# 5. 模型训练（核心方法）
# --------------------------
"""
fit()：sklearn 所有模型统一训练方法
传入：训练集特征、训练集真实标签
作用：让模型从数据中学习规则、划分边界
"""
dt_model.fit(X_train, y_train)

# --------------------------
# 6. 模型评估
# --------------------------
"""
score()：自动计算 准确率 accuracy
传入：测试集特征、测试集真实标签
返回：0~1 之间的小数，越高越准
"""
train_acc = dt_model.score(X_train, y_train)  # 训练集得分
test_acc = dt_model.score(X_test, y_test)    # 测试集得分

print(f"决策树-训练集准确率：{train_acc:.2f}")
print(f"决策树-测试集准确率：{test_acc:.2f}")

# --------------------------
# 7. 单样本预测
# --------------------------
"""
predict()：用训练好的模型做预测
传入：二维数组格式的特征
返回：预测的分类标签
"""
# 取测试集第一条数据
sample = X_test[:1]
# 模型预测
pred_label = dt_model.predict(sample)
# 真实标签
true_label = y_test[:1]

print("\n单样本预测：")
print("预测类别：", pred_label)
print("真实类别：", true_label)