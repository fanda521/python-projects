


# 导入库
import numpy as np
# 逻辑回归模型
from sklearn.linear_model import LogisticRegression
# 数据集划分
from sklearn.model_selection import train_test_split
# 生成模拟分类数据
from sklearn.datasets import make_classification
# 模型评估报告
from sklearn.metrics import classification_report
# 标准化工具
from sklearn.preprocessing import StandardScaler

# --------------------------
# 1. 生成二分类模拟数据
# --------------------------
"""
make_classification 参数解释：
n_samples=200：生成200条样本
n_features=5：每条样本5个特征
n_informative=3：其中3个是有效特征，2个冗余
random_state=42：固定随机种子，每次数据一样
"""
X, y = make_classification(
    n_samples=200,
    n_features=5,
    n_informative=3,
    random_state=42
)

# --------------------------
# 2. 划分训练集、测试集
# --------------------------
"""
test_size=0.2：20%做测试集
random_state=42：固定拆分结果
stratify=y：分层抽样，保证训练/测试正负样本比例一致
"""
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --------------------------
# 3. 标准化（逻辑回归必须做）
# --------------------------
"""
StandardScaler：标准化
fit：在训练集上学习均值、方差
transform：执行标准化
"""
scaler = StandardScaler()
# 只用训练集拟合，避免泄露测试集信息
X_train_scaled = scaler.fit_transform(X_train)
# 测试集只做转换，不重新拟合
X_test_scaled = scaler.transform(X_test)

# --------------------------
# 4. 初始化逻辑回归模型
# --------------------------
"""
LogisticRegression 参数：
random_state=42：固定随机
max_iter=1000：最大迭代次数
   逻辑回归默认迭代步数不够，经常不收敛，要手动加大
"""
log_model = LogisticRegression(
    random_state=42,
    max_iter=1000
)

# --------------------------
# 5. 模型训练
# --------------------------
# fit：送入标准化后的训练数据，梯度下降优化w和b
log_model.fit(X_train_scaled, y_train)

# --------------------------
# 6. 预测与评估
# --------------------------
# 对测试集做类别预测 0/1
y_pred = log_model.predict(X_test_scaled)

# 输出完整评估报告
print("===== 逻辑回归 分类评估报告 =====")
print(classification_report(y_test, y_pred))

# --------------------------
# 7. 查看模型学到的权重和偏置
# --------------------------
print("\n模型权重 w：")
print(log_model.coef_)
print("\n模型偏置 b：")
print(log_model.intercept_)

# --------------------------
# 8. 预测概率（输出属于0、1的概率）
# --------------------------
# predict_proba：每行 [类别0概率, 类别1概率]
y_pred_prob = log_model.predict_proba(X_test_scaled[:5])
print("\n前5条样本预测概率：")
print(y_pred_prob)