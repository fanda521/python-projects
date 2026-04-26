# ==============================
# 1. 导入所有需要的库（每行解释）
# ==============================
import pandas as pd  # 数据处理库，用于读取、清洗、整理表格数据
from sklearn.model_selection import train_test_split  # 用于把数据分成 训练集 / 测试集
from sklearn.ensemble import RandomForestClassifier  # 随机森林分类模型（工业最常用）
from sklearn.metrics import classification_report    # 一键生成模型评估报告

# ==============================
# 2. 直接在代码里构造泰坦尼克号数据（不联网、不报错）
# ==============================
# 这是泰坦尼克号数据集的简化本地版本，直接运行
data = {
    'Pclass': [3, 1, 3, 1, 3, 1, 3, 3, 2, 3],
    'Sex': ['male', 'female', 'female', 'female', 'male', 'male', 'male', 'female', 'female', 'female'],
    'Age': [22.0, 38.0, 26.0, 35.0, 35.0, 40.0, 19.0, 28.0, 25.0, 30.0],
    'Fare': [7.2500, 71.2833, 7.9250, 53.1000, 8.0500, 50.0, 10.0, 12.0, 15.0, 20.0],
    'SibSp': [1, 1, 0, 1, 0, 0, 0, 0, 0, 1],
    'Survived': [0, 1, 1, 1, 0, 0, 0, 1, 1, 1]
}

# 把字典转成Pandas表格（DataFrame）
df = pd.DataFrame(data)

# ==============================
# 3. 数据预处理（AI必须步骤）
# ==============================
# 选择要用来训练的特征列（模型只能用数字，不能用文字）
feat_cols = ["Pclass", "Sex", "Age", "Fare", "SibSp"]

# 把性别文字转成数字：male→0，female→1（机器只能识别数字）
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# 处理年龄空值（如果有年龄为空，用平均年龄填上）
df["Age"] = df["Age"].fillna(df["Age"].mean())

# ==============================
# 4. 划分特征 X 和标签 y
# ==============================
X = df[feat_cols]   # X：输入特征（用来预测的信息）
y = df["Survived"]  # y：目标标签（我们要预测的结果：0=死亡，1=生存）

# ==============================
# 5. 拆分训练集 和 测试集
# ==============================
"""
train_test_split 参数详解：
X ：特征数据
y ：标签数据
test_size=0.2 ：20% 数据作为测试集，80% 作为训练集
random_state=42：固定随机种子，保证每次运行结果一样
stratify=y：保持训练集和测试集的类别比例一致（防止数据倾斜）
"""
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==============================
# 6. 创建随机森林模型
# ==============================
"""
RandomForestClassifier 参数详解：
n_estimators=80 ：森林里有80棵决策树（越多越稳，但速度变慢）
max_depth=6 ：限制树的深度，防止过拟合
random_state=42 ：固定随机数
n_jobs=-1 ：使用电脑全部CPU核心，加速训练
"""
model = RandomForestClassifier(
    n_estimators=80,
    max_depth=6,
    random_state=42,
    n_jobs=-1
)

# ==============================
# 7. 训练模型
# ==============================
# fit()：所有sklearn模型统一的训练方法
# 作用：让模型从【训练集】中学习规律
model.fit(X_train, y_train)

# ==============================
# 8. 预测 + 评估
# ==============================
# predict()：用训练好的模型预测测试集结果
y_pred = model.predict(X_test)

# 输出完整评估报告（准确率、精确率、召回率、F1）
print("=" * 50)
print("泰坦尼克号生存预测 - 模型评估报告")
print("=" * 50)
print(classification_report(y_test, y_pred))

# ==============================
# 9. 查看特征重要性（模型告诉我们：什么因素最影响生存）
# ==============================
print("\n特征重要性排序（影响生存的关键因素）：")
imp_df = pd.DataFrame({
    "特征": feat_cols,
    "重要性": model.feature_importances_
}).sort_values("重要性", ascending=False)

print(imp_df)