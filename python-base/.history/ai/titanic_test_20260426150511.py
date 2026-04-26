# 导入库
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# ======================
# 1. 读取官方数据集（内置，无需下载）
# ======================
# 加载泰坦尼克号数据集 (使用seaborn内置数据集，避免网络问题)
df = sns.load_dataset('titanic')

# 如果上面的方法失败，可以尝试备用URL：
# url = "https://raw.githubusercontent.com/pandas-dev/pandas/master/doc/data/titanic.csv"
# df = pd.read_csv(url)

# ======================
# 2. 数据预处理（AI建模最重要环节）
# ======================
# 筛选有用特征，删掉无用字段
# Pclass:船舱等级  Sex:性别  Age:年龄  Fare:票价  SibSp:兄弟姐妹/配偶数
feat_cols = ["Pclass", "Sex", "Age", "Fare", "SibSp"]
df = df[feat_cols + ["Survived"]]  # 拼接特征+标签

# 处理缺失值：Age年龄存在空值
# fillna()：填充缺失值，这里用平均值填充
df["Age"] = df["Age"].fillna(df["Age"].mean())

# 文本特征数字化：Sex原本是male/female，模型只能读数字
# map映射：male→0  female→1
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# ======================
# 3. 划分特征X 和 标签y
# ======================
X = df[feat_cols]   # 全部输入特征
y = df["Survived"]  # 预测目标：是否生还

# ======================
# 4. 拆分训练集、测试集
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,        # 20%做测试，模拟真实未知数据
    random_state=42,      # 固定随机拆分
    stratify=y            # 分层采样，保证生还/死亡比例一致
)

# ======================
# 5. 搭建随机森林模型
# ======================
model = RandomForestClassifier(
    n_estimators=80,      # 80棵决策树集成
    max_depth=6,          # 限制树深度，主动防止过拟合
    random_state=42,
    n_jobs=-1
)

# ======================
# 6. 训练 + 预测 + 评估
# ======================
# fit：模型训练，学习特征与生还的规律
model.fit(X_train, y_train)

# 测试集预测
y_pred = model.predict(X_test)

# 打印完整评估报告
print("===== 泰坦尼克号 生还预测 模型报告 =====")
print(classification_report(y_test, y_pred))

# 输出每个特征的重要程度
print("\n===== 特征重要性 =====")
feature_import = pd.DataFrame({
    "特征": feat_cols,
    "重要性": model.feature_importances_
}).sort_values("重要性", ascending=False)

print(feature_import)