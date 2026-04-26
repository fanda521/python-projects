# --------------------------
# 1. 导入依赖
# --------------------------
# 随机森林分类器
from sklearn.ensemble import RandomForestClassifier
# 数据拆分
from sklearn.model_selection import train_test_split
# 红酒数据集
from sklearn.datasets import load_wine

# --------------------------
# 2. 加载数据
# --------------------------
wine = load_wine()
X = wine.data   # 特征
y = wine.target # 标签

# --------------------------
# 3. 划分训练/测试集
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,      # 测试集占比20%
    random_state=42     # 固定随机种子，结果不变
)

# --------------------------
# 4. 初始化随机森林 【超详细参数】
# --------------------------
"""
RandomForestClassifier 核心参数：
n_estimators=100：森林里有多少棵决策树，默认100棵
   - 树越多效果越好，算力消耗越大
criterion="gini"：节点划分标准，同决策树
max_depth=None：每棵树最大深度
random_state=42：全局随机种子固定
n_jobs=-1：使用电脑全部CPU核心并行训练，加速运行
"""
rf_model = RandomForestClassifier(
    n_estimators=100,
    criterion="gini",
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

# --------------------------
# 5. 模型训练
# --------------------------
# fit：喂训练数据，让多棵决策树各自学习
rf_model.fit(X_train, y_train)

# --------------------------
# 6. 模型打分评估
# --------------------------
train_acc = rf_model.score(X_train, y_train)
test_acc = rf_model.score(X_test, y_test)

print("===== 随机森林 结果 =====")
print(f"训练集准确率：{train_acc:.2f}")
print(f"测试集准确率：{test_acc:.2f}")

# --------------------------
# 7. 特征重要性（随机森林独家功能）
# --------------------------
"""
feature_importances_
作用：自动算出每一列特征对结果的影响权重
AI特征筛选、数据分析非常常用
"""
feature_importance = rf_model.feature_importances_
print("\n各特征重要性权重：")
print(feature_importance)

# --------------------------
# 8. 批量预测
# --------------------------
# 预测前5条测试数据
y_pred = rf_model.predict(X_test[:5])
print("\n前5条数据预测结果：")
print(y_pred)
print("前5条数据真实结果：")
print(y_test[:5])