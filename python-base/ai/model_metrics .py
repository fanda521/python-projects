# 导入工具包
from sklearn.ensemble import RandomForestClassifier  # 随机森林模型
from sklearn.datasets import load_wine                # 红酒数据集
from sklearn.model_selection import train_test_split  # 划分训练/测试集
# 导入评估指标工具
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score, confusion_matrix, classification_report

# 1. 加载数据
wine = load_wine()
X = wine.data   # 特征矩阵
y = wine.target # 分类标签

# 2. 数据集划分
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,       # 测试集比例 20%
    random_state=42,     # 固定随机种子，结果可复现
    stratify=y           # 分层抽样：保证训练/测试集类别比例一致，防止数据倾斜
)

# 3. 初始化随机森林模型
rf = RandomForestClassifier(
    n_estimators=100,    # 集成100棵决策树
    random_state=42,     # 固定随机化操作
    n_jobs=-1            # 调用全部CPU核心，加速训练
)

# 4. 模型训练
rf.fit(X_train, y_train)

# 5. 测试集预测
y_pred = rf.predict(X_test)

# ========== 逐个指标详解 ==========
# 1）准确率：全部样本中，预测正确的比例
acc = accuracy_score(y_test, y_pred)

# 2）精确率：预测为某类的样本里，真实是该类的比例
# average='macro'：多分类平均计算，不考虑类别样本数量权重
precision = precision_score(y_test, y_pred, average="macro")

# 3）召回率：真实为某类的样本里，被成功预测出来的比例
recall = recall_score(y_test, y_pred, average="macro")

# 4）F1分数：精确率和召回率的调和平均，综合衡量模型
f1 = f1_score(y_test, y_pred, average="macro")

# 6. 打印指标
print("===== 模型评估指标 =====")
print(f"准确率Accuracy: {acc:.2f}")
print(f"精确率Precision: {precision:.2f}")
print(f"召回率Recall: {recall:.2f}")
print(f"F1分数: {f1:.2f}")

# 7. 混淆矩阵：直观看到哪一类容易分错
cm = confusion_matrix(y_test, y_pred)
print("\n===== 混淆矩阵 =====")
print(cm)

# 8. 一键生成完整评估报告（面试/项目最常用）
print("\n===== 完整分类报告 =====")
print(classification_report(y_test, y_pred))