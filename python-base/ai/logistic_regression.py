# 1. 导包
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 2. 生成二分类模拟数据
X, y = make_classification(
    n_samples=200, n_features=5, random_state=42
)

# 3. 划分训练/测试
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. 模型 + 训练
model = LogisticRegression()
model.fit(X_train, y_train)

# 5. 评估
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print(f"训练集准确率：{train_acc:.2f}")
print(f"测试集准确率：{test_acc:.2f}")

# 6. 概率预测
y_pred_prob = model.predict_proba(X_test[:5])
print("\n前5个样本分类概率：")
print(y_pred_prob)



from sklearn.metrics import classification_report

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

