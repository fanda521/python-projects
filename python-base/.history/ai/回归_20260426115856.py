# 1. 导入库
from sklearn.linear_model import LinearRegression
import numpy as np

# 2. 构造训练数据
# 特征 x：学习时长(小时)
X = np.array([[1], [2], [3], [4], [5]])
# 标签 y：考试分数
y = np.array([60, 65, 75, 80, 95])

# 3. 创建模型 + 训练
model = LinearRegression()
model.fit(X, y)

# 4. 查看模型参数
print("斜率(权重)：", model.coef_)
print("截距：", model.intercept_)

# 5. 预测
predict_score = model.predict([[6]])
print("学习6小时，预测分数：", predict_score[0])