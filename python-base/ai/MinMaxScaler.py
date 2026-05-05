# 导入库
import numpy as np
from sklearn.preprocessing import MinMaxScaler  # 归一化
from sklearn.preprocessing import StandardScaler # 标准化

# 模拟原始数据：年龄、票价（数值差距极大）
data = np.array([
    [22, 7.25],
    [38, 71.28],
    [26, 7.92],
    [35, 53.10],
    [35, 8.05]
])

# --------------------------
# 1. 归一化 MinMaxScaler
# --------------------------
# 初始化归一化器
"""
feature_range=(0,1)  默认参数
把所有特征压缩到 0～1 之间
"""
mm_scaler = MinMaxScaler(feature_range=(0, 1))

# fit：学习当前数据的最大、最小值
mm_scaler.fit(data)

# transform：执行缩放
data_mm = mm_scaler.transform(data)

print("===== 归一化结果 [0,1] =====")
print(data_mm)

# --------------------------
# 2. 标准化 StandardScaler
# --------------------------
"""
会自动计算：均值μ、标准差σ
"""
std_scaler = StandardScaler()

# fit 统计均值、方差
std_scaler.fit(data)

# 标准化转换
data_std = std_scaler.transform(data)

print("\n===== 标准化结果(均值0 方差1) =====")
print(data_std)

# --------------------------
# 3. 关键配套方法（项目必用）
# --------------------------
# 反向还原为原始数据
data_origin = mm_scaler.inverse_transform(data_mm)
print("\n===== 还原原始数据 =====")
print(data_origin)