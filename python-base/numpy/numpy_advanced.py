# NumPy高级用法示例
# 这个文件演示了NumPy的高级功能，包括广播、线性代数、文件I/O等

import numpy as np

print("=== NumPy高级用法 ===")

# 1. 广播 (Broadcasting)
print("\n1. 广播机制:")
a = np.array([1, 2, 3])
b = np.array([[1], [2], [3]])
print(f"a.shape = {a.shape}, b.shape = {b.shape}")
print(f"a + b =\n{a + b}")  # 广播相加

# 2. 数组拼接和分割
print("\n2. 数组拼接和分割:")
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

# 拼接
concatenated = np.concatenate([arr1, arr2])
print(f"拼接: {concatenated}")

# 垂直堆叠
vstacked = np.vstack([arr1, arr2])
print(f"垂直堆叠:\n{vstacked}")

# 水平堆叠
hstacked = np.hstack([arr1, arr2])
print(f"水平堆叠: {hstacked}")

# 分割
split_arr = np.array_split(concatenated, 3)
print(f"分割成3部分: {split_arr}")

# 3. 线性代数
print("\n3. 线性代数:")
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(f"A =\n{A}")
print(f"B =\n{B}")

# 矩阵乘法
print(f"A @ B =\n{A @ B}")  # 或 np.dot(A, B)

# 转置
print(f"A.T =\n{A.T}")

# 行列式
det_A = np.linalg.det(A)
print(f"det(A) = {det_A}")

# 逆矩阵
inv_A = np.linalg.inv(A)
print(f"A^(-1) =\n{inv_A}")

# 特征值和特征向量
eigenvals, eigenvecs = np.linalg.eig(A)
print(f"特征值: {eigenvals}")
print(f"特征向量:\n{eigenvecs}")

# 4. 统计函数
print("\n4. 统计函数:")
data = np.random.randn(1000)  # 生成1000个标准正态分布随机数
print(f"数据样本: {data[:10]}...")  # 只显示前10个
print(f"均值: {np.mean(data):.4f}")
print(f"中位数: {np.median(data):.4f}")
print(f"方差: {np.var(data):.4f}")
print(f"标准差: {np.std(data):.4f}")
print(f"最小值: {np.min(data):.4f}")
print(f"最大值: {np.max(data):.4f}")
print(f"25%分位数: {np.percentile(data, 25):.4f}")
print(f"75%分位数: {np.percentile(data, 75):.4f}")

# 5. 数组排序和搜索
print("\n5. 数组排序和搜索:")
unsorted = np.array([3, 1, 4, 1, 5, 9, 2, 6])
print(f"未排序数组: {unsorted}")
sorted_arr = np.sort(unsorted)
print(f"排序后: {sorted_arr}")

# 搜索
search_val = 4
indices = np.where(unsorted == search_val)
print(f"值{search_val}的位置: {indices[0]}")

# 6. 文件I/O
print("\n6. 文件I/O:")
# 保存数组到文件
arr_to_save = np.array([[1, 2, 3], [4, 5, 6]])
np.savetxt('sample_array.txt', arr_to_save, delimiter=',')
print("数组已保存到 sample_array.txt")

# 从文件加载数组
loaded_arr = np.loadtxt('sample_array.txt', delimiter=',')
print(f"从文件加载的数组:\n{loaded_arr}")

# 7. 高级索引
print("\n7. 高级索引:")
arr = np.arange(12).reshape(3, 4)
print(f"原始数组:\n{arr}")

# 花式索引
rows = np.array([0, 2])
cols = np.array([1, 3])
print(f"花式索引 arr[{rows}, {cols}] = {arr[rows, cols]}")

# 布尔索引
mask = arr > 5
print(f"布尔掩码:\n{mask}")
print(f"筛选结果: {arr[mask]}")

# 8. 数组操作的高级技巧
print("\n8. 数组操作技巧:")

# 使用 np.where 进行条件操作
x = np.array([1, 2, 3, 4, 5])
result = np.where(x > 3, '大', '小')
print(f"条件操作: {result}")

# 数组展平
nested = np.array([[1, 2], [3, 4]])
flattened = nested.flatten()
print(f"展平: {flattened}")

# 唯一值
with_duplicates = np.array([1, 2, 2, 3, 3, 3])
unique_vals = np.unique(with_duplicates)
print(f"唯一值: {unique_vals}")

print("\n=== NumPy高级用法完成 ===")