# NumPy基础教程示例
# 这个文件演示了NumPy的基本用法，包括数组创建、操作和常用函数

import numpy as np

print("=== NumPy基础教程 ===")

# 1. 数组创建
print("\n1. 数组创建:")

# 从列表创建数组
arr1 = np.array([1, 2, 3, 4, 5])
print(f"从列表创建: {arr1}")

# 创建全零数组
zeros = np.zeros(5)
print(f"全零数组: {zeros}")

# 创建全一数组
ones = np.ones(5)
print(f"全一数组: {ones}")

# 创建指定范围的数组
range_arr = np.arange(0, 10, 2)  # 从0到10，步长2
print(f"范围数组: {range_arr}")

# 创建等间隔数组
linspace_arr = np.linspace(0, 1, 5)  # 从0到1，5个点
print(f"等间隔数组: {linspace_arr}")

# 2. 数组属性
print("\n2. 数组属性:")
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
print(f"二维数组:\n{arr2d}")
print(f"形状 (shape): {arr2d.shape}")
print(f"维度 (ndim): {arr2d.ndim}")
print(f"大小 (size): {arr2d.size}")
print(f"数据类型 (dtype): {arr2d.dtype}")

# 3. 数组操作
print("\n3. 数组操作:")

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(f"a = {a}")
print(f"b = {b}")
print(f"a + b = {a + b}")  # 元素-wise 加法
print(f"a * b = {a * b}")  # 元素-wise 乘法
print(f"a ** 2 = {a ** 2}")  # 平方

# 4. 数组索引和切片
print("\n4. 数组索引和切片:")
arr = np.arange(10)
print(f"原始数组: {arr}")
print(f"索引 arr[0] = {arr[0]}")
print(f"切片 arr[1:5] = {arr[1:5]}")
print(f"步长切片 arr[::2] = {arr[::2]}")

# 二维数组索引
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"\n二维数组:\n{matrix}")
print(f"matrix[0, 1] = {matrix[0, 1]}")  # 第一行第二列
print(f"matrix[:, 1] = {matrix[:, 1]}")  # 第二列

# 5. 常用数学函数
print("\n5. 常用数学函数:")
data = np.array([1, 2, 3, 4, 5])
print(f"数据: {data}")
print(f"求和: {np.sum(data)}")
print(f"平均值: {np.mean(data)}")
print(f"最大值: {np.max(data)}")
print(f"最小值: {np.min(data)}")
print(f"标准差: {np.std(data)}")

# 6. 数组重塑
print("\n6. 数组重塑:")
arr_1d = np.arange(12)
print(f"一维数组: {arr_1d}")
arr_2d = arr_1d.reshape(3, 4)
print(f"重塑为3x4:\n{arr_2d}")
arr_3d = arr_1d.reshape(2, 2, 3)
print(f"重塑为2x2x3:\n{arr_3d}")

# 7. 随机数生成
print("\n7. 随机数生成:")
np.random.seed(42)  # 设置随机种子保证结果可重现
random_arr = np.random.rand(5)  # 0-1之间的随机数
print(f"随机数组: {random_arr}")

random_int = np.random.randint(1, 10, 5)  # 1-10之间的随机整数
print(f"随机整数: {random_int}")

# 8. 布尔索引
print("\n8. 布尔索引:")
numbers = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
mask = numbers > 5
print(f"原始数组: {numbers}")
print(f"布尔掩码 (>5): {mask}")
print(f"筛选结果: {numbers[mask]}")

print("\n=== NumPy基础教程完成 ===")