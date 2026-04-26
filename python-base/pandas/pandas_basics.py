# Pandas基础教程示例
# 这个文件演示了Pandas的基本用法，包括Series和DataFrame

import pandas as pd
import numpy as np

print("=== Pandas基础教程 ===")

# 1. Series创建
print("\n1. Series创建:")

# 从列表创建Series
s1 = pd.Series([1, 3, 5, 6, 8])
print(f"从列表创建Series:\n{s1}")

# 从字典创建Series
s2 = pd.Series({'a': 1, 'b': 2, 'c': 3, 'd': 4})
print(f"\n从字典创建Series:\n{s2}")

# 指定索引
s3 = pd.Series([10, 20, 30], index=['x', 'y', 'z'])
print(f"\n指定索引的Series:\n{s3}")

# 2. Series属性和方法
print("\n2. Series属性和方法:")
print(f"值: {s1.values}")
print(f"索引: {s1.index}")
print(f"数据类型: {s1.dtype}")
print(f"形状: {s1.shape}")
print(f"大小: {s1.size}")

# 基本统计
print(f"求和: {s1.sum()}")
print(f"均值: {s1.mean()}")
print(f"最大值: {s1.max()}")
print(f"最小值: {s1.min()}")

# 3. DataFrame创建
print("\n3. DataFrame创建:")

# 从字典创建DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 28],
    'City': ['Beijing', 'Shanghai', 'Guangzhou', 'Shenzhen'],
    'Salary': [5000, 6000, 7000, 5500]
}
df1 = pd.DataFrame(data)
print(f"从字典创建DataFrame:\n{df1}")

# 从列表创建DataFrame
data2 = [
    ['Alice', 25, 'Beijing', 5000],
    ['Bob', 30, 'Shanghai', 6000],
    ['Charlie', 35, 'Guangzhou', 7000],
    ['David', 28, 'Shenzhen', 5500]
]
df2 = pd.DataFrame(data2, columns=['Name', 'Age', 'City', 'Salary'])
print(f"\n从列表创建DataFrame:\n{df2}")

# 4. DataFrame属性
print("\n4. DataFrame属性:")
print(f"形状: {df1.shape}")
print(f"列名: {list(df1.columns)}")
print(f"索引: {list(df1.index)}")
print(f"数据类型:\n{df1.dtypes}")

# 5. 数据选择
print("\n5. 数据选择:")

# 选择列
print(f"选择Name列:\n{df1['Name']}")
print(f"\n选择多列:\n{df1[['Name', 'Age']]}")

# 选择行
print(f"\n选择第一行:\n{df1.iloc[0]}")
print(f"\n选择前两行:\n{df1.iloc[0:2]}")

# 条件选择
print(f"\n年龄大于28的人:\n{df1[df1['Age'] > 28]}")

# 6. 数据操作
print("\n6. 数据操作:")

# 添加新列
df1['Department'] = ['HR', 'IT', 'Finance', 'Marketing']
print(f"添加部门列:\n{df1}")

# 修改数据
df1.loc[0, 'Salary'] = 5200
print(f"\n修改Alice的薪资:\n{df1}")

# 删除列
df_temp = df1.drop('Department', axis=1)
print(f"\n删除部门列:\n{df_temp}")

# 7. 数据统计
print("\n7. 数据统计:")
print(f"基本统计信息:\n{df1.describe()}")

print(f"\n按城市分组统计:\n{df1.groupby('City')['Salary'].mean()}")

# 8. 处理缺失值
print("\n8. 处理缺失值:")
df_missing = df1.copy()
df_missing.loc[0, 'Salary'] = np.nan
print(f"包含缺失值的DataFrame:\n{df_missing}")

print(f"\n检查缺失值:\n{df_missing.isnull()}")

# 填充缺失值
df_filled = df_missing.fillna(df_missing['Salary'].mean())
print(f"\n填充缺失值:\n{df_filled}")

# 9. 数据排序
print("\n9. 数据排序:")
print(f"按年龄排序:\n{df1.sort_values('Age')}")
print(f"\n按薪资降序排序:\n{df1.sort_values('Salary', ascending=False)}")

# 10. 文件操作
print("\n10. 文件操作:")

# 保存到CSV
df1.to_csv('sample_data.csv', index=False)
print("数据已保存到 sample_data.csv")

# 从CSV读取
df_read = pd.read_csv('sample_data.csv')
print(f"从CSV读取的数据:\n{df_read}")

print("\n=== Pandas基础教程完成 ===")