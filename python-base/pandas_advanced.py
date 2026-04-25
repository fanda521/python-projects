# Pandas高级用法示例
# 这个文件演示了Pandas的高级功能，包括数据合并、透视表、时间序列等

import pandas as pd
import numpy as np

print("=== Pandas高级用法 ===")

# 1. 数据合并
print("\n1. 数据合并:")

# 创建两个DataFrame
df1 = pd.DataFrame({
    'employee_id': [1, 2, 3, 4],
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'department': ['HR', 'IT', 'Finance', 'Marketing']
})

df2 = pd.DataFrame({
    'employee_id': [1, 2, 3, 5],
    'salary': [5000, 6000, 7000, 6500],
    'bonus': [500, 600, 700, 650]
})

print(f"员工信息:\n{df1}")
print(f"\n薪资信息:\n{df2}")

# 内连接合并
merged_inner = pd.merge(df1, df2, on='employee_id', how='inner')
print(f"\n内连接合并:\n{merged_inner}")

# 左连接合并
merged_left = pd.merge(df1, df2, on='employee_id', how='left')
print(f"\n左连接合并:\n{merged_left}")

# 外连接合并
merged_outer = pd.merge(df1, df2, on='employee_id', how='outer')
print(f"\n外连接合并:\n{merged_outer}")

# 2. 数据连接（concatenation）
print("\n2. 数据连接:")

df_a = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
df_b = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})

print(f"DataFrame A:\n{df_a}")
print(f"\nDataFrame B:\n{df_b}")

# 垂直连接
concat_vertical = pd.concat([df_a, df_b])
print(f"\n垂直连接:\n{concat_vertical}")

# 水平连接
concat_horizontal = pd.concat([df_a, df_b], axis=1)
print(f"\n水平连接:\n{concat_horizontal}")

# 3. 透视表
print("\n3. 透视表:")

# 创建销售数据
sales_data = pd.DataFrame({
    'Date': ['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02', '2023-01-03', '2023-01-03'],
    'Product': ['A', 'B', 'A', 'B', 'A', 'B'],
    'Region': ['North', 'North', 'South', 'South', 'North', 'South'],
    'Sales': [100, 150, 200, 250, 120, 180]
})

print(f"销售数据:\n{sales_data}")

# 创建透视表
pivot_table = pd.pivot_table(sales_data,
                           values='Sales',
                           index='Product',
                           columns='Region',
                           aggfunc='sum')
print(f"\n透视表（产品vs地区销售额）:\n{pivot_table}")

# 4. 分组操作的高级用法
print("\n4. 分组操作的高级用法:")

# 多级分组
grouped = sales_data.groupby(['Region', 'Product'])['Sales'].agg(['sum', 'mean', 'count'])
print(f"多级分组统计:\n{grouped}")

# 自定义聚合函数
def range_func(x):
    return x.max() - x.min()

custom_agg = sales_data.groupby('Product')['Sales'].agg(['sum', 'mean', range_func])
custom_agg.columns = ['总销售额', '平均销售额', '销售额范围']
print(f"\n自定义聚合:\n{custom_agg}")

# 5. 时间序列处理
print("\n5. 时间序列处理:")

# 创建时间序列数据
dates = pd.date_range('2023-01-01', periods=10, freq='D')
ts_data = pd.DataFrame({
    'date': dates,
    'value': np.random.randn(10).cumsum()
})
ts_data.set_index('date', inplace=True)

print(f"时间序列数据:\n{ts_data}")

# 重采样
monthly_resample = ts_data.resample('2D').mean()
print(f"\n2天重采样:\n{monthly_resample}")

# 移动平均
ts_data['MA_3'] = ts_data['value'].rolling(window=3).mean()
print(f"\n3期移动平均:\n{ts_data}")

# 6. 数据清理
print("\n6. 数据清理:")

# 创建包含问题的DataFrame
messy_data = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'age': [25, 30, np.nan, 28, 35],
    'salary': [5000, 6000, 7000, np.nan, 8000],
    'department': ['HR', 'IT', 'Finance', 'Marketing', 'HR']
})

print(f"原始数据（包含缺失值）:\n{messy_data}")

# 检查缺失值
print(f"\n缺失值统计:\n{messy_data.isnull().sum()}")

# 填充缺失值
cleaned_data = messy_data.copy()
cleaned_data['age'] = cleaned_data['age'].fillna(cleaned_data['age'].mean())
cleaned_data['salary'] = cleaned_data['salary'].fillna(cleaned_data['salary'].median())

print(f"\n填充后的数据:\n{cleaned_data}")

# 7. 数据转换
print("\n7. 数据转换:")

# 使用apply函数
def categorize_salary(salary):
    if salary < 5500:
        return '低'
    elif salary < 7500:
        return '中'
    else:
        return '高'

cleaned_data['salary_category'] = cleaned_data['salary'].apply(categorize_salary)
print(f"添加薪资等级:\n{cleaned_data}")

# 使用map函数
dept_mapping = {'HR': '人力资源', 'IT': '信息技术', 'Finance': '财务', 'Marketing': '市场营销'}
cleaned_data['department_cn'] = cleaned_data['department'].map(dept_mapping)
print(f"\n部门名称映射:\n{cleaned_data}")

# 8. 数据透视和重塑
print("\n8. 数据透视和重塑:")

# 创建宽格式数据
wide_data = pd.DataFrame({
    'student': ['Alice', 'Bob', 'Charlie'],
    'math': [90, 85, 88],
    'english': [85, 90, 87],
    'physics': [88, 86, 92]
})

print(f"宽格式数据:\n{wide_data}")

# 转换为长格式
long_data = pd.melt(wide_data, id_vars=['student'], var_name='subject', value_name='score')
print(f"\n长格式数据:\n{long_data}")

# 9. 字符串操作
print("\n9. 字符串操作:")

# 创建包含字符串的DataFrame
text_data = pd.DataFrame({
    'name': ['Alice Johnson', 'Bob Smith', 'Charlie Brown', 'David Wilson'],
    'email': ['alice@example.com', 'bob.smith@company.com', 'charlie.brown@test.org', 'david.wilson@data.net']
})

print(f"文本数据:\n{text_data}")

# 字符串方法
text_data['first_name'] = text_data['name'].str.split().str[0]
text_data['last_name'] = text_data['name'].str.split().str[1]
text_data['domain'] = text_data['email'].str.split('@').str[1]

print(f"\n提取的信息:\n{text_data[['name', 'first_name', 'last_name', 'domain']]}")

# 10. 性能优化
print("\n10. 性能优化:")

# 创建大数据集
large_data = pd.DataFrame({
    'A': np.random.randn(100000),
    'B': np.random.randn(100000),
    'C': np.random.randn(100000)
})

print(f"大数据集形状: {large_data.shape}")

# 使用向量化操作（推荐）
import time
start_time = time.time()
result_vectorized = large_data['A'] + large_data['B'] + large_data['C']
vectorized_time = time.time() - start_time

# 使用循环（不推荐）
start_time = time.time()
result_loop = []
for i in range(len(large_data)):
    result_loop.append(large_data.iloc[i]['A'] + large_data.iloc[i]['B'] + large_data.iloc[i]['C'])
loop_time = time.time() - start_time

print(f"向量化计算时间: {vectorized_time:.4f} 秒")
print(f"循环计算时间: {loop_time:.4f} 秒")
print("\n=== Pandas高级用法完成 ===")