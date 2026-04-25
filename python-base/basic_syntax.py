# Python基础语法示例
# 这个文件演示了Python的基础语法，包括变量、数据类型、运算符和控制流

# 1. 变量和数据类型
# Python是动态类型语言，不需要声明变量类型
name = "Alice"  # 字符串类型
age = 25        # 整数类型
height = 1.68   # 浮点数类型
is_student = True  # 布尔类型

# 列表（可变序列）
fruits = ["apple", "banana", "orange"]
# 字典（键值对）
person = {"name": "Bob", "age": 30, "city": "Beijing"}

print("=== 基础语法演示 ===")
print(f"姓名: {name}, 年龄: {age}, 身高: {height}")
print(f"是否学生: {is_student}")
print(f"水果列表: {fruits}")
print(f"个人信息: {person}")

# 2. 运算符
a = 10
b = 3
print(f"\n=== 运算符演示 ===")
print(f"加法: {a} + {b} = {a + b}")
print(f"减法: {a} - {b} = {a - b}")
print(f"乘法: {a} * {b} = {a * b}")
print(f"除法: {a} / {b} = {a / b}")
print(f"整除: {a} // {b} = {a // b}")
print(f"取余: {a} % {b} = {a % b}")
print(f"幂运算: {a} ** {b} = {a ** b}")

# 3. 控制流
print(f"\n=== 控制流演示 ===")

# if-else语句
if age >= 18:
    print("你是成年人")
else:
    print("你是未成年人")

# for循环
print("水果列表:")
for fruit in fruits:
    print(f"- {fruit}")

# while循环
count = 0
print("计数到5:")
while count < 5:
    print(count)
    count += 1

print("\n基础语法演示完成！")