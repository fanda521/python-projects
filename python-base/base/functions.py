# Python函数示例
# 这个文件演示了Python函数的定义、使用和相关概念

# 1. 函数定义
# 使用def关键字定义函数
def greet(name):
    """问候函数
    
    Args:
        name (str): 要问候的人名
    
    Returns:
        str: 问候语
    """
    return f"你好，{name}！"

# 2. 函数参数
# 位置参数
def add_numbers(a, b):
    """两个数相加
    
    Args:
        a (int/float): 第一个数
        b (int/float): 第二个数
    
    Returns:
        int/float: 相加结果
    """
    return a + b

# 默认参数
def power(base, exponent=2):
    """计算幂
    
    Args:
        base (int/float): 底数
        exponent (int, optional): 指数，默认为2
    
    Returns:
        int/float: 幂运算结果
    """
    return base ** exponent

# 可变参数（*args）
def sum_all(*numbers):
    """计算所有参数的和
    
    Args:
        *numbers: 可变数量的数字参数
    
    Returns:
        int/float: 所有数字的和
    """
    total = 0
    for num in numbers:
        total += num
    return total

# 关键字参数（**kwargs）
def create_person(**info):
    """创建个人信息字典
    
    Args:
        **info: 关键字参数
    
    Returns:
        dict: 个人信息字典
    """
    return info

# 3. 函数调用和返回值
print("=== 函数演示 ===")

# 基本函数调用
greeting = greet("小明")
print(greeting)

# 位置参数
result = add_numbers(5, 3)
print(f"5 + 3 = {result}")

# 默认参数
print(f"2的平方 = {power(2)}")
print(f"2的立方 = {power(2, 3)}")

# 可变参数
total = sum_all(1, 2, 3, 4, 5)
print(f"1+2+3+4+5 = {total}")

# 关键字参数
person = create_person(name="李华", age=28, city="上海")
print(f"创建的人: {person}")

# 4. 递归函数
def factorial(n):
    """计算阶乘（递归函数）
    
    Args:
        n (int): 正整数
    
    Returns:
        int: n的阶乘
    """
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)

print(f"5! = {factorial(5)}")

# 5. lambda函数（匿名函数）
square = lambda x: x ** 2
print(f"3的平方 = {square(3)}")

print("\n函数演示完成！")