# Python面向对象编程示例
# 这个文件演示了Python的面向对象编程概念，包括类、对象、继承等

# 1. 类定义
class Person:
    """人 类
    
    Attributes:
        name (str): 姓名
        age (int): 年龄
    """
    
    # 类变量
    species = "人类"
    
    # 构造函数
    def __init__(self, name, age):
        """初始化方法
        
        Args:
            name (str): 姓名
            age (int): 年龄
        """
        self.name = name  # 实例变量
        self.age = age
    
    # 实例方法
    def introduce(self):
        """自我介绍方法
        
        Returns:
            str: 介绍信息
        """
        return f"我是{self.name}，今年{self.age}岁。"
    
    def celebrate_birthday(self):
        """庆祝生日（年龄加1）
        """
        self.age += 1
        print(f"{self.name}生日快乐！现在{self.age}岁了。")
    
    # 类方法
    @classmethod
    def get_species(cls):
        """获取物种信息
        
        Returns:
            str: 物种名称
        """
        return cls.species
    
    # 静态方法
    @staticmethod
    def is_adult(age):
        """判断是否成年
        
        Args:
            age (int): 年龄
        
        Returns:
            bool: 是否成年（>=18岁）
        """
        return age >= 18

# 2. 继承
class Student(Person):
    """学生类，继承自Person类
    
    Attributes:
        name (str): 姓名
        age (int): 年龄
        student_id (str): 学号
        major (str): 专业
    """
    
    def __init__(self, name, age, student_id, major):
        """初始化学生
        
        Args:
            name (str): 姓名
            age (int): 年龄
            student_id (str): 学号
            major (str): 专业
        """
        super().__init__(name, age)  # 调用父类构造函数
        self.student_id = student_id
        self.major = major
    
    # 重写父类方法
    def introduce(self):
        """学生自我介绍
        
        Returns:
            str: 介绍信息
        """
        return f"我是{self.name}，今年{self.age}岁，学号{self.student_id}，专业是{self.major}。"
    
    # 新增方法
    def study(self):
        """学习方法
        """
        print(f"{self.name}正在学习{self.major}。")

# 3. 多态示例
class Teacher(Person):
    """教师类，继承自Person类
    
    Attributes:
        name (str): 姓名
        age (int): 年龄
        subject (str): 教授科目
    """
    
    def __init__(self, name, age, subject):
        super().__init__(name, age)
        self.subject = subject
    
    def introduce(self):
        """教师自我介绍
        
        Returns:
            str: 介绍信息
        """
        return f"我是{self.name}老师，今年{self.age}岁，教授{self.subject}。"
    
    def teach(self):
        """教学方法
        """
        print(f"{self.name}老师正在教授{self.subject}。")

# 4. 对象创建和使用
print("=== 面向对象编程演示 ===")

# 创建Person对象
person1 = Person("张三", 25)
print(person1.introduce())
person1.celebrate_birthday()

# 创建Student对象
student1 = Student("李四", 20, "2021001", "计算机科学")
print(student1.introduce())
student1.study()

# 创建Teacher对象
teacher1 = Teacher("王五", 35, "数学")
print(teacher1.introduce())
teacher1.teach()

# 5. 类方法和静态方法
print(f"\n物种: {Person.get_species()}")
print(f"25岁是否成年: {Person.is_adult(25)}")
print(f"16岁是否成年: {Person.is_adult(16)}")

# 6. 多态演示
people = [person1, student1, teacher1]
print("\n=== 多态演示 ===")
for person in people:
    print(person.introduce())

print("\n面向对象编程演示完成！")