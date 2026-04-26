# Matplotlib可视化示例
# 这个文件演示了Matplotlib的基本用法，包括各种图表类型

import matplotlib.pyplot as plt
import numpy as np

print("=== Matplotlib可视化示例 ===")

# 1. 基本线图
print("\n1. 基本线图:")
x = np.linspace(0, 10, 100)  # 创建从0到10的100个等间距点
y = np.sin(x)  # 计算每个x点的正弦值
plt.figure(figsize=(10, 6))  # 创建一个新的图形窗口，尺寸为10x6英寸
plt.plot(x, y, label='sin(x)', color='blue', linewidth=2)  # 绘制线图，x和y是坐标，label是图例，color是颜色，linewidth是线宽
plt.title('正弦函数')  # 设置图表标题
plt.xlabel('x')  # 设置x轴标签
plt.ylabel('sin(x)')  # 设置y轴标签
plt.grid(True, alpha=0.3)  # 添加网格线，透明度0.3
plt.legend()  # 显示图例
plt.savefig('line_plot.png')  # 将图表保存为PNG文件
print("线图已保存为 line_plot.png")
# plt.show()

# 2. 散点图
print("\n2. 散点图:")
x = np.random.randn(100)  # 生成100个标准正态分布的随机x坐标
y = np.random.randn(100)  # 生成100个标准正态分布的随机y坐标
colors = np.random.rand(100)  # 生成100个0到1之间的随机颜色值
sizes = 1000 * np.random.rand(100)  # 生成100个随机大小（0到1000之间）
plt.figure(figsize=(8, 6))  # 创建一个新的图形窗口，尺寸为8x6英寸
plt.scatter(x, y, c=colors, s=sizes, alpha=0.5, cmap='viridis')  # 绘制散点图，c是颜色，s是大小，alpha是透明度，cmap是颜色映射
plt.title('随机散点图')  # 设置图表标题
plt.xlabel('X轴')  # 设置x轴标签
plt.ylabel('Y轴')  # 设置y轴标签
plt.colorbar()  # 添加颜色条
plt.savefig('scatter_plot.png')  # 将图表保存为PNG文件
print("散点图已保存为 scatter_plot.png")
# plt.show()

# 3. 柱状图
print("\n3. 柱状图:")
categories = ['A', 'B', 'C', 'D', 'E']  # 定义类别标签
values = [23, 45, 56, 78, 32]  # 定义每个类别的值
plt.figure(figsize=(8, 6))  # 创建一个新的图形窗口，尺寸为8x6英寸
bars = plt.bar(categories, values, color=['red', 'green', 'blue', 'orange', 'purple'])  # 绘制柱状图，返回柱子对象
plt.title('类别数据柱状图')  # 设置图表标题
plt.xlabel('类别')  # 设置x轴标签
plt.ylabel('值')  # 设置y轴标签
for bar, value in zip(bars, values):  # 遍历每个柱子，在柱子上方添加数值标签
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, str(value), ha='center', va='bottom')  # 添加文本标签
plt.savefig('bar_plot.png')  # 将图表保存为PNG文件
print("柱状图已保存为 bar_plot.png")
# plt.show()

# 4. 饼图
print("\n4. 饼图:")
labels = ['苹果', '香蕉', '橙子', '葡萄', '其他']  # 定义饼图的标签
sizes = [30, 25, 20, 15, 10]  # 定义每个部分的百分比大小
colors = ['red', 'yellow', 'orange', 'purple', 'gray']  # 定义每个部分的颜色
explode = (0.1, 0, 0, 0, 0)  # 定义突出显示的部分（第一个部分突出0.1）
plt.figure(figsize=(8, 8))  # 创建一个新的图形窗口，尺寸为8x8英寸
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)  # 绘制饼图，autopct显示百分比，shadow添加阴影
plt.title('水果偏好调查')  # 设置图表标题
plt.axis('equal')  # 确保饼图是圆形的
plt.savefig('pie_plot.png')  # 将图表保存为PNG文件
print("饼图已保存为 pie_plot.png")
# plt.show()

# 5. 直方图
print("\n5. 直方图:")
data = np.random.normal(0, 1, 1000)  # 生成1000个均值为0，标准差为1的正态分布随机数
plt.figure(figsize=(8, 6))  # 创建一个新的图形窗口，尺寸为8x6英寸
plt.hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')  # 绘制直方图，bins是柱子数量，alpha是透明度
plt.title('正态分布直方图')  # 设置图表标题
plt.xlabel('值')  # 设置x轴标签
plt.ylabel('频次')  # 设置y轴标签
plt.grid(True, alpha=0.3)  # 添加网格线
plt.savefig('histogram.png')  # 将图表保存为PNG文件
print("直方图已保存为 histogram.png")
# plt.show()

# 6. 箱线图
print("\n6. 箱线图:")
data1 = np.random.normal(0, 1, 100)  # 生成第一个数据集：均值0，标准差1的100个随机数
data2 = np.random.normal(1, 1.5, 100)  # 生成第二个数据集：均值1，标准差1.5的100个随机数
data3 = np.random.normal(-1, 1, 100)  # 生成第三个数据集：均值-1，标准差1的100个随机数
plt.figure(figsize=(8, 6))  # 创建一个新的图形窗口，尺寸为8x6英寸
plt.boxplot([data1, data2, data3], tick_labels=['数据集1', '数据集2', '数据集3'])  # 绘制箱线图，显示数据的分布情况
plt.title('三个数据集的箱线图')  # 设置图表标题
plt.ylabel('值')  # 设置y轴标签
plt.grid(True, alpha=0.3)  # 添加网格线
plt.savefig('boxplot.png')  # 将图表保存为PNG文件
print("箱线图已保存为 boxplot.png")
# plt.show()

# 7. 子图
print("\n7. 子图:")
x = np.linspace(0, 10, 100)  # 创建从0到10的100个等间距点
y1 = np.sin(x)  # 计算正弦值
y2 = np.cos(x)  # 计算余弦值
y3 = np.tan(x)  # 计算正切值

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))  # 创建2x2的子图布局，fig是整个图形，ax是每个子图的轴

ax1.plot(x, y1, 'b-')  # 在第一个子图中绘制正弦曲线，'b-'表示蓝色实线
ax1.set_title('sin(x)')  # 设置第一个子图的标题
ax1.grid(True)  # 为第一个子图添加网格

ax2.plot(x, y2, 'r-')  # 在第二个子图中绘制余弦曲线，'r-'表示红色实线
ax2.set_title('cos(x)')  # 设置第二个子图的标题
ax2.grid(True)  # 为第二个子图添加网格

ax3.plot(x, y3, 'g-')  # 在第三个子图中绘制正切曲线，'g-'表示绿色实线
ax3.set_title('tan(x)')  # 设置第三个子图的标题
ax3.grid(True)  # 为第三个子图添加网格

# 第四个子图：散点图
ax4.scatter(np.random.randn(50), np.random.randn(50))  # 在第四个子图中绘制随机散点图
ax4.set_title('随机散点')  # 设置第四个子图的标题
ax4.grid(True)  # 为第四个子图添加网格

plt.tight_layout()  # 自动调整子图布局，避免重叠
plt.savefig('subplots.png')  # 将整个子图保存为PNG文件
print("子图已保存为 subplots.png")
# plt.show()

print("\n=== Matplotlib可视化示例完成 ===")