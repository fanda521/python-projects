# Matplotlib可视化示例
# 这个文件演示了Matplotlib的基本用法，包括各种图表类型

import matplotlib.pyplot as plt
import numpy as np

print("=== Matplotlib可视化示例 ===")

# 1. 基本线图
print("\n1. 基本线图:")
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='sin(x)', color='blue', linewidth=2)
plt.title('正弦函数')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('line_plot.png')
print("线图已保存为 line_plot.png")
# plt.show()

# 2. 散点图
print("\n2. 散点图:")
x = np.random.randn(100)
y = np.random.randn(100)
colors = np.random.rand(100)
sizes = 1000 * np.random.rand(100)
plt.figure(figsize=(8, 6))
plt.scatter(x, y, c=colors, s=sizes, alpha=0.5, cmap='viridis')
plt.title('随机散点图')
plt.xlabel('X轴')
plt.ylabel('Y轴')
plt.colorbar()
plt.savefig('scatter_plot.png')
print("散点图已保存为 scatter_plot.png")
# plt.show()

# 3. 柱状图
print("\n3. 柱状图:")
categories = ['A', 'B', 'C', 'D', 'E']
values = [23, 45, 56, 78, 32]
plt.figure(figsize=(8, 6))
bars = plt.bar(categories, values, color=['red', 'green', 'blue', 'orange', 'purple'])
plt.title('类别数据柱状图')
plt.xlabel('类别')
plt.ylabel('值')
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, str(value), ha='center', va='bottom')
plt.savefig('bar_plot.png')
print("柱状图已保存为 bar_plot.png")
# plt.show()

# 4. 饼图
print("\n4. 饼图:")
labels = ['苹果', '香蕉', '橙子', '葡萄', '其他']
sizes = [30, 25, 20, 15, 10]
colors = ['red', 'yellow', 'orange', 'purple', 'gray']
explode = (0.1, 0, 0, 0, 0)  # 突出显示第一个扇形
plt.figure(figsize=(8, 8))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('水果偏好调查')
plt.axis('equal')
plt.savefig('pie_plot.png')
print("饼图已保存为 pie_plot.png")
# plt.show()

# 5. 直方图
print("\n5. 直方图:")
data = np.random.normal(0, 1, 1000)
plt.figure(figsize=(8, 6))
plt.hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('正态分布直方图')
plt.xlabel('值')
plt.ylabel('频次')
plt.grid(True, alpha=0.3)
plt.savefig('histogram.png')
print("直方图已保存为 histogram.png")
# plt.show()

# 6. 箱线图
print("\n6. 箱线图:")
data1 = np.random.normal(0, 1, 100)
data2 = np.random.normal(1, 1.5, 100)
data3 = np.random.normal(-1, 1, 100)
plt.figure(figsize=(8, 6))
plt.boxplot([data1, data2, data3], labels=['数据集1', '数据集2', '数据集3'])
plt.title('三个数据集的箱线图')
plt.ylabel('值')
plt.grid(True, alpha=0.3)
plt.savefig('boxplot.png')
print("箱线图已保存为 boxplot.png")
# plt.show()

# 7. 子图
print("\n7. 子图:")
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.tan(x)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

ax1.plot(x, y1, 'b-')
ax1.set_title('sin(x)')
ax1.grid(True)

ax2.plot(x, y2, 'r-')
ax2.set_title('cos(x)')
ax2.grid(True)

ax3.plot(x, y3, 'g-')
ax3.set_title('tan(x)')
ax3.grid(True)

# 第四个子图：散点图
ax4.scatter(np.random.randn(50), np.random.randn(50))
ax4.set_title('随机散点')
ax4.grid(True)

plt.tight_layout()
plt.savefig('subplots.png')
print("子图已保存为 subplots.png")
# plt.show()

print("\n=== Matplotlib可视化示例完成 ===")