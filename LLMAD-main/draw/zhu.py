import matplotlib.pyplot as plt
import numpy as np

# 数据（转换为数值，去掉 % 符号）
data = {
    'StarLightCurves': [-1.01,  -2.02, 4.04],
    'Epilepsy': [-6.19,  -2.83, -4.51],
    'ArrowHead': [0.1,  0.1, -1.43]
}

x_labels = ['2',  '4', '5']  # x轴标签

# 设置颜色（区分不同类别）
colors = ['#949B93', '#92B9BE', '#A9DA3D']

# 创建图形
plt.figure(figsize=(10, 6))

# 定义柱子宽度
bar_width = 0.25
x_pos = np.arange(len(x_labels))  # x轴位置
plt.axhline(y=0, color='r', linestyle='-', linewidth=1.5)  # 红色实线

# 绘制每组数据的柱状图
for i, (name, values) in enumerate(data.items()):
    offset = i * bar_width  # 每组偏移
    bars = plt.bar(x_pos + offset, values, bar_width, label=name, color=colors[i], edgecolor='black', linewidth=0.8)

    # # 添加数值标签在柱子上方（注意：负数要放在下方）
    # for j, bar in enumerate(bars):
    #     height = bar.get_height()
    #     if height > 0:
    #         plt.annotate(f'{height:+.2f}%', xy=(bar.get_x() + bar.get_width()/2, height),
    #                     xytext=(0, 3), textcoords='offset points',
    #                     ha='center', va='bottom', fontsize=10)
    #     else:
    #         plt.annotate(f'{height:+.2f}%', xy=(bar.get_x() + bar.get_width()/2, height),
    #                     xytext=(0, -10), textcoords='offset points',
    #                     ha='center', va='top', fontsize=10)

# 设置 x 轴标签和标题
plt.xlabel('In-context Image Number', fontsize=15)
plt.ylabel('Accuracy Change (%)', fontsize=15)
# plt.title('Performance Change Relative to Baseline (0%)', fontsize=16)

# 设置 x 轴标签
plt.xticks(x_pos + bar_width/2, x_labels, fontsize=15)
plt.yticks(fontsize=15)
# 设置 y 轴范围，确保包含 0 并对称
y_min = min(min(values) for values in data.values())
y_max = max(max(values) for values in data.values())
plt.ylim(-10, 10)

# 添加网格（水平线）
plt.grid(True, axis='y', alpha=0.3)

# 图例
plt.legend(fontsize=15)

# 紧凑布局
plt.tight_layout()

# 显示图表
plt.savefig('zhu.png')