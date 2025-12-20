import matplotlib.pyplot as plt
import numpy as np

# 数据
labels = ['1st neighbor correct', '2nd neighbor correct', '3rd neighbor correct', 'other correct']
bar_data = [90.36, 72.73, 61.53, 75.00]
pie_data = [70.39, 9.37, 3.96, 16.28]

# 统一颜色（每个类别一种颜色）
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Matplotlib 默认前4色

# 创建图形
fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(12, 8))

# 柱状图
ax1.bar(labels, bar_data, color=colors, width=0.6)
ax1.set_title('(b)', fontsize=16)
ax1.set_ylabel('Case Accuracy(%)', fontsize=20)
ax1.set_ylim([0, 100])
# 去掉柱形图x轴标注
ax1.set_xticks([])  # 隐藏x轴刻度
ax1.set_xticklabels([])  # 隐藏x轴标签

# 增大y轴刻度字体
ax1.tick_params(axis='y', labelsize=16)

# 饼图
ax2.pie(pie_data, colors=colors, autopct='%1.1f%%',textprops={'fontsize': 16},  startangle=90)
ax2.axis('equal')
ax2.set_title('(a)', fontsize=16)

# 创建图例（只标注 A/B/C/D 的颜色和名称）
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=colors[i], label=labels[i]) for i in range(len(labels))]

# 添加图例到图形下方（调整bbox_to_anchor的y值，避免被挤压）
fig.legend(handles=legend_elements, loc='lower center', ncol=len(labels),
           bbox_to_anchor=(0.5, -0.1), fontsize=16)  # y=-0.05 更靠下

# 调整底部间距，确保图例显示
plt.subplots_adjust(bottom=0.2)  # 底部预留足够空间

# 移除 tight_layout()，避免与手动布局冲突
plt.tight_layout()

plt.savefig('pie.png', bbox_inches='tight')  # 保存时加上bbox_inches='tight'，确保所有元素被包含