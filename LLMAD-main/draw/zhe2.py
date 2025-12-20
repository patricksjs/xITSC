import matplotlib.pyplot as plt

# 数据
x_labels = ['0', '1', '2', '3']  # 分类 x 轴标签

data = {
    'Computers': [0.54, 0.62, 0.68, 0.64],
    'Epilepsy': [0.8403, 0.8992, 0.9275, 0.9275],
    'StarLightCurves': [0.7374, 0.8619, 0.8476, 0.8476]
}

# 创建图形
plt.figure(figsize=(10, 6))

# 定义新颜色（更柔和、清晰区分）
colors = ['#FF6B6B', '#4ECDC4', '#2537D1']  # 红、青、蓝

# 绘制每条折线
for i, (name, values) in enumerate(data.items()):
    plt.plot(x_labels, values, marker='s', linestyle='-', color=colors[i], label=name, linewidth=3)

# 设置标题和坐标轴标签（字号增大）
# plt.title('Performance Comparison Across Different Datasets', fontsize=16)
plt.ylim(0.5, 1)
plt.xlabel('Iteration', fontsize=20)
plt.ylabel('Accuracy (%)', fontsize=20)

# 增大刻度标签字号
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# 增大图例字号
plt.legend(fontsize=15)

# 添加网格
plt.grid(True, alpha=0.3)

# 自动调整布局
plt.tight_layout()

# 显示图表
plt.savefig('zhe2.png')