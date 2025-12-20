import matplotlib.pyplot as plt

# 数据
x_labels = ['0', '1', '2', '3']  # x轴标签

data = {
    'length': [146, 410, 481, 481],
    'keywords': [60, 112, 115, 114]
}

# 创建图形
plt.figure(figsize=(10, 6))

# 定义颜色
colors = ['#5861AC', '#F28080']  # 红、青

# 绘制每条折线
for i, (name, values) in enumerate(data.items()):
    plt.plot(x_labels, values, marker='o', linestyle='-', color=colors[i], label=name, linewidth=3)

# 设置标题和坐标轴标签（字号增大）
# plt.title('Length and Keywords Over Time', fontsize=16)
plt.xlabel('Iteration', fontsize=20)
plt.ylabel('Count', fontsize=20)

# 增大刻度标签字号
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# 图例
plt.legend(fontsize=15)

# 添加网格
plt.grid(True, alpha=0.3)

# 自动调整布局
plt.tight_layout()

# 显示图表
plt.savefig('zhe3.png')