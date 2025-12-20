import matplotlib.pyplot as plt

# 数据
x_labels = ['3', '6', '9']  # x轴：时间点或参数值，作为分类数据

data = {
    'Trace': [0.98, 0.94, 0.91],
    'Epilepsy': [0.9275, 0.8519, 0.8382],
    'StarLightCurves': [0.8619, 0.8476, 0.8619]
}



# 创建图形
plt.figure(figsize=(10, 6))

# 绘制每条折线
for name, values in data.items():
    plt.plot(x_labels, values, marker='o', linestyle='-', label=name, linewidth=3)

# 标题和标签
# plt.title('Performance Comparison Across Different Datasets')
plt.xlabel('Sample Number', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
plt.ylim([0.5, 1])

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)


plt.grid(True, alpha=0.3)
plt.legend(fontsize=15)

# 显示图表
plt.tight_layout()
plt.savefig('zhe1.png')