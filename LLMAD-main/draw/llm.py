
import matplotlib.pyplot as plt

# 数据：LLM 名称和对应的 accuracy
llm_names = [
    "gpt-5-mini",
    "gpt-4o",
    "gemini-2.5-flash",
    "deepseek-v3-2-exp",
    "o4-mini",
    "claude-sonnet-4-5",
    "qwen3-235b-a22b"
]

accuracies = [
    87.33,
    78.29,
    86.48,
    84.17,
    87.20,
    77.56,
    73.83
]

# 不同颜色列表

# 创建柱状图
plt.figure(figsize=(10, 6))  # 增大图形尺寸以便更好地显示文字
bar_width = 0.4  # 减小柱子宽度
colors = plt.cm.Paired.colors[:len(llm_names)]  # 取前7种
bars = plt.bar(llm_names, accuracies, color=colors, edgecolor='black', width=bar_width)

plt.ylabel("Accuracy (%)", fontsize=18)

# 设置 y 轴刻度显示百分比
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
# 为每个柱子添加数值标签
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.5, f'{acc:.2f}', ha='center', va='bottom', fontsize=18)

# 旋转 x 轴标签（避免重叠）
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 100)
# 调整布局防止标签被截断
plt.tight_layout()

# 显示图表
plt.savefig('llms.png')