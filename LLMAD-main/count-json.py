import json


def calculate_accuracy(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    correct = 0
    total = len(data)

    for item in data:
        # 确保 label 和 predicted 都是相同类型（比如都是字符串或整数）
        if int(item['label']) == int(item['predicted']):
            correct += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy


# 使用示例
json_file = 'fewshot-ds/EOGHorizontalSignal2.json'  # 替换为你的 JSON 文件路径
acc = calculate_accuracy(json_file)
print(f"准确率: {acc:.4f} ({acc * 100:.2f}%)")
