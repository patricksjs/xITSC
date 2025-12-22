import json
import re
from collections import Counter
from typing import List, Dict, Tuple, Any, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def safe_json_parse(json_str: str) -> Optional[Any]:
    """安全地解析JSON字符串，支持更宽松的解析"""
    if not json_str or not isinstance(json_str, str):
        return None

    # 尝试直接解析
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # 尝试修复常见的JSON格式问题
        try:
            # 1. 移除额外的引号或转义字符
            cleaned = json_str.strip()

            # 2. 修复缺少逗号的问题（简单的修复尝试）
            # 匹配类似 "label 1"]\n    "score" 的情况
            cleaned = re.sub(r'("\s*\]\s*\n\s*")', r'"],\n"', cleaned)

            # 3. 修复键值对之间缺少逗号的问题
            lines = cleaned.split('\n')
            fixed_lines = []
            for i, line in enumerate(lines):
                fixed_lines.append(line)
                # 如果这一行以"结束，下一行以"开头但不是数组或对象结束，可能缺少逗号
                if i < len(lines) - 1:
                    if (line.strip().endswith('"') and
                            lines[i + 1].strip().startswith('"') and
                            not lines[i + 1].strip().startswith('"}')):
                        # 在当前行末尾添加逗号
                        fixed_lines[-1] = line.rstrip() + ','

            cleaned = '\n'.join(fixed_lines)

            return json.loads(cleaned)
        except:
            return None


def extract_label_from_response(response_str: str) -> Optional[int]:
    """
    从响应字符串中提取标签，使用多种方法尝试解析
    """
    if not response_str:
        return None

    # 方法1: 尝试JSON解析
    response = safe_json_parse(response_str)

    if response and isinstance(response, dict):
        # 尝试从result字段提取
        if 'result' in response:
            result = response['result']
            if isinstance(result, list) and len(result) > 0:
                label_str = str(result[0]).lower()
                if 'label 0' in label_str or 'label0' in label_str:
                    return 0
                elif 'label 1' in label_str or 'label1' in label_str:
                    return 1

    # 方法2: 如果JSON解析失败，尝试正则表达式匹配
    response_lower = response_str.lower()

    # 查找 "label 0" 或 "label 1"
    if '"label 0"' in response_lower or "'label 0'" in response_lower:
        return 0
    elif '"label 1"' in response_lower or "'label 1'" in response_lower:
        return 1

    # 方法3: 查找其他格式的标签声明
    if '"result": ["label 0"' in response_lower:
        return 0
    elif '"result": ["label 1"' in response_lower:
        return 1
    if '"result": ["label0"' in response_lower:
        return 0
    elif '"result": ["label1"' in response_lower:
        return 1

    # 方法4: 使用更宽松的正则表达式
    label_patterns = [
        r'label\s*0',
        r'label\s*1',
        r'class\s*0',
        r'class\s*1',
        r'类别\s*0',
        r'类别\s*1'
    ]

    for i, pattern in enumerate(label_patterns[:2]):  # 只检查前两个模式
        matches = re.findall(pattern, response_lower, re.IGNORECASE)
        if matches:
            return i  # 0 for label 0, 1 for label 1

    return None


def calculate_metrics(y_true: List[int], y_pred: List[int], label_name: str = ""):
    """
    计算并打印分类指标
    """
    print(f"\n{label_name} 指标:")
    print(f"样本数量: {len(y_true)}")
    print(f"真实标签分布: {Counter(y_true)}")
    print(f"预测标签分布: {Counter(y_pred)}")

    # 检查是否有预测结果
    if len(y_true) == 0 or len(y_pred) == 0:
        print("错误: 样本数量为0!")
        return None

    # 计算指标
    try:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    except Exception as e:
        print(f"计算指标时出错: {e}")
        return None

    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1分数: {f1:.4f}")

    # 混淆矩阵
    tp = sum((true == 1 and pred == 1) for true, pred in zip(y_true, y_pred))
    fp = sum((true == 0 and pred == 1) for true, pred in zip(y_true, y_pred))
    tn = sum((true == 0 and pred == 0) for true, pred in zip(y_true, y_pred))
    fn = sum((true == 1 and pred == 0) for true, pred in zip(y_true, y_pred))

    print(f"混淆矩阵:")
    print(f"             预测")
    print(f"            0     1")
    print(f"真实  0  [{tn:3d}  {fp:3d}]")
    print(f"      1  [{fn:3d}  {tp:3d}]")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}
    }


def main():
    # 读取JSON文件
    file_path = r"D:\zuhui\xITSC\result_phase2.json"

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return

    print(f"成功加载 {len(data)} 个样本")

    # 存储结果
    true_labels = []
    round5_predictions = []
    round4_first_predictions = []
    round4_majority_predictions = []

    # 跟踪无法解析的样本
    parse_errors = {
        'round5': 0,
        'round4_first': 0,
        'round4_responses': 0
    }

    # 处理每个样本
    for idx, sample in enumerate(data):
        # 提取真实标签
        true_label = sample.get('true_label')
        if true_label not in [0, 1]:
            print(f"样本 {idx} (ID: {sample.get('sample_id', 'unknown')}) 的真实标签无效: {true_label}")
            continue

        true_labels.append(true_label)

        # 1. 提取round5_response的分类结果
        round5_response = sample.get('round5_response', '')
        if round5_response:
            round5_label = extract_label_from_response(round5_response)
            if round5_label is None:
                parse_errors['round5'] += 1
                print(f"样本 {idx} (ID: {sample.get('sample_id', 'unknown')}) round5_response 无法解析标签")
            round5_predictions.append(round5_label)
        else:
            print(f"样本 {idx} (ID: {sample.get('sample_id', 'unknown')}) 缺少 round5_response")
            round5_predictions.append(None)

        # 2. 提取round4_responses第一轮的分类结果
        round4_responses = sample.get('round4_responses', [])
        if round4_responses and len(round4_responses) > 0:
            first_response = round4_responses[0]
            if first_response:
                first_label = extract_label_from_response(first_response)
                if first_label is None:
                    parse_errors['round4_first'] += 1
                    print(f"样本 {idx} (ID: {sample.get('sample_id', 'unknown')}) round4_responses[0] 无法解析标签")
                round4_first_predictions.append(first_label)
            else:
                round4_first_predictions.append(None)
        else:
            print(f"样本 {idx} (ID: {sample.get('sample_id', 'unknown')}) 缺少或空的 round4_responses")
            round4_first_predictions.append(None)

        # 3. 提取round4_responses三轮中的多数分类结果
        if round4_responses and len(round4_responses) >= 3:
            # 提取三个响应的标签
            labels = []
            for i in range(min(3, len(round4_responses))):
                if i < len(round4_responses) and round4_responses[i]:
                    label = extract_label_from_response(round4_responses[i])
                    if label is not None:
                        labels.append(label)

            if labels:
                # 多数投票
                counter = Counter(labels)
                majority_label = counter.most_common(1)[0][0]
                round4_majority_predictions.append(majority_label)
            else:
                parse_errors['round4_responses'] += 1
                print(f"样本 {idx} (ID: {sample.get('sample_id', 'unknown')}) 无法从 round4_responses 解析任何标签")
                round4_majority_predictions.append(None)
        else:
            print(f"样本 {idx} (ID: {sample.get('sample_id', 'unknown')}) round4_responses 少于3个")
            round4_majority_predictions.append(None)

    print(f"\n解析错误统计:")
    for key, count in parse_errors.items():
        print(f"  {key}: {count}")

    # 过滤掉None值
    valid_indices = []
    for i, (t, r5, r4f, r4m) in enumerate(zip(
            true_labels, round5_predictions, round4_first_predictions, round4_majority_predictions
    )):
        if t is not None and r5 is not None and r4f is not None and r4m is not None:
            valid_indices.append(i)

    valid_true = [true_labels[i] for i in valid_indices]
    valid_round5 = [round5_predictions[i] for i in valid_indices]
    valid_round4_first = [round4_first_predictions[i] for i in valid_indices]
    valid_round4_majority = [round4_majority_predictions[i] for i in valid_indices]

    print(f"\n总样本数: {len(data)}")
    print(f"有效样本数（所有预测均有效）: {len(valid_true)}")

    if len(valid_true) == 0:
        print("错误: 没有有效样本可用于计算指标!")
        return

    # 计算各项指标
    results = {}
    print("\n" + "=" * 60)
    results['round5'] = calculate_metrics(valid_true, valid_round5, "Round5 Response")
    print("\n" + "=" * 60)
    results['round4_first'] = calculate_metrics(valid_true, valid_round4_first, "Round4 First Response")
    print("\n" + "=" * 60)
    results['round4_majority'] = calculate_metrics(valid_true, valid_round4_majority, "Round4 Majority Vote")

    # 汇总比较
    print("\n" + "=" * 60)
    print("指标汇总比较:")
    print("=" * 60)
    print(f"{'方法':<25} {'准确率':<10} {'精确率':<10} {'召回率':<10} {'F1分数':<10}")
    print("-" * 60)
    for name, metrics in results.items():
        if metrics:  # 确保metrics不为None
            print(f"{name:<25} {metrics['accuracy']:.4f}    {metrics['precision']:.4f}    "
                  f"{metrics['recall']:.4f}    {metrics['f1']:.4f}")

    # 输出一些诊断信息
    print(f"\n诊断信息:")
    print(f"总样本数: {len(data)}")
    print(f"有效样本数: {len(valid_true)}")
    print(f"丢弃样本数: {len(data) - len(valid_true)}")

    # 显示前几个样本的真实和预测标签
    print(f"\n前5个样本的标签:")
    print(f"{'样本索引':<10} {'真实标签':<10} {'Round5':<10} {'Round4第一轮':<15} {'Round4多数投票':<15}")
    for i in range(min(5, len(valid_indices))):
        idx = valid_indices[i]
        print(f"{i:<10} {true_labels[idx]:<10} {round5_predictions[idx]:<10} "
              f"{round4_first_predictions[idx]:<15} {round4_majority_predictions[idx]:<15}")

    return results


if __name__ == "__main__":
    results = main()