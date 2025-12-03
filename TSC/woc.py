import json
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import os

# JSON文件路径（确保路径正确）
json_file_path = r"C:\Users\34517\Desktop\zuhui\xITSC\result\classification_results.json"

# 读取JSON文件
try:
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"成功读取文件: {json_file_path}")
    print(f"数据包含 {len(data.get('results', []))} 个样本")
except FileNotFoundError:
    print(f"错误：找不到文件 {json_file_path}")
    print("请检查文件路径是否正确，或文件是否存在")
    exit(1)
except json.JSONDecodeError:
    print(f"错误：文件 {json_file_path} 不是有效的JSON格式")
    exit(1)
except Exception as e:
    print(f"读取文件时发生错误：{str(e)}")
    exit(1)

# 验证数据结构
if 'results' not in data or not isinstance(data['results'], list):
    print("错误：JSON文件格式不正确，缺少'results'列表")
    exit(1)

# 从数据中提取真实标签
true_labels = []
for item in data['results']:
    if 'true_label' in item:
        true_labels.append(item['true_label'])
    else:
        print(f"警告：样本 {item.get('sample_id', '未知')} 缺少 'true_label' 字段")

        true_labels.append(None)

# 检查是否有无效的真实标签
if any(label is None for label in true_labels):
    print("错误：部分样本缺少真实标签，无法计算准确率")
    exit(1)

print(f"成功获取 {len(true_labels)} 个样本的真实标签")
print(f"真实标签分布: {pd.Series(true_labels).value_counts().to_dict()}")

# 提取预测结果（增加键存在性检查）
required_keys = ['round4_r', 'round3_1', 'round3_2', 'round3_3', 'round4_m']
for idx, item in enumerate(data['results']):
    missing_keys = [key for key in required_keys if key not in item]
    if missing_keys:
        print(f"错误：样本 {item.get('sample_id', f'索引{idx}')} 缺少必要字段：{missing_keys}")
        exit(1)

round4_r_pred = [item['round4_r'] for item in data['results']]
round3_1_pred = [item['round3_1'] for item in data['results']]
round3_2_pred = [item['round3_2'] for item in data['results']]
round3_3_pred = [item['round3_3'] for item in data['results']]
round4_m_pred = [item['round4_m'] for item in data['results']]

# 计算round3三个结果的投票结果（>=2/3）
round3_vote_pred = []
for item in data['results']:
    votes = [item['round3_1'], item['round3_2'], item['round3_3']]
    if sum(votes) >= 2:  # 大于等于2/3即2票或3票
        round3_vote_pred.append(1)
    else:
        round3_vote_pred.append(0)


def calculate_metrics(true_labels, pred_labels, method_name):
    """计算分类指标"""
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    recall = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)
    cm = confusion_matrix(true_labels, pred_labels)

    print(f"\n{method_name} 结果:")
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    print(f"混淆矩阵:\n{cm}")

    return {
        'method': method_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm.tolist()  # 转换为列表，方便后续处理
    }


# 计算各方法的指标
results = []
results.append(calculate_metrics(true_labels, round4_r_pred, "Round4_R"))
results.append(calculate_metrics(true_labels, round3_1_pred, "Round3_1"))
results.append(calculate_metrics(true_labels, round3_vote_pred, "Round3_Vote"))
results.append(calculate_metrics(true_labels, round4_m_pred, "Round4_M"))

# 分析修正情况
print("\n" + "=" * 50)
print("修正效果分析 (Round4_R vs Round4_M)")
print("=" * 50)

# 统计修正情况
correction_correct = 0
correction_wrong = 0
no_correction_correct = 0
no_correction_wrong = 0

for i, item in enumerate(data['results']):
    round4_m = item['round4_m']
    round4_r = item['round4_r']
    true_label = true_labels[i]

    if round4_m != round4_r:
        # 发生了修正
        if round4_r == true_label:
            correction_correct += 1
        else:
            correction_wrong += 1
    else:
        # 未修正
        if round4_m == true_label:
            no_correction_correct += 1
        else:
            no_correction_wrong += 1

# 计算修正比例
total_corrections = correction_correct + correction_wrong
if total_corrections > 0:
    correction_accuracy = correction_correct / total_corrections
else:
    correction_accuracy = 0

print(f"\n修正情况统计:")
print(f"发生修正的样本数: {total_corrections}")
print(f"修正正确: {correction_correct}次")
print(f"修正错误: {correction_wrong}次")
print(f"修正正确率: {correction_accuracy:.4f} ({correction_accuracy*100:.2f}%)")
print(f"未修正且正确: {no_correction_correct}次")
print(f"未修正且错误: {no_correction_wrong}次")

# 创建汇总表格
print("\n" + "=" * 60)
print("分类性能汇总")
print("=" * 60)
summary_df = pd.DataFrame(results)
summary_df = summary_df[['method', 'accuracy', 'precision', 'recall', 'f1']]
print(summary_df.round(4))

# 添加修正准确率到汇总表格
correction_summary = pd.DataFrame([{
    'method': '修正准确率',
    'accuracy': correction_accuracy,
    'precision': '-',
    'recall': '-',
    'f1': '-'
}])
summary_df = pd.concat([summary_df, correction_summary], ignore_index=True)
print(f"\n包含修正准确率的汇总:")
print(summary_df.round(4))

# 可选：将结果保存到Excel文件（方便后续查看）
output_excel = r"C:\Users\34517\Desktop\zuhui\xITSC\result\classification_metrics.xlsx"
with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
    # 性能汇总
    summary_df.round(4).to_excel(writer, sheet_name='性能汇总', index=False)

    # 混淆矩阵
    cm_data = []
    for res in results:
        cm = res['confusion_matrix']
        # 处理不同大小的混淆矩阵（1x1或2x2）
        if len(cm) == 1:
            tn, fp, fn, tp = 0, 0, 0, cm[0][0]
        else:
            tn = cm[0][0] if len(cm[0]) > 0 else 0
            fp = cm[0][1] if len(cm[0]) > 1 else 0
            fn = cm[1][0] if len(cm) > 1 and len(cm[1]) > 0 else 0
            tp = cm[1][1] if len(cm) > 1 and len(cm[1]) > 1 else 0

        cm_data.append({
            'method': res['method'],
            'True Negative (TN)': tn,
            'False Positive (FP)': fp,
            'False Negative (FN)': fn,
            'True Positive (TP)': tp
        })
    cm_df = pd.DataFrame(cm_data)
    cm_df.to_excel(writer, sheet_name='混淆矩阵详情', index=False)

    # 修正分析
    correction_stats = pd.DataFrame({
        '统计项': ['总修正次数', '修正正确', '修正错误', '修正正确率'],
        '数值': [total_corrections, correction_correct, correction_wrong, f"{correction_accuracy:.4f}"]
    })
    correction_stats.to_excel(writer, sheet_name='修正统计', index=False)

print(f"\n结果已保存到Excel文件: {output_excel}")