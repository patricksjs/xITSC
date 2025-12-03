# filter_and_sort_samples.py
import os
import json
import re
from config import STAGE2_OUTPUT_FILE, CLASSIFICATION_RESULTS_FILE


def extract_number_from_string(text):
    """从字符串中提取第一个数字"""
    if not text:
        return None
    match = re.search(r'(\d+)', str(text))
    if match:
        return int(match.group(1))
    return None


def filter_and_sort_samples():
    """删除true_label与round4_r不同的样本，并对两个文件进行排序"""

    # 检查文件是否存在
    if not os.path.exists(STAGE2_OUTPUT_FILE):
        print(f"错误：stage2文件不存在 - {STAGE2_OUTPUT_FILE}")
        return

    if not os.path.exists(CLASSIFICATION_RESULTS_FILE):
        print(f"错误：分类结果文件不存在 - {CLASSIFICATION_RESULTS_FILE}")
        return

    # 加载stage2数据
    try:
        with open(STAGE2_OUTPUT_FILE, 'r', encoding='utf-8') as f:
            stage2_data = json.load(f)
    except Exception as e:
        print(f"加载stage2文件失败: {e}")
        return

    # 加载分类结果数据
    try:
        with open(CLASSIFICATION_RESULTS_FILE, 'r', encoding='utf-8') as f:
            classification_data = json.load(f)
    except Exception as e:
        print(f"加载分类结果文件失败: {e}")
        return

    # 获取需要删除的样本ID（true_label != round4_r）
    samples_to_delete = set()

    print("检查需要删除的样本（true_label != round4_r）:")
    print("-" * 50)

    for result in classification_data.get("results", []):
        sample_id = str(result.get("sample_id"))
        true_label = result.get("true_label")
        round4_r = result.get("round4_r")

        # 检查是否需要删除
        if true_label != round4_r:
            print(f"样本 {sample_id}: true_label={true_label}, round4_r={round4_r} -> 需要删除")
            samples_to_delete.add(sample_id)
        else:
            print(f"样本 {sample_id}: true_label={true_label}, round4_r={round4_r} -> 保留")

    print(f"\n总共需要删除 {len(samples_to_delete)} 个样本")

    # 从分类结果文件中删除样本
    if samples_to_delete:
        print(f"\n从分类结果文件中删除 {len(samples_to_delete)} 个样本...")
        classification_data["results"] = [
            result for result in classification_data.get("results", [])
            if str(result.get("sample_id")) not in samples_to_delete
        ]
        print(f"删除后分类结果文件剩余 {len(classification_data['results'])} 个样本")

    # 从stage2文件中删除样本
    if samples_to_delete:
        print(f"\n从stage2文件中删除 {len(samples_to_delete)} 个样本...")
        original_count = len(stage2_data.get("test_samples", []))
        stage2_data["test_samples"] = [
            sample for sample in stage2_data.get("test_samples", [])
            if str(sample.get("sample_id")) not in samples_to_delete
        ]
        new_count = len(stage2_data.get("test_samples", []))
        print(f"删除后stage2文件剩余 {new_count} 个样本（原 {original_count} 个）")

    # 对分类结果文件按样本ID排序
    print(f"\n对分类结果文件按样本ID排序...")
    classification_data["results"].sort(
        key=lambda x: extract_number_from_string(x.get("sample_id", 0)) or 0
    )
    print("分类结果文件排序完成")

    # 对stage2文件按样本ID排序
    print(f"对stage2文件按样本ID排序...")
    stage2_data["test_samples"].sort(
        key=lambda x: extract_number_from_string(x.get("sample_id", 0)) or 0
    )
    print("stage2文件排序完成")

    # 保存更新后的分类结果文件
    print(f"\n保存更新后的分类结果文件...")
    os.makedirs(os.path.dirname(CLASSIFICATION_RESULTS_FILE), exist_ok=True)
    with open(CLASSIFICATION_RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(classification_data, f, indent=4, ensure_ascii=False)
    print(f"分类结果文件已保存到: {CLASSIFICATION_RESULTS_FILE}")

    # 保存更新后的stage2文件
    print(f"保存更新后的stage2文件...")
    os.makedirs(os.path.dirname(STAGE2_OUTPUT_FILE), exist_ok=True)
    with open(STAGE2_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(stage2_data, f, indent=4, ensure_ascii=False)
    print(f"stage2文件已保存到: {STAGE2_OUTPUT_FILE}")

    # 验证两个文件的样本ID是否一致
    stage2_ids = {str(sample.get("sample_id")) for sample in stage2_data.get("test_samples", [])}
    classification_ids = {str(result.get("sample_id")) for result in classification_data.get("results", [])}

    print(f"\n验证结果:")
    print(f"stage2文件样本数量: {len(stage2_ids)}")
    print(f"分类结果文件样本数量: {len(classification_ids)}")

    if stage2_ids == classification_ids:
        print("✓ 两个文件的样本ID完全一致")
        print(f"样本ID列表: {sorted(stage2_ids, key=lambda x: extract_number_from_string(x) or 0)}")
    else:
        print("✗ 两个文件的样本ID不一致!")
        print(f"stage2独有的样本: {stage2_ids - classification_ids}")
        print(f"分类结果独有的样本: {classification_ids - stage2_ids}")


if __name__ == "__main__":
    filter_and_sort_samples()