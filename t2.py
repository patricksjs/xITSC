import torch
import numpy as np
import argparse
from TSC.t1 import compute_energy_regions_similarity, find_most_similar_samples_improved  # 替换为核心函数所在模块名
from data.data_loader import load_data, normalize_data  # 替换为你的数据加载模块名（即提供的代码所在模块）

# 全局设备设置（与你的主代码保持一致）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_custom_ref_id_mapping(manual_ids, dataset_name):
    """
    按手动指定的索引创建参考样本ID映射（支持自定义ID格式）
    MANUAL_IDS格式：{标签: [样本索引列表], ...}
    ID格式：{dataset_name}_REF_{标签}_{序号}（例如 computer_REF_0_01）
    """
    sample_id_to_idx = {}
    idx_to_sample_id = {}
    label_sample_ids = {}  # 记录每个标签下的参考样本ID

    for label, indices in manual_ids.items():
        label_sample_ids[label] = []
        for seq, idx in enumerate(indices, 1):  # 序号从1开始
            # 自定义ID格式：数据集_REF_标签_两位序号（可按需修改）
            sample_id = f"{dataset_name}_REF_{label}_{seq:02d}"
            sample_id_to_idx[sample_id] = idx
            idx_to_sample_id[idx] = sample_id
            label_sample_ids[label].append(sample_id)

    return sample_id_to_idx, idx_to_sample_id, label_sample_ids


def prepare_reference_data_by_label(data, labels, idx_to_sample_id):
    """
    按类别整理参考数据（用于相似度计算）
    返回格式：{label: [样本数据, 样本数据, ...], ...}
    同时记录每个类别下的样本ID映射
    """
    reference_data = {}
    label_sample_id_mapping = {}  # 记录每个类别下的样本ID：{label: [sample_id1, sample_id2, ...], ...}

    unique_labels = torch.unique(labels).tolist()
    for label in unique_labels:
        # 找到当前类别的所有样本索引
        label_indices = [idx for idx in range(len(labels)) if labels[idx].item() == label]
        # 提取样本数据和对应的ID
        label_samples = [data[idx] for idx in label_indices]
        label_sample_ids = [idx_to_sample_id[idx] for idx in label_indices]

        reference_data[label] = label_samples
        label_sample_id_mapping[label] = label_sample_ids

    return reference_data, label_sample_id_mapping


def find_similar_samples_by_id(
        test_sample_id,
        dataset_name,
        manual_ref_ids=None,
        samples_per_label=2,
        min_similarity=0.3,
        nperseg=60,
        fs=1.0,
        energy_threshold=0.7
):
    """
    核心函数：根据测试样本ID查找最相似的参考样本
    Args:
        test_sample_id: 待测试样本的ID（格式如 computer_S0005）
        dataset_name: 数据集名称（需与ID前缀一致）
        samples_per_label: 每个类别返回的最相似样本数
        min_similarity: 最小相似度阈值
        nperseg: STFT窗口长度（与核心函数一致）
        fs: 采样频率（与核心函数一致）
        energy_threshold: 高能量区域阈值（与核心函数一致）
    Returns:
        最相似样本结果字典
    """
    # 1. 加载数据
    print(f"正在加载数据集: {dataset_name}")
    raw_data, labels = load_data(dataset_name)

    # 2. 数据归一化（使用你的归一化函数，保持数据一致性）
    print("正在进行数据归一化...")
    normalized_data, _, _ = normalize_data(raw_data)

    # 3. 创建样本ID映射
    print("正在创建样本ID映射...")
    if manual_ref_ids is not None:
        # 使用自定义参考样本ID（按MANUAL_IDS配置）
        sample_id_to_idx, idx_to_sample_id, label_sample_ids = create_custom_ref_id_mapping(
            manual_ids=manual_ref_ids,
            dataset_name=dataset_name
        )
        # 验证测试样本ID是否在自定义映射中
        if test_sample_id not in sample_id_to_idx:
            raise ValueError(f"测试样本ID {test_sample_id} 不在自定义参考样本列表中！")


    # 4. 准备参考数据（仅使用手动指定的索引对应的样本）
    print("正在整理自定义参考数据...")
    reference_data = {}
    if manual_ref_ids is not None:
        for label, indices in manual_ref_ids.items():
            # 提取手动指定索引的样本数据
            reference_data[label] = [normalized_data[idx] for idx in indices]
    else:
        # 兼容原有逻辑
        reference_data, _ = prepare_reference_data_by_label(normalized_data, labels, idx_to_sample_id)
    # 5. 获取测试样本数据
    test_idx = sample_id_to_idx[test_sample_id]
    test_sample = normalized_data[test_idx]
    test_label = labels[test_idx].item()
    print(f"\n测试样本信息：")
    print(f"ID: {test_sample_id} | 索引: {test_idx} | 真实标签: {test_label}")

    if manual_ref_ids is not None:
        label_sample_ids = {}
        for label, indices in manual_ref_ids.items():
            label_sample_ids[label] = [idx_to_sample_id[idx] for idx in indices]
    # 7. 计算相似度（调用核心函数）
    print("正在计算高能量区域相似度...")
    similarities = compute_energy_regions_similarity(
        test_sample_data=test_sample,
        reference_samples_data=reference_data,
        nperseg=nperseg,
        fs=fs,
        energy_threshold=energy_threshold
    )

    # 8. 查找最相似样本（调用改进版函数）
    print("正在查找最相似样本...")
    most_similar_indices = find_most_similar_samples_improved(
        test_sample_data=test_sample,
        reference_samples_data=reference_data,
        samples_per_label=samples_per_label,
        min_similarity=min_similarity
    )

    # 9. 整理结果（将索引转换为ID，并添加相似度得分）
    final_results = {
        "test_sample_info": {
            "id": test_sample_id,
            "index": test_idx,
            "true_label": test_label
        },
        "similar_samples": {}
    }

    for label, top_indices in most_similar_indices.items():
        label_results = []
        for idx in top_indices:
            similar_sample_id = label_sample_ids[label][idx]
            similar_sample_score = similarities[label][idx]
            label_results.append({
                "id": similar_sample_id,
                "index": sample_id_to_idx[similar_sample_id],
                "label": label,
                "similarity_score": round(similar_sample_score, 4)
            })
        final_results["similar_samples"][f"label_{label}"] = label_results

    return final_results


def print_results(results):
    """格式化打印结果"""
    print("\n" + "=" * 80)
    print("最相似样本查找结果")
    print("=" * 80)

    # 打印测试样本信息
    test_info = results["test_sample_info"]
    print(f"\n【测试样本】")
    print(f"ID: {test_info['id']} | 索引: {test_info['index']} | 真实标签: {test_info['true_label']}")

    # 打印每个类别的相似样本
    print(f"\n【最相似样本列表】")
    for label_key, similar_samples in results["similar_samples"].items():
        label = label_key.split("_")[-1]
        print(f"\n类别 {label}（共 {len(similar_samples)} 个相似样本）:")
        print("-" * 50)
        print(f"{'排名':<6} {'样本ID':<20} {'相似度得分':<15}")
        print("-" * 50)
        for rank, sample in enumerate(similar_samples, 1):
            print(f"{rank:<6} {sample['id']:<20} {sample['similarity_score']:<15}")

    print("\n" + "=" * 80)


def main():
    # 参数解析（支持命令行传入参数）
    parser = argparse.ArgumentParser(description='根据样本ID查找最相似参考样本')
    # 修正：测试样本ID参数名改为 --test_id（与代码中引用一致）
    parser.add_argument('--test_id', type=str, required=True,
                        help='待测试样本的ID（格式：数据集名称_REF_标签_序号，例如 computer_REF_0_01）')
    # 修正：数据集参数名改为 --dataset（符合通用命名，与ID验证逻辑一致）
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['computer', 'cincecgtorso', 'yoga', 'RFD', 'midair', 'UMD', 'forda',
                                 'fordb', 'strawberry', 'ECG200', 'gunpointmalefemale', 'Freezer',
                                 'blink', 'arrowhead', 'EPG', 'EPG1', 'LKA', 'Blink', 'ShapeletSim', 'twopatterns'],
                        help='数据集名称（必须与测试样本ID的前缀一致）')
    parser.add_argument('--samples_per_label', type=int, default=2,
                        help='每个类别返回的最相似样本数')
    parser.add_argument('--min_similarity', type=float, default=0.3,
                        help='最小相似度阈值（低于此值的样本不返回）')
    parser.add_argument('--nperseg', type=int, default=60,
                        help='STFT窗口长度')
    parser.add_argument('--energy_threshold', type=float, default=0.7,
                        help='高能量区域百分比阈值（0-1之间）')
    parser.add_argument('--use_manual_ref', action='store_true',
                        help='是否使用手动指定的参考样本ID（需在代码中配置MANUAL_IDS）')
    args = parser.parse_args()

    # 验证测试ID与数据集名称一致性（修正：使用 args.dataset 而非 args.computer）
    expected_id_prefix = f"{args.dataset}_REF_"  # 匹配自定义ID格式（数据集_REF_标签_序号）
    if not args.test_id.startswith(expected_id_prefix):
        raise ValueError(f"测试样本ID前缀必须为 {expected_id_prefix}（例如 {expected_id_prefix}0_01）")

    # 你的自定义参考样本ID配置（不变）
    MANUAL_IDS = {
        0: [18, 14, 40, 85, 86],  # 标签0的参考样本索引
        1: [139, 125, 129, 132, 246]  # 标签1的参考样本索引
    }

    # 执行相似样本查找（参数引用不变，因为已修正 args.test_id 和 args.dataset）
    try:
        results = find_similar_samples_by_id(
            test_sample_id=args.test_id,
            dataset_name=args.dataset,
            samples_per_label=args.samples_per_label,
            min_similarity=args.min_similarity,
            nperseg=args.nperseg,
            energy_threshold=args.energy_threshold,
            manual_ref_ids=MANUAL_IDS if args.use_manual_ref else None  # 启用自定义ID
        )

        # 打印结果
        print_results(results)

        # 可选：保存结果到JSON文件
        import json
        output_file = f"similarity_results_{args.test_id}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
        print(f"\n结果已保存到文件：{output_file}")

    except Exception as e:
        print(f"执行出错：{str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()