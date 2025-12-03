import argparse
import numpy as np
import torch
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean as scipy_euclidean
import os
import json
import config

def calculate_dtw_distance(sample1, sample2):
    """计算两个样本之间的DTW距离"""
    dtw_distance, _ = fastdtw(sample1.reshape(-1, 1), sample2.reshape(-1, 1), dist=scipy_euclidean)
    return dtw_distance


def find_closest_samples(test_sample_data, reference_samples_data, manual_ids):
    """
    找到待分类样本在每个label中最接近的两个参考样本

    Args:
        test_sample_data: 待分类样本的数据
        reference_samples_data: 所有参考样本的数据
        manual_ids: 参考样本的ID字典 {label: [sample_ids]}

    Returns:
        closest_samples: 每个label中最接近的两个样本ID {label: [closest_sample_id1, closest_sample_id2]}
    """
    closest_samples = {}

    # 遍历每个label
    for label, sample_ids in manual_ids.items():
        distances = []

        # 计算待分类样本与该label中所有参考样本的距离
        for sample_id in sample_ids:
            if sample_id < len(reference_samples_data):
                ref_sample_data = reference_samples_data[sample_id]
                distance = calculate_dtw_distance(test_sample_data, ref_sample_data)
                distances.append((sample_id, distance))

        # 按距离排序，取最近的两个样本
        distances.sort(key=lambda x: x[1])
        closest_samples[label] = [sample_id for sample_id, _ in distances[:2]]

    return closest_samples


def get_sample_images_by_id(sample_ids, label, dataset_name, root_folder):
    """
    根据样本ID获取对应的三类图片路径

    Args:
        sample_ids: 样本ID列表
        label: 标签
        dataset_name: 数据集名称
        root_folder: 根文件夹路径

    Returns:
        image_paths: 图片路径字典 {sample_id: {"line_chart": path, "spectrogram": path, "heatmap": path}}
    """
    image_paths = {}

    for sample_id in sample_ids:
        # 构建图片路径
        line_path = os.path.join(root_folder, f"{dataset_name}_{label}", f"plot_sample{sample_id}.png")
        spectrogram_path = os.path.join(root_folder, f"{dataset_name}_{label}", f"STFT_sample{sample_id}.png")
        heatmap_path = os.path.join(root_folder, f"{dataset_name}_{label}", f"shap_sample{sample_id}.png")

        # 检查文件是否存在
        if not os.path.exists(line_path):
            line_path = os.path.join(root_folder, f"{dataset_name}_{label}", f"plot_sample{sample_id}.png")
        if not os.path.exists(spectrogram_path):
            spectrogram_path = os.path.join(root_folder, f"{dataset_name}_{label}", f"STFT_sample{sample_id}.png")
        if not os.path.exists(heatmap_path):
            heatmap_path = os.path.join(root_folder, f"{dataset_name}_{label}", f"shap_sample{sample_id}.png")

        image_paths[sample_id] = {
            "line_chart": line_path if os.path.exists(line_path) else None,
            "spectrogram": spectrogram_path if os.path.exists(spectrogram_path) else None,
            "heatmap": heatmap_path if os.path.exists(heatmap_path) else None
        }

    return image_paths


def process_test_sample_with_dtw(test_sample_id, data, labels, manual_ids, dataset_name, root_folder):
    """
    处理单个测试样本，找到每个label中最接近的两个参考样本

    Args:
        test_sample_id: 待分类样本ID
        data: 所有样本数据
        labels: 所有样本标签
        manual_ids: 参考样本ID字典
        dataset_name: 数据集名称
        root_folder: 根文件夹路径

    Returns:
        result: 包含最近样本信息和图片路径的结果
    """
    # 获取待分类样本数据
    if test_sample_id >= len(data):
        print(f"错误：样本ID {test_sample_id} 超出数据范围")
        return None

    test_sample_data = data[test_sample_id]

    # 找到每个label中最接近的两个样本
    closest_samples = find_closest_samples(test_sample_data, data, manual_ids)

    # 获取这些样本的图片路径
    closest_samples_images = {}
    for label, sample_ids in closest_samples.items():
        closest_samples_images[label] = get_sample_images_by_id(sample_ids, label, dataset_name, root_folder)

    result = {
        "test_sample_id": test_sample_id,
        "closest_samples": closest_samples,
        "closest_samples_images": closest_samples_images
    }

    return result


def main():
    parser = argparse.ArgumentParser(description='根据待分类样本ID找到最接近的参考样本')
    parser.add_argument('--data_name', type=str, default='computer',
                        choices=['computer', 'cincecgtorso'], help='数据集名称')
    parser.add_argument('--test_sample_ids', type=str, required=True,
                        help='待分类样本ID列表，用逗号分隔，例如: 18,25,30')

    args = parser.parse_args()

    try:
        from data.data_loader import load_data
    except ImportError:
        print("错误：无法导入 load_data 函数。请检查你的项目结构和导入路径。")
        return

    # 加载数据
    data, labels = load_data(args.data_name)

    # 解析待分类样本ID
    test_sample_ids = [int(x.strip()) for x in args.test_sample_ids.split(',')]

    # 定义参考样本ID（从config中获取）
    manual_ids = config.MANUAL_IDS

    print(f"正在处理 {len(test_sample_ids)} 个待分类样本...")

    results = {}
    for test_sample_id in test_sample_ids:
        print(f"\n处理待分类样本 ID: {test_sample_id}")

        result = process_test_sample_with_dtw(
            test_sample_id, data, labels, manual_ids,
            args.data_name, r"C:\Users\34517\Desktop\zuhui\xITSC\data\image1"
        )

        if result:
            results[test_sample_id] = result

            # 打印结果
            print(f"样本 {test_sample_id} 的最接近参考样本:")
            for label, sample_ids in result["closest_samples"].items():
                print(f"  标签 {label}: 样本 {sample_ids}")

    # 保存结果到文件
    output_file = "closest_samples_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\n结果已保存至: {output_file}")
    return results


if __name__ == "__main__":
    results = main()