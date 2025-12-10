import numpy as np
import os
from scipy.spatial.distance import cosine
import pandas as pd


def find_most_similar_shap_matrix(test_sample_idx, dataset_name, output_path="./t",
                                  use_sliced=True, label=None, exclude_self=True):
    """
    查找与指定测试样本最相似的SHAP矩阵

    Args:
        test_sample_idx: 测试样本的索引
        dataset_name: 数据集名称
        output_path: SHAP矩阵保存的路径
        use_sliced: 是否使用切片后的SHAP矩阵（默认True），False则使用完整矩阵
        label: 指定标签（可选），如果为None则搜索所有标签
        exclude_self: 是否排除测试样本自身

    Returns:
        most_similar_idx: 最相似样本的索引
        similarity_score: 相似度分数（余弦相似度）
        all_similarities: 所有样本的相似度字典
    """

    # 1. 加载测试样本的SHAP矩阵
    test_shap_paths = []
    shap_matrix_dir = ""

    if label is not None:
        # 如果指定了标签，直接在指定标签目录中查找
        shap_matrix_dir = f"{output_path}/shap_matrices/{dataset_name}_{label}"
        test_shap_path = f"{shap_matrix_dir}/shap_matrix_{'sliced' if use_sliced else 'full'}_sample{test_sample_idx}.npy"
        test_shap_paths.append(test_shap_path)
    else:
        # 如果没有指定标签，遍历所有标签目录查找测试样本
        base_shap_dir = f"{output_path}/shap_matrices/"
        if not os.path.exists(base_shap_dir):
            raise ValueError(f"SHAP矩阵目录不存在: {base_shap_dir}")

        # 遍历所有可能的标签目录
        for dir_name in os.listdir(base_shap_dir):
            if dir_name.startswith(f"{dataset_name}_"):
                current_dir = os.path.join(base_shap_dir, dir_name)
                test_shap_path = os.path.join(current_dir,
                                              f"shap_matrix_{'sliced' if use_sliced else 'full'}_sample{test_sample_idx}.npy")
                if os.path.exists(test_shap_path):
                    test_shap_paths.append(test_shap_path)
                    shap_matrix_dir = current_dir
                    break

    if not test_shap_paths:
        # 如果没找到，尝试搜索所有可能的样本文件
        print("警告：未找到测试样本的SHAP矩阵，尝试搜索所有样本...")
        base_shap_dir = f"{output_path}/shap_matrices/"
        for dir_name in os.listdir(base_shap_dir):
            if dir_name.startswith(f"{dataset_name}_"):
                current_dir = os.path.join(base_shap_dir, dir_name)
                for file_name in os.listdir(current_dir):
                    if f"_sample{test_sample_idx}.npy" in file_name:
                        test_shap_paths.append(os.path.join(current_dir, file_name))
                        shap_matrix_dir = current_dir
                        print(f"找到测试样本文件: {file_name}")
                        break

    if not test_shap_paths:
        raise FileNotFoundError(f"未找到测试样本 {test_sample_idx} 的SHAP矩阵")

    # 加载测试样本的SHAP矩阵（取第一个找到的）
    test_shap_matrix = np.load(test_shap_paths[0])
    print(f"测试样本 {test_sample_idx} 的SHAP矩阵形状: {test_shap_matrix.shape}")

    # 2. 获取所有样本的SHAP矩阵文件
    all_samples = {}

    # 如果指定了标签，只在该标签目录中搜索
    if label is not None:
        search_dirs = [f"{output_path}/shap_matrices/{dataset_name}_{label}"]
    else:
        # 否则搜索所有标签目录
        search_dirs = []
        base_shap_dir = f"{output_path}/shap_matrices/"
        for dir_name in os.listdir(base_shap_dir):
            if dir_name.startswith(f"{dataset_name}_"):
                search_dirs.append(os.path.join(base_shap_dir, dir_name))

    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue

        for file_name in os.listdir(search_dir):
            if file_name.endswith('.npy'):
                # 提取样本索引
                if 'full' in file_name and use_sliced:
                    continue
                if 'sliced' in file_name and not use_sliced:
                    continue

                # 从文件名中提取样本索引
                try:
                    # 处理文件名格式: shap_matrix_sliced_sample{idx}.npy 或 shap_matrix_full_sample{idx}.npy
                    parts = file_name.split('_')
                    for part in parts:
                        if part.startswith('sample'):
                            idx_str = part.replace('sample', '').replace('.npy', '')
                            sample_idx = int(idx_str)

                            if exclude_self and sample_idx == test_sample_idx:
                                continue

                            file_path = os.path.join(search_dir, file_name)
                            all_samples[sample_idx] = file_path
                            break
                except (ValueError, IndexError) as e:
                    print(f"无法解析文件名 {file_name}: {e}")
                    continue

    print(f"找到 {len(all_samples)} 个其他样本的SHAP矩阵")

    # 3. 计算余弦相似度
    similarities = {}
    test_shap_flat = test_shap_matrix.flatten()

    for sample_idx, file_path in all_samples.items():
        try:
            # 加载其他样本的SHAP矩阵
            other_shap_matrix = np.load(file_path)
            other_shap_flat = other_shap_matrix.flatten()

            # 确保两个向量长度相同（如果不同，进行填充或截断）
            min_len = min(len(test_shap_flat), len(other_shap_flat))
            test_vec = test_shap_flat[:min_len]
            other_vec = other_shap_flat[:min_len]

            # 计算余弦相似度（1 - 余弦距离）
            cos_dist = cosine(test_vec, other_vec)
            cos_similarity = 1 - cos_dist

            similarities[sample_idx] = cos_similarity

        except Exception as e:
            print(f"处理样本 {sample_idx} 时出错: {e}")
            continue

    # 4. 找到最相似的样本
    if not similarities:
        raise ValueError("没有找到任何有效的相似度计算结果")

    # 按相似度排序
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    # 获取最相似的样本
    most_similar_idx, max_similarity = sorted_similarities[0]

    print(f"\n测试样本 {test_sample_idx} 的结果:")
    print(f"最相似的样本: {most_similar_idx}")
    print(f"相似度分数: {max_similarity:.4f}")

    # 打印前5个最相似的样本
    print("\n前5个最相似的样本:")
    for i, (idx, sim) in enumerate(sorted_similarities[:5]):
        print(f"  {i + 1}. 样本 {idx}: 相似度 = {sim:.4f}")

    return most_similar_idx, max_similarity, similarities


def batch_find_similar_samples(test_indices, dataset_name, output_path="./t",
                               use_sliced=True, top_k=3):
    """
    批量查找多个测试样本的最相似样本

    Args:
        test_indices: 测试样本索引列表
        dataset_name: 数据集名称
        output_path: SHAP矩阵保存路径
        use_sliced: 是否使用切片矩阵
        top_k: 返回前K个最相似的样本

    Returns:
        results: 包含每个测试样本结果的字典
    """
    results = {}

    for test_idx in test_indices:
        print(f"\n{'=' * 50}")
        print(f"处理测试样本 {test_idx}")
        print(f"{'=' * 50}")

        try:
            # 查找最相似的样本
            most_similar_idx, max_similarity, all_similarities = find_most_similar_shap_matrix(
                test_sample_idx=test_idx,
                dataset_name=dataset_name,
                output_path=output_path,
                use_sliced=use_sliced,
                exclude_self=True
            )

            # 获取前top_k个相似的样本
            sorted_sims = sorted(all_similarities.items(), key=lambda x: x[1], reverse=True)
            top_matches = [(idx, sim) for idx, sim in sorted_sims[:top_k]]

            results[test_idx] = {
                'most_similar': most_similar_idx,
                'max_similarity': max_similarity,
                'top_matches': top_matches,
                'all_similarities': all_similarities
            }

        except Exception as e:
            print(f"处理测试样本 {test_idx} 时出错: {e}")
            results[test_idx] = {'error': str(e)}

    # 打印汇总结果
    print("\n" + "=" * 60)
    print("批量处理结果汇总")
    print("=" * 60)

    successful_results = {k: v for k, v in results.items() if 'error' not in v}

    for test_idx, result in successful_results.items():
        print(f"\n测试样本 {test_idx}:")
        print(f"  最相似样本: {result['most_similar']} (相似度: {result['max_similarity']:.4f})")
        print(f"  前{top_k}个匹配样本:")
        for i, (idx, sim) in enumerate(result['top_matches']):
            print(f"    {i + 1}. 样本 {idx}: {sim:.4f}")

    return results


def save_similarity_results(results, dataset_name, output_path="./similarity_results"):
    """
    保存相似度分析结果

    Args:
        results: 相似度分析结果字典
        dataset_name: 数据集名称
        output_path: 输出路径
    """
    os.makedirs(output_path, exist_ok=True)

    # 保存为CSV文件
    csv_data = []
    for test_idx, result in results.items():
        if 'error' in result:
            csv_data.append({
                'test_sample': test_idx,
                'most_similar': 'Error',
                'similarity': 'Error',
                'error': result['error']
            })
        else:
            csv_data.append({
                'test_sample': test_idx,
                'most_similar': result['most_similar'],
                'similarity': f"{result['max_similarity']:.4f}"
            })

    df = pd.DataFrame(csv_data)
    csv_file = f"{output_path}/{dataset_name}_similarity_results.csv"
    df.to_csv(csv_file, index=False)
    print(f"结果已保存到: {csv_file}")

    # 保存详细结果
    detailed_file = f"{output_path}/{dataset_name}_detailed_results.txt"
    with open(detailed_file, 'w', encoding='utf-8') as f:
        f.write(f"数据集: {dataset_name}\n")
        f.write(f"测试样本数: {len(results)}\n")
        f.write("=" * 60 + "\n\n")

        for test_idx, result in results.items():
            f.write(f"测试样本 {test_idx}:\n")
            if 'error' in result:
                f.write(f"  错误: {result['error']}\n")
            else:
                f.write(f"  最相似样本: {result['most_similar']} (相似度: {result['max_similarity']:.4f})\n")
                f.write(f"  前5个匹配样本:\n")
                for i, (idx, sim) in enumerate(result.get('top_matches', [])[:5]):
                    f.write(f"    {i + 1}. 样本 {idx}: {sim:.4f}\n")
            f.write("\n")

    print(f"详细结果已保存到: {detailed_file}")

    return csv_file, detailed_file




# 使用示例
if __name__ == "__main__":
    # 示例1: 查找单个样本的最相似样本
    test_sample_idx = 74  # 测试样本索引
    dataset_name = "computer"  # 数据集名称
    output_path = "./t"  # SHAP矩阵保存路径

    try:
        most_similar_idx, similarity_score, all_similarities = find_most_similar_shap_matrix(
            test_sample_idx=test_sample_idx,
            dataset_name=dataset_name,
            output_path=output_path,
            use_sliced=True,  # 使用切片矩阵
            label=None,  # 不指定标签，自动搜索
            exclude_self=True
        )

        print(f"\n最终结果:")
        print(f"测试样本 {test_sample_idx} 的最相似样本是 {most_similar_idx}")
        print(f"余弦相似度: {similarity_score:.4f}")

    except Exception as e:
        print(f"错误: {e}")

    # 示例2: 批量查找多个样本的最相似样本
    test_indices = [0, 1, 2, 3, 4]  # 多个测试样本
    print("\n" + "=" * 60)
    print("批量查找示例")
    print("=" * 60)

    try:
        batch_results = batch_find_similar_samples(
            test_indices=test_indices,
            dataset_name=dataset_name,
            output_path=output_path,
            use_sliced=True,
            top_k=3
        )

        # 保存结果
        csv_file, detailed_file = save_similarity_results(
            results=batch_results,
            dataset_name=dataset_name,
            output_path="./similarity_results"
        )



    except Exception as e:
        print(f"批量处理错误: {e}")