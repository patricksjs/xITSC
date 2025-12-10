import numpy as np
import os
from scipy.spatial.distance import euclidean
import pandas as pd


def find_most_similar_shap_matrix_three_class(test_sample_idx, dataset_name, output_path="./t",
                                              use_sliced=True, label=None, exclude_self=True,
                                              top_k_candidates=10):
    """
    使用三分类两阶段方法查找与指定测试样本最相似的SHAP矩阵：
    1. 第一阶段：将SHAP矩阵三值化（>0为1，<0为0，==0为0.5），找出重叠区域最多的前k个候选样本
    2. 第二阶段：对候选样本使用原始SHAP矩阵计算欧氏距离，找出最相似的样本

    Args:
        test_sample_idx: 测试样本的索引
        dataset_name: 数据集名称
        output_path: SHAP矩阵保存的路径
        use_sliced: 是否使用切片后的SHAP矩阵（默认True），False则使用完整矩阵
        label: 指定标签（可选），如果为None则搜索所有标签
        exclude_self: 是否排除测试样本自身
        top_k_candidates: 第一阶段选择的候选样本数量

    Returns:
        most_similar_idx: 最相似样本的索引
        similarity_score: 相似度分数（欧氏距离的倒数，越大越相似）
        overlap_scores: 所有样本的重叠区域分数
        euclidean_scores: 所有样本的欧氏距离分数
        all_overlap_info: 所有样本的重叠区域详细信息
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

    # 测试样本的三值化矩阵（>0为1，<0为0，==0为0.5）
    test_ternary_matrix = np.zeros_like(test_shap_matrix)
    test_ternary_matrix[test_shap_matrix > 0] = 1
    test_ternary_matrix[test_shap_matrix < 0] = 0
    test_ternary_matrix[test_shap_matrix == 0] = 0.5

    test_shap_flat = test_shap_matrix.flatten()
    test_ternary_flat = test_ternary_matrix.flatten()

    # 统计三值的分布
    positive_count = np.sum(test_shap_matrix > 0)
    negative_count = np.sum(test_shap_matrix < 0)
    zero_count = np.sum(test_shap_matrix == 0)
    total_count = test_shap_matrix.size

    print(f"测试样本三值分布: 正值({positive_count}/{total_count}, {positive_count / total_count:.2%}), "
          f"负值({negative_count}/{total_count}, {negative_count / total_count:.2%}), "
          f"零值({zero_count}/{total_count}, {zero_count / total_count:.2%})")

    # 记录测试样本路径，用于后续加载原始矩阵
    test_shap_path = test_shap_paths[0]

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
                # 检查是否与使用的矩阵类型一致
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

    # 3. 第一阶段：计算三值化矩阵的重叠区域
    overlap_scores = {}
    all_overlap_info = {}
    total_elements = len(test_ternary_flat)

    for sample_idx, file_path in all_samples.items():
        try:
            # 加载其他样本的SHAP矩阵并三值化
            other_shap_matrix = np.load(file_path)
            other_ternary_matrix = np.zeros_like(other_shap_matrix)
            other_ternary_matrix[other_shap_matrix > 0] = 1
            other_ternary_matrix[other_shap_matrix < 0] = 0
            other_ternary_matrix[other_shap_matrix == 0] = 0.5

            other_ternary_flat = other_ternary_matrix.flatten()

            # 确保两个向量长度相同
            min_len = min(len(test_ternary_flat), len(other_ternary_flat))
            test_vec = test_ternary_flat[:min_len]
            other_vec = other_ternary_flat[:min_len]

            # 计算重叠区域（相同位置数值相同的数量）
            overlap_count = np.sum(test_vec == other_vec)
            overlap_ratio = overlap_count / min_len

            # 计算各分类的重叠情况
            positive_overlap = np.sum((test_vec == 1) & (other_vec == 1))
            negative_overlap = np.sum((test_vec == 0) & (other_vec == 0))
            zero_overlap = np.sum((test_vec == 0.5) & (other_vec == 0.5))

            overlap_scores[sample_idx] = overlap_ratio
            all_overlap_info[sample_idx] = {
                'overlap_count': overlap_count,
                'overlap_ratio': overlap_ratio,
                'positive_overlap': positive_overlap,
                'negative_overlap': negative_overlap,
                'zero_overlap': zero_overlap,
                'total_elements': min_len,
                'file_path': file_path
            }

        except Exception as e:
            print(f"处理样本 {sample_idx} 时出错: {e}")
            continue

    # 如果没有有效的重叠分数，返回错误
    if not overlap_scores:
        raise ValueError("没有找到任何有效的重叠区域计算结果")

    # 按重叠比例排序，选择前top_k_candidates个候选样本
    sorted_overlaps = sorted(overlap_scores.items(), key=lambda x: x[1], reverse=True)
    top_candidates = sorted_overlaps[:top_k_candidates]

    print(f"\n第一阶段结果（基于三值重叠区域）:")
    print(f"从 {len(overlap_scores)} 个样本中选择了前 {top_k_candidates} 个候选样本")
    print(f"前3个候选样本的重叠情况:")
    for i, (idx, overlap_score) in enumerate(top_candidates[:3]):
        info = all_overlap_info[idx]
        print(f"  样本 {idx}: 重叠比例={overlap_score:.4f}, "
              f"正值重叠={info['positive_overlap']}, 负值重叠={info['negative_overlap']}, "
              f"零值重叠={info['zero_overlap']}")

    # 4. 第二阶段：对候选样本计算欧氏距离
    euclidean_scores = {}

    for sample_idx, overlap_score in top_candidates:
        try:
            # 重新加载候选样本的原始SHAP矩阵（确保是原始数值）
            file_path = all_samples[sample_idx]
            other_shap_matrix = np.load(file_path)
            other_shap_flat = other_shap_matrix.flatten()

            # 确保两个向量长度相同
            min_len = min(len(test_shap_flat), len(other_shap_flat))
            test_vec = test_shap_flat[:min_len]
            other_vec = other_shap_flat[:min_len]

            # 计算欧氏距离
            euclidean_dist = euclidean(test_vec, other_vec)

            # 将欧氏距离转换为相似度分数（距离越小越相似）
            # 使用1/(1+距离)确保分数在0-1之间
            euclidean_sim = 1 / (1 + euclidean_dist)

            # 获取三值重叠的详细信息
            overlap_info = all_overlap_info[sample_idx]

            euclidean_scores[sample_idx] = {
                'euclidean_distance': euclidean_dist,
                'euclidean_similarity': euclidean_sim,
                'overlap_ratio': overlap_score,
                'positive_overlap': overlap_info['positive_overlap'],
                'negative_overlap': overlap_info['negative_overlap'],
                'zero_overlap': overlap_info['zero_overlap']
            }

        except Exception as e:
            print(f"计算候选样本 {sample_idx} 的欧氏距离时出错: {e}")
            continue

    # 如果没有有效的欧氏距离分数，返回错误
    if not euclidean_scores:
        raise ValueError("没有找到任何有效的欧氏距离计算结果")

    # 按欧氏距离相似度排序，找出最相似的样本
    sorted_euclidean = sorted(euclidean_scores.items(),
                              key=lambda x: x[1]['euclidean_similarity'],
                              reverse=True)

    # 获取最相似的样本
    most_similar_idx, similarity_info = sorted_euclidean[0]
    similarity_score = similarity_info['euclidean_similarity']

    print(f"\n第二阶段结果（基于欧氏距离）:")
    print(f"测试样本 {test_sample_idx} 的最相似样本: {most_similar_idx}")
    print(f"欧氏距离相似度: {similarity_score:.4f}")
    print(f"欧氏距离: {similarity_info['euclidean_distance']:.4f}")
    print(f"重叠比例: {similarity_info['overlap_ratio']:.4f}")
    print(f"正值重叠: {similarity_info['positive_overlap']}")
    print(f"负值重叠: {similarity_info['negative_overlap']}")
    print(f"零值重叠: {similarity_info['zero_overlap']}")

    # 打印前5个最相似的候选样本
    print(f"\n前{min(5, top_k_candidates)}个候选样本的详细信息:")
    for i, (idx, info) in enumerate(sorted_euclidean[:min(5, top_k_candidates)]):
        print(f"  {i + 1}. 样本 {idx}: "
              f"欧氏相似度={info['euclidean_similarity']:.4f}, "
              f"重叠比例={info['overlap_ratio']:.4f}, "
              f"距离={info['euclidean_distance']:.4f}, "
              f"正值重叠={info['positive_overlap']}, "
              f"负值重叠={info['negative_overlap']}, "
              f"零值重叠={info['zero_overlap']}")

    return most_similar_idx, similarity_score, overlap_scores, euclidean_scores, all_overlap_info


def batch_find_similar_samples_three_class(test_indices, dataset_name, output_path="./t",
                                           use_sliced=True, top_k_candidates=10, top_k_results=3):
    """
    批量查找多个测试样本的最相似样本（使用三分类两阶段方法）

    Args:
        test_indices: 测试样本索引列表
        dataset_name: 数据集名称
        output_path: SHAP矩阵保存路径
        use_sliced: 是否使用切片矩阵
        top_k_candidates: 第一阶段选择的候选样本数量
        top_k_results: 返回前K个最相似的样本

    Returns:
        results: 包含每个测试样本结果的字典
    """
    results = {}

    for test_idx in test_indices:
        print(f"\n{'=' * 70}")
        print(f"处理测试样本 {test_idx} (三分类两阶段方法)")
        print(f"{'=' * 70}")

        try:
            # 查找最相似的样本
            most_similar_idx, similarity_score, overlap_scores, euclidean_scores, all_overlap_info = find_most_similar_shap_matrix_three_class(
                test_sample_idx=test_idx,
                dataset_name=dataset_name,
                output_path=output_path,
                use_sliced=use_sliced,
                exclude_self=True,
                top_k_candidates=top_k_candidates
            )

            # 获取前top_k_results个相似的样本（基于欧氏距离）
            sorted_euclidean = sorted(euclidean_scores.items(),
                                      key=lambda x: x[1]['euclidean_similarity'],
                                      reverse=True)
            top_matches = [(idx, info['euclidean_similarity'], info['overlap_ratio'],
                            info['positive_overlap'], info['negative_overlap'], info['zero_overlap'])
                           for idx, info in sorted_euclidean[:top_k_results]]

            # 获取重叠区域最多的前top_k_results个样本（基于第一阶段）
            sorted_overlaps = sorted(overlap_scores.items(), key=lambda x: x[1], reverse=True)
            top_overlaps = [(idx, ratio,
                             all_overlap_info[idx]['positive_overlap'],
                             all_overlap_info[idx]['negative_overlap'],
                             all_overlap_info[idx]['zero_overlap'])
                            for idx, ratio in sorted_overlaps[:top_k_results]]

            results[test_idx] = {
                'most_similar': most_similar_idx,
                'euclidean_similarity': similarity_score,
                'euclidean_distance': euclidean_scores[most_similar_idx]['euclidean_distance'],
                'overlap_ratio': euclidean_scores[most_similar_idx]['overlap_ratio'],
                'positive_overlap': euclidean_scores[most_similar_idx]['positive_overlap'],
                'negative_overlap': euclidean_scores[most_similar_idx]['negative_overlap'],
                'zero_overlap': euclidean_scores[most_similar_idx]['zero_overlap'],
                'top_matches': top_matches,
                'top_overlaps': top_overlaps,
                'euclidean_scores': euclidean_scores,
                'overlap_scores': overlap_scores,
                'all_overlap_info': all_overlap_info
            }

        except Exception as e:
            print(f"处理测试样本 {test_idx} 时出错: {e}")
            results[test_idx] = {'error': str(e)}

    # 打印汇总结果
    print("\n" + "=" * 80)
    print("批量处理结果汇总（三分类两阶段方法）")
    print("=" * 80)

    successful_results = {k: v for k, v in results.items() if 'error' not in v}

    for test_idx, result in successful_results.items():
        print(f"\n测试样本 {test_idx}:")
        print(f"  最相似样本: {result['most_similar']}")
        print(f"  欧氏相似度: {result['euclidean_similarity']:.4f}")
        print(f"  欧氏距离: {result['euclidean_distance']:.4f}")
        print(f"  重叠比例: {result['overlap_ratio']:.4f}")
        print(
            f"  正值重叠: {result['positive_overlap']}, 负值重叠: {result['negative_overlap']}, 零值重叠: {result['zero_overlap']}")

        print(f"  前{top_k_results}个匹配样本（基于两阶段）:")
        for i, (idx, euclidean_sim, overlap_ratio, pos_overlap, neg_overlap, zero_overlap) in enumerate(
                result['top_matches']):
            print(f"    {i + 1}. 样本 {idx}: 欧氏相似度={euclidean_sim:.4f}, 重叠比例={overlap_ratio:.4f}, "
                  f"正值重叠={pos_overlap}, 负值重叠={neg_overlap}, 零值重叠={zero_overlap}")

        print(f"  前{top_k_results}个重叠样本（仅第一阶段）:")
        for i, (idx, overlap_ratio, pos_overlap, neg_overlap, zero_overlap) in enumerate(result['top_overlaps']):
            print(f"    {i + 1}. 样本 {idx}: 重叠比例={overlap_ratio:.4f}, "
                  f"正值重叠={pos_overlap}, 负值重叠={neg_overlap}, 零值重叠={zero_overlap}")

    return results


def save_similarity_results_three_class(results, dataset_name, output_path="./similarity_results_three_class"):
    """
    保存三分类两阶段相似度分析结果

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
                'euclidean_similarity': 'Error',
                'euclidean_distance': 'Error',
                'overlap_ratio': 'Error',
                'positive_overlap': 'Error',
                'negative_overlap': 'Error',
                'zero_overlap': 'Error',
                'error': result['error']
            })
        else:
            # 准备两阶段匹配信息
            match_strs = []
            for idx, euclidean_sim, overlap_ratio, pos_overlap, neg_overlap, zero_overlap in result.get('top_matches',
                                                                                                        []):
                match_strs.append(
                    f"{idx}(E:{euclidean_sim:.3f}/O:{overlap_ratio:.3f}/P:{pos_overlap}/N:{neg_overlap}/Z:{zero_overlap})")

            # 准备第一阶段匹配信息
            overlap_strs = []
            for idx, overlap_ratio, pos_overlap, neg_overlap, zero_overlap in result.get('top_overlaps', []):
                overlap_strs.append(f"{idx}(O:{overlap_ratio:.3f}/P:{pos_overlap}/N:{neg_overlap}/Z:{zero_overlap})")

            csv_data.append({
                'test_sample': test_idx,
                'most_similar': result['most_similar'],
                'euclidean_similarity': f"{result['euclidean_similarity']:.4f}",
                'euclidean_distance': f"{result['euclidean_distance']:.4f}",
                'overlap_ratio': f"{result['overlap_ratio']:.4f}",
                'positive_overlap': result['positive_overlap'],
                'negative_overlap': result['negative_overlap'],
                'zero_overlap': result['zero_overlap'],
                'top_matches': '; '.join(match_strs),
                'top_overlaps': '; '.join(overlap_strs)
            })

    df = pd.DataFrame(csv_data)
    csv_file = f"{output_path}/{dataset_name}_three_class_similarity_results.csv"
    df.to_csv(csv_file, index=False)
    print(f"结果已保存到: {csv_file}")

    # 保存详细结果
    detailed_file = f"{output_path}/{dataset_name}_three_class_detailed_results.txt"
    with open(detailed_file, 'w', encoding='utf-8') as f:
        f.write(f"数据集: {dataset_name}\n")
        f.write(f"测试样本数: {len(results)}\n")
        f.write(f"三分类两阶段相似度分析方法\n")
        f.write("=" * 100 + "\n\n")

        for test_idx, result in results.items():
            f.write(f"测试样本 {test_idx}:\n")
            if 'error' in result:
                f.write(f"  错误: {result['error']}\n")
            else:
                f.write(f"  最相似样本: {result['most_similar']}\n")
                f.write(f"  欧氏相似度: {result['euclidean_similarity']:.4f}\n")
                f.write(f"  欧氏距离: {result['euclidean_distance']:.4f}\n")
                f.write(f"  重叠比例: {result['overlap_ratio']:.4f}\n")
                f.write(
                    f"  正值重叠: {result['positive_overlap']}, 负值重叠: {result['negative_overlap']}, 零值重叠: {result['zero_overlap']}\n")
                f.write(f"  前5个匹配样本（两阶段）:\n")
                for i, (idx, euclidean_sim, overlap_ratio, pos_overlap, neg_overlap, zero_overlap) in enumerate(
                        result.get('top_matches', [])[:5]):
                    f.write(f"    {i + 1}. 样本 {idx}: 欧氏相似度={euclidean_sim:.4f}, 重叠比例={overlap_ratio:.4f}, "
                            f"正值重叠={pos_overlap}, 负值重叠={neg_overlap}, 零值重叠={zero_overlap}\n")
                f.write(f"  前5个重叠样本（仅第一阶段）:\n")
                for i, (idx, overlap_ratio, pos_overlap, neg_overlap, zero_overlap) in enumerate(
                        result.get('top_overlaps', [])[:5]):
                    f.write(f"    {i + 1}. 样本 {idx}: 重叠比例={overlap_ratio:.4f}, "
                            f"正值重叠={pos_overlap}, 负值重叠={neg_overlap}, 零值重叠={zero_overlap}\n")
            f.write("\n")

    print(f"详细结果已保存到: {detailed_file}")

    return csv_file, detailed_file


# 使用示例
if __name__ == "__main__":
    # 示例1: 查找单个样本的最相似样本（三分类两阶段方法）
    test_sample_idx = 74  # 测试样本索引
    dataset_name = "computer"  # 数据集名称
    output_path = "./t"  # SHAP矩阵保存路径

    print("=" * 80)
    print("三分类两阶段相似度查找示例")
    print("=" * 80)

    try:
        most_similar_idx, similarity_score, overlap_scores, euclidean_scores, all_overlap_info = find_most_similar_shap_matrix_three_class(
            test_sample_idx=test_sample_idx,
            dataset_name=dataset_name,
            output_path=output_path,
            use_sliced=True,  # 使用切片矩阵
            label=None,  # 不指定标签，自动搜索
            exclude_self=True,
            top_k_candidates=10  # 第一阶段选择10个候选样本
        )

        print(f"\n最终结果:")
        print(f"测试样本 {test_sample_idx} 的最相似样本是 {most_similar_idx}")
        print(f"欧氏距离相似度: {similarity_score:.4f}")

    except Exception as e:
        print(f"错误: {e}")

    # 示例2: 批量查找多个样本的最相似样本（三分类两阶段方法）
    test_indices = [0, 1, 2, 3, 4]  # 多个测试样本
    print("\n" + "=" * 80)
    print("批量查找示例（三分类两阶段方法）")
    print("=" * 80)

    try:
        batch_results = batch_find_similar_samples_three_class(
            test_indices=test_indices,
            dataset_name=dataset_name,
            output_path=output_path,
            use_sliced=True,
            top_k_candidates=10,
            top_k_results=1
        )

        # 保存结果
        csv_file, detailed_file = save_similarity_results_three_class(
            results=batch_results,
            dataset_name=dataset_name,
            output_path="./similarity_results_three_class"
        )

    except Exception as e:
        print(f"批量处理错误: {e}")