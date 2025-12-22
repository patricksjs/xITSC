import numpy as np
import os


def find_most_similar_for_each_label(test_sample_idx, label, dataset_name,
                                     test_shap_path=r"D:\zuhui\xITSC\data\image\test",
                                     train_shap_path=r"D:\zuhui\xITSC\data\image\train",
                                     top_k_candidates=10):
    """
    为给定测试样本，从训练集中为每个标签找到最相似的样本

    Args:
        test_sample_idx: 测试样本的索引
        label: 测试样本的标签
        dataset_name: 数据集名称
        test_shap_path: 测试样本SHAP矩阵保存的路径
        train_shap_path: 训练样本SHAP矩阵保存的路径
        top_k_candidates: 第一阶段选择的候选样本数量

    Returns:
        dict: 每个标签对应的最相似训练样本ID，格式为 {label: most_similar_idx}
    """

    def cosine_similarity(vec1, vec2):
        """计算两个向量的余弦相似度"""
        # 避免除零错误
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return np.dot(vec1, vec2) / (norm1 * norm2)

    # 1. 获取训练集中的所有标签
    base_train_shap_dir = train_shap_path
    if not os.path.exists(base_train_shap_dir):
        raise ValueError(f"训练集SHAP矩阵目录不存在: {base_train_shap_dir}")

    # 找出所有标签
    all_labels = []
    for dir_name in os.listdir(base_train_shap_dir):
        if dir_name.startswith(f"{dataset_name}_"):
            # 从目录名中提取标签
            dir_label = dir_name.split("_")[-1]
            all_labels.append(dir_label)

    print(f"在训练集中找到标签: {all_labels}")

    # 2. 加载测试样本的SHAP矩阵（从test文件夹）
    test_shap_dir = os.path.join(test_shap_path, f"{dataset_name}_{label}")
    test_shap_file = f"shap_matrix_sample{test_sample_idx}.npy"
    test_shap_path_full = os.path.join(test_shap_dir, test_shap_file)

    if not os.path.exists(test_shap_path_full):
        # 尝试其他可能的文件名格式
        found = False
        if os.path.exists(test_shap_dir):
            for file_name in os.listdir(test_shap_dir):
                if f"_sample{test_sample_idx}.npy" in file_name or f"sample{test_sample_idx}.npy" in file_name:
                    test_shap_path_full = os.path.join(test_shap_dir, file_name)
                    found = True
                    print(f"找到测试样本文件: {file_name}")
                    break

        if not found:
            raise FileNotFoundError(f"未找到测试样本 {test_sample_idx} 的SHAP矩阵于 {test_shap_dir}")

    test_shap_matrix = np.load(test_shap_path_full)
    print(f"测试样本 {test_sample_idx} (标签: {label}) 的SHAP矩阵形状: {test_shap_matrix.shape}")

    # 测试样本的三值化矩阵
    test_ternary_matrix = np.zeros_like(test_shap_matrix)
    test_ternary_matrix[test_shap_matrix > 0] = 1
    test_ternary_matrix[test_shap_matrix < 0] = 0
    test_ternary_matrix[test_shap_matrix == 0] = 0.5

    test_shap_flat = test_shap_matrix.flatten()
    test_ternary_flat = test_ternary_matrix.flatten()

    # 3. 为每个标签从训练集中找到最相似的样本
    label_to_most_similar = {}

    for current_label in all_labels:
        print(f"\n处理标签: {current_label}")

        # 获取当前标签下的所有训练样本
        current_dir = os.path.join(base_train_shap_dir, f"{dataset_name}_{current_label}")
        if not os.path.exists(current_dir):
            print(f"警告: 标签 {current_label} 的训练集目录不存在，跳过")
            label_to_most_similar[current_label] = None
            continue

        # 收集当前标签下的所有训练样本文件
        sample_files = {}
        for file_name in os.listdir(current_dir):
            if not file_name.endswith('.npy'):
                continue

            # 从文件名中提取样本索引
            try:
                # 处理文件名格式: shap_matrix_sample{idx}.npy
                if file_name.startswith("shap_matrix_sample"):
                    idx_str = file_name.replace("shap_matrix_sample", "").replace(".npy", "")
                    sample_idx = int(idx_str)
                    sample_files[sample_idx] = os.path.join(current_dir, file_name)
                # 处理其他可能的格式
                elif "_sample" in file_name:
                    parts = file_name.split('_')
                    for part in parts:
                        if part.startswith('sample'):
                            idx_str = part.replace('sample', '').replace('.npy', '')
                            sample_idx = int(idx_str)
                            sample_files[sample_idx] = os.path.join(current_dir, file_name)
                            break
            except (ValueError, IndexError) as e:
                print(f"无法解析文件名 {file_name}: {e}")
                continue

        if not sample_files:
            print(f"标签 {current_label} 的训练集下没有找到样本")
            label_to_most_similar[current_label] = None
            continue

        print(f"标签 {current_label} 的训练集下找到 {len(sample_files)} 个样本")

        # 第一阶段：计算三值化矩阵的重叠区域
        overlap_scores = {}

        for sample_idx, file_path in sample_files.items():
            try:
                # 加载训练样本的SHAP矩阵并三值化
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

                overlap_scores[sample_idx] = {
                    'overlap_ratio': overlap_ratio,
                    'file_path': file_path
                }

            except Exception as e:
                print(f"处理训练样本 {sample_idx} 时出错: {e}")
                continue

        if not overlap_scores:
            print(f"标签 {current_label} 的训练集没有有效的重叠区域计算结果")
            label_to_most_similar[current_label] = None
            continue

        # 按重叠比例排序，选择前top_k_candidates个候选样本
        sorted_overlaps = sorted(overlap_scores.items(), key=lambda x: x[1]['overlap_ratio'], reverse=True)
        top_candidates = sorted_overlaps[:min(top_k_candidates, len(sorted_overlaps))]

        # 第二阶段：对候选样本计算余弦相似度
        cosine_scores = {}

        for sample_idx, overlap_info in top_candidates:
            try:
                # 加载候选样本的原始SHAP矩阵
                other_shap_matrix = np.load(overlap_info['file_path'])
                other_shap_flat = other_shap_matrix.flatten()

                # 确保两个向量长度相同
                min_len = min(len(test_shap_flat), len(other_shap_flat))
                test_vec = test_shap_flat[:min_len]
                other_vec = other_shap_flat[:min_len]

                # 计算余弦相似度
                cosine_sim = cosine_similarity(test_vec, other_vec)

                cosine_scores[sample_idx] = {
                    'cosine_similarity': cosine_sim,
                    'overlap_ratio': overlap_info['overlap_ratio']
                }

            except Exception as e:
                print(f"计算候选训练样本 {sample_idx} 的余弦相似度时出错: {e}")
                continue

        if not cosine_scores:
            print(f"标签 {current_label} 的训练集没有有效的余弦相似度计算结果")
            label_to_most_similar[current_label] = None
            continue

        # 按余弦相似度排序，找出最相似的样本（余弦相似度越大越相似）
        sorted_cosine = sorted(cosine_scores.items(),
                               key=lambda x: x[1]['cosine_similarity'],
                               reverse=True)

        # 获取最相似的样本
        most_similar_idx, similarity_info = sorted_cosine[0]

        print(f"标签 {current_label} 的最相似训练样本: {most_similar_idx}")
        print(f"  余弦相似度: {similarity_info['cosine_similarity']:.4f}")
        print(f"  重叠比例: {similarity_info['overlap_ratio']:.4f}")

        label_to_most_similar[current_label] = most_similar_idx

    return label_to_most_similar


# 使用示例
if __name__ == "__main__":
    # 示例：为样本74（标签为0）在每个标签中找到最相似的训练样本
    test_sample_idx = 9
    label = 0  # 测试样本的标签
    dataset_name = "computer"

    print("=" * 80)
    print(f"为测试样本 {test_sample_idx} (标签: {label}) 在每个标签中查找最相似训练样本")
    print("=" * 80)

    try:
        result = find_most_similar_for_each_label(
            test_sample_idx=test_sample_idx,
            label=label,
            dataset_name=dataset_name,
            top_k_candidates=5
        )

        print("\n" + "=" * 80)
        print("最终结果:")
        print("=" * 80)
        for label_id, similar_sample in result.items():
            if similar_sample is None:
                print(f"标签 {label_id}: 未找到相似训练样本")
            else:
                print(f"标签 {label_id}: 最相似训练样本ID = {similar_sample}")

    except Exception as e:
        print(f"错误: {e}")