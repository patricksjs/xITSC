import numpy as np
import torch
from sklearn.cluster import KMeans
from fastdtw import fastdtw
from collections import Counter

# 导入数据加载函数
from data.dataloader import load_test, load_train


def find_similar_samples_with_dtw(test_sample, dataset, labels, k=4):
    """
    使用DTW找到与测试样本最相似的k个样本

    参数:
        test_sample: 测试样本，形状为 (n_features,)
        dataset: 训练数据集，形状为 (n_samples, n_features)
        labels: 训练数据标签，形状为 (n_samples,)
        k: 要找到的相似样本数量，默认为4

    返回:
        similar_samples: 相似样本的(id, label)元组列表
    """
    distances = []

    # 计算测试样本与数据集中所有样本的DTW距离
    for i, sample in enumerate(dataset):
        dist, _ = fastdtw(
            test_sample.flatten().numpy(),
            sample.flatten().numpy(),
            dist=lambda u, v: abs(u - v)
        )
        distances.append((i, dist))

    # 按距离排序，取最近的k个样本
    distances.sort(key=lambda x: x[1])
    similar_samples = [(idx, labels[idx].item()) for idx, _ in distances[:k]]

    return similar_samples


def find_dissimilar_samples_with_clustering(test_sample, dataset, labels, n_clusters=2):
    """
    通过聚类找到与测试样本最不相似的2个样本

    参数:
        test_sample: 测试样本，形状为 (n_features,)
        dataset: 训练数据集，形状为 (n_samples, n_features)
        labels: 训练数据标签，形状为 (n_samples,)
        n_clusters: 聚类数量，默认为2

    返回:
        dissimilar_samples: 不相似样本的(id, label)元组列表
    """
    # 调整聚类数量，确保至少2个簇
    if n_clusters is None:
        n_clusters = min(10, len(dataset) // 10)
        n_clusters = max(2, n_clusters)

    # 将数据转换为numpy数组用于KMeans
    dataset_np = dataset.numpy()

    # 使用KMeans进行聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(dataset_np)

    # 计算测试样本与每个簇中心的欧氏距离
    test_sample_np = test_sample.numpy().reshape(1, -1)
    cluster_centers = kmeans.cluster_centers_

    # 找到最近的簇
    distances_to_centers = []
    for i, center in enumerate(cluster_centers):
        dist = np.linalg.norm(test_sample_np - center.reshape(1, -1))
        distances_to_centers.append((i, dist))

    distances_to_centers.sort(key=lambda x: x[1])
    nearest_cluster = distances_to_centers[0][0]

    # 收集所有其他簇的样本
    other_cluster_samples = []
    for cluster_id in range(n_clusters):
        if cluster_id == nearest_cluster:
            continue
        # 获取当前簇的所有样本
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        other_cluster_samples.extend([(int(idx), dataset[int(idx)]) for idx in cluster_indices])

    # 从其他簇的所有样本中使用DTW距离选择最远的两个样本
    dissimilar_samples = []
    if other_cluster_samples:
        # 计算测试样本到所有其他簇样本的DTW距离
        cluster_distances = []
        test_sample_np_flat = test_sample.flatten().numpy()

        for idx, sample in other_cluster_samples:
            sample_np = sample.flatten().numpy()
            # 使用DTW距离
            dist, _ = fastdtw(test_sample_np_flat, sample_np, dist=lambda u, v: abs(u - v))
            cluster_distances.append((idx, dist))

        # 按DTW距离从大到小排序，选择最远的两个
        cluster_distances.sort(key=lambda x: x[1], reverse=True)
        # 只取前2个最远的样本
        farthest_samples = [idx for idx, _ in cluster_distances[:2]]
        dissimilar_samples = [(idx, labels[idx].item()) for idx in farthest_samples]

    return dissimilar_samples


def find_similar_and_dissimilar_samples(test_sample_id, data_name="computer"):
    """
    主函数：找到与测试样本最相似的4个样本和最不相似的2个样本

    参数:
        test_sample_id: 测试样本在测试集中的索引
        data_name: 数据集名称

    返回:
        result: 包含相似样本和不相似样本的字典
            - similar_samples: 4个相似样本的(id, label)列表
            - dissimilar_samples: 2个不相似样本的(id, label)列表
            - test_sample_info: 测试样本的(id, label)信息
    """
    # 1. 加载数据
    test_data, test_labels = load_test(data_name)
    train_data, train_labels = load_train(data_name)

    # 2. 检查test_sample_id是否有效
    if test_sample_id >= len(test_data):
        raise ValueError(f"test_sample_id {test_sample_id} 超出测试集范围 (0-{len(test_data) - 1})")

    # 3. 获取测试样本
    test_sample = test_data[test_sample_id]
    test_label = test_labels[test_sample_id].item()

    print(f"测试样本信息:")
    print(f"  - ID: {test_sample_id}")
    print(f"  - Label: {test_label}")
    print(f"  - 测试集大小: {len(test_data)}")
    print(f"  - 训练集大小: {len(train_data)}")

    # 4. 使用DTW找到4个最相似的样本（从训练集中找）
    print("\n寻找4个最相似的样本...")
    similar_samples = find_similar_samples_with_dtw(
        test_sample, train_data, train_labels, k=4
    )

    print("找到的4个最相似样本:")
    for i, (sample_id, sample_label) in enumerate(similar_samples):
        # 计算DTW距离用于展示
        dist, _ = fastdtw(
            test_sample.flatten().numpy(),
            train_data[sample_id].flatten().numpy(),
            dist=lambda u, v: abs(u - v)
        )
        print(f"  样本 {i + 1}: ID={sample_id}, Label={sample_label}, DTW距离={dist:.4f}")

    # 5. 通过聚类找到2个最不相似的样本（从训练集中找）
    print("\n寻找2个最不相似的样本...")
    dissimilar_samples = find_dissimilar_samples_with_clustering(
        test_sample, train_data, train_labels, n_clusters=2
    )

    print("找到的2个最不相似样本:")
    for i, (sample_id, sample_label) in enumerate(dissimilar_samples):
        # 计算DTW距离用于展示
        dist, _ = fastdtw(
            test_sample.flatten().numpy(),
            train_data[sample_id].flatten().numpy(),
            dist=lambda u, v: abs(u - v)
        )
        print(f"  样本 {i + 1}: ID={sample_id}, Label={sample_label}, DTW距离={dist:.4f}")

    # 6. 统计结果
    print("\n=== 结果统计 ===")

    # 相似样本的标签统计
    similar_labels = [label for _, label in similar_samples]
    similar_label_counts = Counter(similar_labels)
    print(f"相似样本的标签分布: {dict(similar_label_counts)}")

    # 不相似样本的标签统计
    dissimilar_labels = [label for _, label in dissimilar_samples]
    dissimilar_label_counts = Counter(dissimilar_labels)
    print(f"不相似样本的标签分布: {dict(dissimilar_label_counts)}")

    # 7. 返回结果
    result = {
        'test_sample_info': {
            'id': test_sample_id,
            'label': test_label
        },
        'similar_samples': similar_samples,  # [(id, label), ...]
        'dissimilar_samples': dissimilar_samples,  # [(id, label), ...]
        'dataset_info': {
            'name': data_name,
            'test_set_size': len(test_data),
            'train_set_size': len(train_data)
        }
    }

    return result


# 使用示例
if __name__ == "__main__":
    # 示例1: 查找测试样本0的相似和不相似样本
    print("=" * 60)
    print("示例1: 查找测试样本0的相似和不相似样本")
    print("=" * 60)

    try:
        result1 = find_similar_and_dissimilar_samples(
            test_sample_id=0,
            data_name="computer"
        )

        # 打印简洁结果
        print("\n最终结果:")
        print(f"测试样本: ID={result1['test_sample_info']['id']}, Label={result1['test_sample_info']['label']}")
        print(f"4个相似样本: {result1['similar_samples']}")
        print(f"2个不相似样本: {result1['dissimilar_samples']}")

    except Exception as e:
        print(f"处理时出错: {e}")

    # 示例2: 查找测试样本5的相似和不相似样本
    print("\n" + "=" * 60)
    print("示例2: 查找测试样本5的相似和不相似样本")
    print("=" * 60)

    try:
        result2 = find_similar_and_dissimilar_samples(
            test_sample_id=5,
            data_name="computer"
        )

        # 打印简洁结果
        print("\n最终结果:")
        print(f"测试样本: ID={result2['test_sample_info']['id']}, Label={result2['test_sample_info']['label']}")
        print(f"4个相似样本: {result2['similar_samples']}")
        print(f"2个不相似样本: {result2['dissimilar_samples']}")

    except Exception as e:
        print(f"处理时出错: {e}")