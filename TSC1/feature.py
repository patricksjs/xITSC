import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
from typing import List, Tuple

# 导入您提供的模块
from data.data_loader import load_data_for_sample, load_all_train_data


def extract_features_from_model(model, data_tensor, device):
    """从模型中提取表征特征"""
    model.eval()
    with torch.no_grad():
        # 调整数据格式以适应Transformer模型
        # 假设输入形状需要是 [batch_size, timesteps, features]
        if len(data_tensor.shape) == 2:
            data_tensor = data_tensor.unsqueeze(-1)  # 添加特征维度

        data_tensor = data_tensor.to(device)
        _, features = model(data_tensor)
        return features.cpu().numpy()


def find_top_k_similar_samples(test_sample_id: int, k: int = 5,
                               dataset_name: str = "computer",
                               model_path: str = None, model_args=None) -> Tuple[List[int], List[float]]:
    """
    找到与测试样本最相似的前K个训练样本

    参数:
    - test_sample_id: 测试样本的ID（在测试集中的索引）
    - k: 返回的最相似样本数量
    - dataset_name: 数据集名称
    - model_path: 模型文件路径
    - model_args: 模型参数

    返回:
    - top_k_indices: 最相似的K个训练样本ID列表
    - top_k_similarities: 对应的相似度分数列表
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载测试样本（使用原始数据，不进行归一化）
    print(f"加载测试样本 {test_sample_id}...")
    test_data, test_label = load_data_for_sample(test_sample_id, dataset_name)

    if test_data is None:
        raise ValueError(f"无法加载测试样本 {test_sample_id}")

    # 直接将原始数据转换为tensor
    test_tensor = torch.tensor(test_data, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

    # 2. 加载所有训练数据（使用原始数据，不进行归一化）
    print("加载训练数据...")
    train_data, train_labels, train_indices = load_all_train_data(dataset_name)

    if train_data is None:
        raise ValueError("无法加载训练数据")

    # 直接将原始数据转换为tensor
    train_tensor = torch.tensor(train_data, dtype=torch.float32).unsqueeze(-1)

    # 3. 加载模型（如果提供了模型路径）
    model = None
    if model_path and model_args:
        print("加载Transformer模型...")
        from models.feat import TransformerModel
        model = TransformerModel(model_args, num_classes=model_args.num_classes).to(device)

        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        except:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
    else:
        print("警告：未提供模型路径，将使用原始数据作为特征")

    # 4. 提取特征
    print("提取特征...")

    # 提取测试样本特征
    if model:
        test_features = extract_features_from_model(model, test_tensor, device)
    else:
        # 如果没有模型，使用原始数据作为特征
        test_features = test_data.reshape(1, -1)

    # 提取训练样本特征
    if model:
        # 批量提取训练样本特征以提高效率
        batch_size = 64
        train_features_list = []

        for i in range(0, len(train_tensor), batch_size):
            batch = train_tensor[i:i + batch_size]
            batch_features = extract_features_from_model(model, batch, device)
            train_features_list.append(batch_features)

        train_features = np.vstack(train_features_list)
    else:
        # 如果没有模型，使用原始数据作为特征
        train_features = train_data

    # 5. 计算相似度
    print("计算相似度...")

    # 使用余弦相似度
    similarities = cosine_similarity(test_features, train_features)[0]

    # 6. 找到最相似的前K个样本
    top_k_indices_np = np.argsort(similarities)[-k:][::-1]
    top_k_similarities = similarities[top_k_indices_np].tolist()

    # 修正：将numpy索引转换为Python整数索引，然后获取对应的训练样本ID
    top_k_train_indices = []
    for idx in top_k_indices_np:
        # train_indices是Python列表，使用整数索引访问
        top_k_train_indices.append(train_indices[int(idx)])

    # 7. 打印结果
    print(f"\n测试样本 {test_sample_id} (标签: {test_label}) 的最相似前{k}个样本:")
    print("-" * 60)
    for i, (idx, sim) in enumerate(zip(top_k_train_indices, top_k_similarities)):
        # 获取训练样本的标签
        train_idx_in_train_set = train_indices.index(idx)  # 找到在train_data中的位置
        train_label = train_labels[train_idx_in_train_set]
        print(f"{i + 1}. 训练样本ID: {idx} (标签: {train_label}), 相似度: {sim:.4f}")

    # 8. 返回结果
    return top_k_train_indices, top_k_similarities


def save_features_cache(dataset_name: str, model_path: str, model_args,
                        cache_dir: str = "feature_cache") -> str:
    """
    预计算并保存所有训练样本的特征，加速后续查询
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建缓存目录
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{dataset_name}_features.pkl")

    # 检查是否已有缓存
    if os.path.exists(cache_file):
        print(f"找到缓存文件: {cache_file}")
        return cache_file

    # 1. 加载训练数据
    print("加载训练数据...")
    train_data, train_labels, train_indices = load_all_train_data(dataset_name)

    # 直接使用原始数据，不进行归一化
    train_tensor = torch.tensor(train_data, dtype=torch.float32).unsqueeze(-1)

    # 2. 加载模型
    print("加载模型...")
    from models.feat import TransformerModel
    model = TransformerModel(model_args, num_classes=model_args.num_classes).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # 3. 批量提取特征
    print("提取特征...")
    batch_size = 64
    train_features_list = []

    for i in range(0, len(train_tensor), batch_size):
        batch = train_tensor[i:i + batch_size]
        batch_features = extract_features_from_model(model, batch, device)
        train_features_list.append(batch_features)

    train_features = np.vstack(train_features_list)

    # 4. 保存到缓存
    print(f"保存特征到缓存: {cache_file}")
    cache_data = {
        'features': train_features,
        'labels': train_labels,
        'indices': train_indices,  # 这是Python列表
        'dataset_name': dataset_name
    }

    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)

    return cache_file


def find_top_k_with_cache(test_sample_id: int, k: int = 5, dataset_name: str = "computer",
                          cache_file: str = None, model_path: str = None,
                          model_args=None) -> Tuple[List[int], List[float]]:
    """
    使用缓存的特征文件快速查找相似样本
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载测试样本（使用原始数据）
    test_data, test_label = load_data_for_sample(test_sample_id, dataset_name)
    test_tensor = torch.tensor(test_data, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

    # 2. 加载缓存的特征
    if cache_file and os.path.exists(cache_file):
        print(f"从缓存加载特征: {cache_file}")
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)

        train_features = cache_data['features']
        train_labels = cache_data['labels']
        train_indices = cache_data['indices']  # Python列表
    else:
        # 如果没有缓存，重新计算
        cache_file = save_features_cache(dataset_name, model_path, model_args)
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)

        train_features = cache_data['features']
        train_labels = cache_data['labels']
        train_indices = cache_data['indices']

    # 3. 提取测试样本特征
    if model_path and model_args:
        from models.feat import TransformerModel
        model = TransformerModel(model_args, num_classes=model_args.num_classes).to(device)

        checkpoint = torch.load(model_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        model.eval()

        test_features = extract_features_from_model(model, test_tensor, device)
    else:
        # 如果没有模型，使用原始数据作为特征
        test_features = test_data.reshape(1, -1)

    # 4. 计算相似度
    similarities = cosine_similarity(test_features, train_features)[0]

    # 5. 找到最相似的前K个样本
    top_k_indices_np = np.argsort(similarities)[-k:][::-1]
    top_k_similarities = similarities[top_k_indices_np].tolist()

    # 修正：获取对应的训练样本ID
    top_k_train_indices = []
    for idx in top_k_indices_np:
        top_k_train_indices.append(train_indices[int(idx)])

    # 6. 打印结果
    print(f"\n测试样本 {test_sample_id} (标签: {test_label}) 的最相似前{k}个样本:")
    print("-" * 60)
    for i, (idx, sim) in enumerate(zip(top_k_train_indices, top_k_similarities)):
        # 获取训练样本的标签
        train_idx_in_train_set = train_indices.index(idx)
        train_label = train_labels[train_idx_in_train_set]
        print(f"{i + 1}. 训练样本ID: {idx} (标签: {train_label}), 相似度: {sim:.4f}")

    return top_k_train_indices, top_k_similarities


# 简化版本的主函数
def main_simple(test_id: int, k: int = 5, dataset_name: str = "computer"):
    """
    简化的主函数，直接调用相似样本查找
    """

    # 定义模型参数（需要与训练时一致）
    class ModelArgs:
        def __init__(self):
            self.timesteps = 720
            self.num_layers = 2
            self.d_model = 64
            self.nhead = 8
            self.dim_feedforward = 256
            self.dropout = 0.2
            self.num_classes = 2  # computer数据集有2个类别

    model_args = ModelArgs()

    # 模型路径
    model_path = r'C:\Users\34517\Desktop\zuhui\xITSC\classification_models\computer\transformer\transformer.pt'

    try:
        # 使用缓存查找（更快）
        cache_file = f"feature_cache/{dataset_name}_features.pkl"
        if os.path.exists(cache_file):
            print("使用缓存查找...")
            top_k_indices, top_k_similarities = find_top_k_with_cache(
                test_id, k, dataset_name, cache_file, model_path, model_args
            )
        else:
            print("无缓存，实时计算...")
            top_k_indices, top_k_similarities = find_top_k_similar_samples(
                test_id, k, dataset_name, model_path, model_args
            )


        return top_k_indices, top_k_similarities

    except Exception as e:
        print(f"查找相似样本时出错: {e}")
        return [], []


# 使用示例
if __name__ == "__main__":
    # 示例1：查找测试样本ID=0的最相似的5个训练样本
    test_id = 200
    k = 5
    dataset_name = "computer"

    print(f"查找测试样本 {test_id} 的最相似 {k} 个训练样本...")
    top_k_indices, top_k_similarities = main_simple(test_id, k, dataset_name)

    if top_k_indices:
        print(f"\n找到的最相似样本ID: {top_k_indices}")
        print(f"相似度分数: {top_k_similarities}")