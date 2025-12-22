import torch
from torch.utils.data import Dataset
import pandas as pd
from scipy.fft import fft, ifft
from scipy.signal import stft, istft
import random
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import shap
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans
from fastdtw import fastdtw
import warnings

from models.models import TransformerModel

"""
增加保存热力图数据
"""
# 设置随机种子
random.seed(42)
torch.set_num_threads(32)
torch.manual_seed(911)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore", category=UserWarning)


def load_all(data_name):
    print(f"Dataset is {data_name}")
    # 这里保留您原来的数据加载代码
    # ... (您原来的数据加载代码)

    if data_name == "computer":
        file_path = r"C:\Users\34517\Desktop\zuhui\论文\Computers\Computers_TRAIN.txt"
        file_path2 = r"C:\Users\34517\Desktop\zuhui\论文\Computers\Computers_TEST.txt"
        df = pd.read_csv(file_path, header=None, delim_whitespace=True)
        df2 = pd.read_csv(file_path2, header=None, delim_whitespace=True)

        # Separate labels and data
        labels = df.iloc[:, 0].values
        data = df.iloc[:, 1:].values
        labels2 = df2.iloc[:, 0].values
        data2 = df2.iloc[:, 1:].values
        # Convert to PyTorch tensors
        labels_tensor = torch.tensor(labels, dtype=torch.float32) - 1
        data_tensor = torch.tensor(data, dtype=torch.float32)
        labels_tensor2 = torch.tensor(labels2, dtype=torch.float32) - 1
        data_tensor2 = torch.tensor(data2, dtype=torch.float32)

        data = torch.cat([data_tensor, data_tensor2], dim=0)
        labels = torch.cat((labels_tensor, labels_tensor2), dim=0)
    elif data_name == "EPG":
        # Read the file into a DataFrame using space as the separator
        file_path = r"C:\Users\34517\Desktop\zuhui\论文\EPG\InsectEPGSmallTrain_TRAIN.txt"
        file_path2 =  r"C:\Users\34517\Desktop\zuhui\论文\EPG\InsectEPGSmallTrain_TEST.txt"
        # 读取数据（修正分隔符警告）
        df = pd.read_csv(file_path, header=None, sep=r'\s+')  # 替换delim_whitespace，添加r避免转义警告
        df2 = pd.read_csv(file_path2, header=None, sep=r'\s+')

        # 分离标签和数据，并将标签从[1,2,3]转为[0,1,2]
        labels = df.iloc[:, 0].values - 1  # 关键：标签减1
        data = df.iloc[:, 1:].values
        labels2 = df2.iloc[:, 0].values - 1  # 关键：测试集标签同样减1
        data2 = df2.iloc[:, 1:].values

        # 验证修正后的标签范围
        print("修正后标签最小值：", labels.min())  # 应输出0.0
        print("修正后标签最大值：", labels.max())  # 应输出2.0
        print("修正后所有标签值：", np.unique(labels))  # 应输出[0. 1. 2.]

        # 转换为张量（注意标签需为整数类型，而非float）
        labels_tensor = torch.tensor(labels, dtype=torch.long)  # 交叉熵要求标签为long类型
        data_tensor = torch.tensor(data, dtype=torch.float32)
        labels_tensor2 = torch.tensor(labels2, dtype=torch.long)
        data_tensor2 = torch.tensor(data2, dtype=torch.float32)

        data = torch.cat([data_tensor, data_tensor2], dim=0)
        labels = torch.cat((labels_tensor, labels_tensor2), dim=0)
    elif data_name == "cincecgtorso":
        file_path = r"C:\Users\34517\Desktop\zuhui\论文\CinCECGTorso\CinCECGTorso_TRAIN.txt"

        file_path2 = r"C:\Users\34517\Desktop\zuhui\论文\CinCECGTorso\CinCECGTorso_TEST.txt"
        df2 = pd.read_csv(file_path2, header=None, delim_whitespace=True)
        df = pd.read_csv(file_path, header=None, delim_whitespace=True)
        labels = df.iloc[:, 0].values
        data = df.iloc[:, 1:].values
        labels2 = df2.iloc[:, 0].values
        data2 = df2.iloc[:, 1:].values
        # Convert to PyTorch tensors

        labels_tensor = torch.tensor(labels, dtype=torch.float32) - 1
        data_tensor = torch.tensor(data, dtype=torch.float32)
        labels_tensor2 = torch.tensor(labels2, dtype=torch.float32) - 1
        data_tensor2 = torch.tensor(data2, dtype=torch.float32)

        data = torch.cat([data_tensor, data_tensor2], dim=0)
        labels = torch.cat((labels_tensor, labels_tensor2), dim=0)

        print("labels min:", labels.min().item())
        print("labels max:", labels.max().item())
        print("unique labels:", torch.unique(labels).tolist())
    elif data_name == "ECG200":
        file_path = r"C:\Users\34517\Desktop\zuhui\论文\ECG200\ECG200_TRAIN.txt"

        file_path2 = r"C:\Users\34517\Desktop\zuhui\论文\ECG200\ECG200_TEST.txt"
        df2 = pd.read_csv(file_path2, header=None, delim_whitespace=True)
        df = pd.read_csv(file_path, header=None, delim_whitespace=True)

        labels = df.iloc[:, 0].values
        data = df.iloc[:, 1:].values
        labels2 = df2.iloc[:, 0].values
        data2 = df2.iloc[:, 1:].values

        # ========== 核心修改：标签映射 ==========
        # 方法1：数学运算（推荐，更高效）
        # 原标签：-1 → (-1 + 1)/2 = 0；1 → (1 + 1)/2 = 1
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        labels_tensor = (labels_tensor + 1) / 2  # 把-1→0，1→1

        labels_tensor2 = torch.tensor(labels2, dtype=torch.float32)
        labels_tensor2 = (labels_tensor2 + 1) / 2  # 同理处理测试集标签
        # =======================================

        # 方法2：条件替换（更直观，适合复杂映射）
        # labels_tensor = torch.where(labels_tensor == -1, torch.tensor(0.0), torch.tensor(1.0))
        # labels_tensor2 = torch.where(labels_tensor2 == -1, torch.tensor(0.0), torch.tensor(1.0))

        data_tensor = torch.tensor(data, dtype=torch.float32)
        data_tensor2 = torch.tensor(data2, dtype=torch.float32)

        data = torch.cat([data_tensor, data_tensor2], dim=0)
        labels = torch.cat((labels_tensor, labels_tensor2), dim=0)

        print("labels min:", labels.min().item())  # 输出 0.0
        print("labels max:", labels.max().item())  # 输出 1.0
        print("unique labels:", torch.unique(labels).tolist())  # 输出 [0.0, 1.0]

    return data, labels


def load_train(data_name):
    print(f"Dataset is {data_name}")
    # 这里保留您原来的数据加载代码
    # ... (您原来的数据加载代码)

    if data_name == "computer":
        file_path2 = r"C:\Users\34517\Desktop\zuhui\论文\Computers\Computers_TRAIN.txt"
        df2 = pd.read_csv(file_path2, header=None, sep='\s+')

        labels2 = df2.iloc[:, 0].values
        data2 = df2.iloc[:, 1:].values

        labels = torch.tensor(labels2 - 1, dtype=torch.long)
        data = torch.tensor(data2, dtype=torch.float32)

        print(f"computer数据集标签类型：{labels.dtype}")
        print(f"computer数据集标签范围：{labels.min().item()} ~ {labels.max().item()}")
        # 打印前两个样本的数值范围
        if len(data) >= 2:
            print("\n前两个样本的数值范围：")
            for i in range(2):
                sample_min = data[i].min().item()
                sample_max = data[i].max().item()
                print(f"样本{i + 1}：{sample_min:.6f} ~ {sample_max:.6f}")
        elif len(data) == 1:
            print("\n只有1个样本，其数值范围：")
            sample_min = data[0].min().item()
            sample_max = data[0].max().item()
            print(f"样本1：{sample_min:.6f} ~ {sample_max:.6f}")
        else:
            print("\n没有数据样本")
    elif data_name == "cincecgtorso":
        file_path2 = r"C:\Users\34517\Desktop\zuhui\论文\CinCECGTorso\CinCECGTorso_TRAIN.txt"
        df2 = pd.read_csv(file_path2, header=None, delim_whitespace=True)

        labels2 = df2.iloc[:, 0].values
        data2 = df2.iloc[:, 1:].values
        # Convert to PyTorch tensors

        labels = torch.tensor(labels2 - 1, dtype=torch.long)
        data = torch.tensor(data2, dtype=torch.float32)

        print("labels min:", labels.min().item())
        print("labels max:", labels.max().item())
        print("unique labels:", torch.unique(labels).tolist())
    # 其他数据集的加载代码...

    return data, labels


def load_test(data_name):
    print(f"Dataset is {data_name}")
    # 这里保留您原来的数据加载代码
    # ... (您原来的数据加载代码)

    if data_name == "computer":
        file_path2 = r"C:\Users\34517\Desktop\zuhui\论文\Computers\Computers_TEST.txt"
        df2 = pd.read_csv(file_path2, header=None, sep='\s+')

        labels2 = df2.iloc[:, 0].values
        data2 = df2.iloc[:, 1:].values

        labels = torch.tensor(labels2 - 1, dtype=torch.long)
        data = torch.tensor(data2, dtype=torch.float32)

        print(f"computer数据集标签类型：{labels.dtype}")
        print(f"computer数据集标签范围：{labels.min().item()} ~ {labels.max().item()}")
        # 打印前两个样本的数值范围
        if len(data) >= 2:
            print("\n前两个样本的数值范围：")
            for i in range(2):
                sample_min = data[i].min().item()
                sample_max = data[i].max().item()
                print(f"样本{i + 1}：{sample_min:.6f} ~ {sample_max:.6f}")
        elif len(data) == 1:
            print("\n只有1个样本，其数值范围：")
            sample_min = data[0].min().item()
            sample_max = data[0].max().item()
            print(f"样本1：{sample_min:.6f} ~ {sample_max:.6f}")
        else:
            print("\n没有数据样本")
    elif data_name == "cincecgtorso":
        file_path2 = r"C:\Users\34517\Desktop\zuhui\论文\CinCECGTorso\CinCECGTorso_TEST.txt"
        df2 = pd.read_csv(file_path2, header=None, delim_whitespace=True)

        labels2 = df2.iloc[:, 0].values
        data2 = df2.iloc[:, 1:].values
        # Convert to PyTorch tensors

        labels = torch.tensor(labels2 - 1, dtype=torch.long)
        data = torch.tensor(data2, dtype=torch.float32)

        print("labels min:", labels.min().item())
        print("labels max:", labels.max().item())
        print("unique labels:", torch.unique(labels).tolist())
    # 其他数据集的加载代码...

    return data, labels


def normalize_data(data):
    """样本内单独归一化：对每个样本单独进行均值方差归一化"""
    if isinstance(data, torch.Tensor):
        data_np = data.numpy()
    else:
        data_np = data

    normalized_data_np = np.zeros_like(data_np)
    sample_means = []
    sample_stds = []

    for i in range(len(data_np)):
        sample = data_np[i]
        sample_mean = np.mean(sample)
        sample_std = np.std(sample)

        # 避免除零错误
        if sample_std == 0:
            normalized_sample = np.zeros_like(sample)
        else:
            normalized_sample = sample / sample_std

        normalized_data_np[i] = normalized_sample
        sample_means.append(sample_mean)
        sample_stds.append(sample_std)

    print(f"样本内归一化完成，共处理 {len(data_np)} 个样本")
    print(f"单个样本归一化后范围示例: [{np.min(normalized_data_np[0]):.4f}, {np.max(normalized_data_np[0]):.4f}]")

    return (torch.tensor(normalized_data_np, dtype=torch.float32),
            torch.tensor(sample_means, dtype=torch.float32),
            torch.tensor(sample_stds, dtype=torch.float32))


def plot_time_series(data, labels, sample_idx, output_path, dataset_name, data_type='test'):
    """绘制时间序列图 - y轴自适应"""
    plt.figure(figsize=(12, 6))
    sample = data[sample_idx].numpy()
    time_steps = np.arange(len(sample))

    n_ticks = 12
    tick_indices = np.linspace(0, len(sample) - 1, n_ticks, dtype=int)
    tick_labels = [f"{i}" for i in tick_indices]

    plt.plot(time_steps, sample, linewidth=2)

    # 添加y=0横线（如果数据跨越0）
    sample_min, sample_max = np.min(sample), np.max(sample)
    if sample_min <= 0 <= sample_max:
        plt.axhline(y=0, color='black', linewidth=3, linestyle='-', alpha=0.8)

    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Normalized Value', fontsize=12)

    # 根据数据类型设置标题和保存路径
    if data_type == 'train':
        plt.title(f'Plot - {dataset_name}, Sample {sample_idx}, Label: {labels[sample_idx].item()}',
                  fontsize=14)
        label_dir = f"{output_path}/train/{dataset_name}_{labels[sample_idx].item()}"
        save_dir = label_dir
    else:  # test
        plt.title(f'Plot - {dataset_name}, Sample {sample_idx}',
                  fontsize=14)
        label_dir = f"{output_path}/test/{dataset_name}_{labels[sample_idx].item()}"
        save_dir = label_dir

    plt.grid(True, alpha=0.3)
    plt.xticks(tick_indices, tick_labels)

    # 自适应y轴范围，基于当前样本的数据范围
    margin = (sample_max - sample_min) * 0.1  # 添加10%的边距
    plt.ylim(sample_min - margin, sample_max + margin)

    plt.tight_layout()

    # 创建目录并保存图片
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/plot_sample{sample_idx}.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_time_frequency(data, sample_idx, labels, output_path, dataset_name, nperseg=64, freq_ratio=0.66,
                        data_type='test'):
    """绘制时频图（使用归一化后的数据，只显示前N比例的低频分量，横坐标与时序图一致）"""
    plt.figure(figsize=(12, 6))
    sample = data[sample_idx].numpy()
    total_timesteps = len(sample)  # 获取原始时序的总时间步（与plot一致）

    fs = 1.0
    noverlap = nperseg // 2
    f, t, Zxx = stft(
        sample,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        window='hann',
        boundary='zeros',
        scaling='spectrum'
    )
    spec_magnitude = np.abs(Zxx)

    num_freq_total = len(f)
    num_freq_keep = int(num_freq_total * freq_ratio)
    if num_freq_keep < 1:
        num_freq_keep = 1
    spec_magnitude = spec_magnitude[:num_freq_keep, :]
    f = f[:num_freq_keep]

    n_ticks = 12
    tick_indices = np.linspace(0, total_timesteps - 1, n_ticks, dtype=int)
    tick_positions = tick_indices / total_timesteps * t[-1]
    tick_labels = [f"{idx}" for idx in tick_indices]

    plt.pcolormesh(t, f, spec_magnitude, shading='auto', cmap='viridis')
    plt.colorbar(label='Signal Magnitude')
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)

    # 根据数据类型设置标题和保存路径
    if data_type == 'train':
        plt.title(f'STFT - {dataset_name}, Sample {sample_idx}, Label: {labels[sample_idx].item()}',
                  fontsize=14)
        label_dir = f"{output_path}/train/{dataset_name}_{labels[sample_idx].item()}"
        save_dir = label_dir
    else:  # test
        plt.title(f'STFT - {dataset_name}, Sample {sample_idx}',
                  fontsize=14)
        label_dir = f"{output_path}/test/{dataset_name}_{labels[sample_idx].item()}"
        save_dir = label_dir

    plt.xticks(tick_positions, tick_labels)
    plt.tight_layout()

    # 创建目录并保存图片
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/STFT_sample{sample_idx}.png", dpi=300, bbox_inches='tight')
    plt.close()

    return f, t, Zxx[:num_freq_keep, :]


def load_transformer_model(args):
    """加载训练好的Transformer模型"""
    print(f"\n加载Transformer模型: {args.model_path}")
    model = TransformerModel(args=args, num_classes=args.num_classes).to(args.device)

    try:
        checkpoint = torch.load(args.model_path, map_location=args.device, weights_only=True)
    except:
        checkpoint = torch.load(args.model_path, map_location=args.device, weights_only=False)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print("模型加载完成！")
    return model


class STFTSHAPExplainer:
    def __init__(self, args, model, data, labels):
        self.args = args
        self.model = model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = data  # 原始数据
        self.labels = labels
        self.background_stfts = []  # 保存完整的STFT结果
        self.fs = args.fs
        self.nperseg = args.nperseg
        self.noverlap = args.noverlap
        print(f"设备: {self.device} | STFT参数: fs={self.fs}, nperseg={self.nperseg}, noverlap={self.noverlap}")

    def _get_stft(self, ts):
        """计算单个时域信号的STFT（返回完整的复数STFT）"""
        if ts.ndim > 1:
            ts = ts.mean(axis=1)  # 多变量→单通道
        f, t, Zxx = stft(
            ts,
            fs=self.fs,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            boundary='zeros'
        )
        return f, t, Zxx  # 返回完整的复数STFT

    def _model_predict(self, spec_flat_batch):
        """KernelExplainer专用预测函数：输入展平的复数时频图→直接ISTFT→时域→模型预测概率"""
        batch_size = spec_flat_batch.shape[0]
        probs_batch = []

        # 获取2D时频图的形状
        num_freq = self.args.nperseg // 2 + 1
        num_time = spec_flat_batch.shape[1] // num_freq

        for i in range(batch_size):
            # 1. 展平的复数时频图 → 恢复为2D复数时频图
            spec_flat = spec_flat_batch[i]
            Zxx = spec_flat.reshape((num_freq, num_time)).astype(np.complex128)

            # 2. 直接ISTFT转回时域信号
            _, xrec = istft(
                Zxx,
                fs=self.fs,
                nperseg=self.nperseg,
                noverlap=self.noverlap,
                boundary='zeros'
            )

            # 3. 调整时域长度（匹配Transformer输入的timesteps）
            if xrec.shape[0] > self.args.timesteps:
                xrec = xrec[:self.args.timesteps]
            elif xrec.shape[0] < self.args.timesteps:
                xrec = np.pad(xrec, (0, self.args.timesteps - xrec.shape[0]), mode='constant')

            # 4. 适配Transformer输入格式
            x_tensor = torch.tensor(xrec, dtype=torch.float32).unsqueeze(0).unsqueeze(1)
            x_tensor = x_tensor.to(self.device)

            # 5. 模型预测（输出概率）
            with torch.no_grad():
                logits = self.model(x_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            probs_batch.append(probs)

        return np.array(probs_batch)

    def predict_label(self, sample_data):
        """
        预测单个样本的标签（置信度最高的类别）
        输入: 原始时域数据 [timesteps] 或 [1, timesteps]
        输出: 预测的类别标签 (int)
        """
        # 确保输入格式正确
        if isinstance(sample_data, torch.Tensor):
            sample_data = sample_data.numpy()

        # 确保是2D数组 [batch_size, timesteps]
        if sample_data.ndim == 1:
            sample_data = sample_data.reshape(1, -1)

        # 转换为PyTorch张量并调整形状
        sample_tensor = torch.from_numpy(sample_data).float().to(self.device)
        if len(sample_tensor.shape) == 2:
            sample_tensor = sample_tensor.unsqueeze(1)  # [batch_size, 1, timesteps]

        # 模型预测
        self.model.eval()
        with torch.no_grad():
            output = self.model(sample_tensor)
            probabilities = torch.softmax(output, dim=1)

            # 获取置信度最高的类别
            confidence, predicted = torch.max(probabilities, 1)

            print(f"预测标签: {predicted.item()}, 置信度: {confidence.item():.4f}")
            print(f"所有类别概率: {probabilities.cpu().numpy().flatten()}")

        return predicted.item()

    def select_representative_samples_by_clustering(self, num_clusters_per_class=7, num_samples_per_class=25):
        """
        使用KMeans聚类选择每个类别最具代表性的样本
        Args:
            num_clusters_per_class: 每个类别聚类的簇数（默认7）
            num_samples_per_class: 每个类别最终选择的样本总数（默认25）
        """
        print(f"使用KMeans聚类选择每个类别最具代表性的样本：")
        print(f"  - 每个类别聚类数：{num_clusters_per_class}")
        print(f"  - 每个类别选择样本数：{num_samples_per_class}")

        unique_labels = torch.unique(self.labels).tolist()
        print(f"数据集中包含的类别: {unique_labels}")

        representative_indices = []

        for label in unique_labels:
            # 获取当前类别的所有样本索引
            class_indices = [i for i in range(len(self.labels)) if int(self.labels[i].item()) == label]
            class_data = self.data[class_indices].numpy()
            num_class_samples = len(class_data)

            print(f"\n类别 {label} 有 {num_class_samples} 个样本")

            if num_class_samples <= num_samples_per_class:
                # 如果样本数不足，使用所有样本
                representative_indices.extend(class_indices)
                print(f"类别 {label}: 样本数不足({num_class_samples}), 使用所有样本")
                continue

            # 计算每个簇需要选择的样本数
            samples_per_cluster = num_samples_per_class // num_clusters_per_class
            remaining_samples = num_samples_per_class % num_clusters_per_class  # 余数（分配给前N个簇）

            # 对当前类别的样本进行KMeans聚类（固定7个簇）
            n_clusters = min(num_clusters_per_class, num_class_samples)  # 防止簇数超过样本数
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(class_data)

            # 选择每个聚类中距离聚类中心最近的样本
            cluster_centers = kmeans.cluster_centers_
            selected_indices_in_class = []

            for cluster_id in range(n_clusters):
                # 找到属于当前聚类的样本索引
                cluster_mask = (cluster_labels == cluster_id)
                cluster_samples = class_data[cluster_mask]
                cluster_original_indices = [class_indices[i] for i in range(len(class_indices)) if
                                            cluster_labels[i] == cluster_id]

                if len(cluster_samples) == 0:
                    continue

                # 计算每个样本到聚类中心的距离
                distances = np.linalg.norm(cluster_samples - cluster_centers[cluster_id], axis=1)

                # 确定当前簇要选的样本数（前remaining_samples个簇多选1个）
                current_cluster_samples = samples_per_cluster
                if cluster_id < remaining_samples:
                    current_cluster_samples += 1

                # 选择距离最近的N个样本
                current_cluster_samples = min(current_cluster_samples, len(cluster_samples))  # 防止超过簇内样本数
                closest_indices = np.argsort(distances)[:current_cluster_samples]

                # 添加原始索引
                selected_cluster_indices = [cluster_original_indices[idx] for idx in closest_indices]
                selected_indices_in_class.extend(selected_cluster_indices)

                print(f"  类别 {label} - 簇 {cluster_id}: 选择 {len(selected_cluster_indices)} 个样本")

            # 确保最终选择的样本数不超过设定值
            selected_indices_in_class = selected_indices_in_class[:num_samples_per_class]
            representative_indices.extend(selected_indices_in_class)
            print(f"类别 {label}: 最终选择了 {len(selected_indices_in_class)} 个代表性样本")

        print(f"\n总共选择了 {len(representative_indices)} 个代表性样本作为背景数据")
        return representative_indices

    def prepare_background_data(self, target_label=None, num_clusters_per_class=7, num_samples_per_class=25):
        """
        准备背景数据用于SHAP计算 - 使用聚类选择的代表性样本
        Args:
            num_clusters_per_class: 每个类别聚类数（默认7）
            num_samples_per_class: 每个类别选择的样本数（默认25）
        """
        bg_specs = []

        # 使用聚类算法选择代表性样本（传入簇数和每类样本数参数）
        representative_indices = self.select_representative_samples_by_clustering(
            num_clusters_per_class=num_clusters_per_class,
            num_samples_per_class=num_samples_per_class
        )

        # 计算代表性样本的STFT并展平
        for idx in representative_indices:
            data = self.data[idx].numpy()  # 使用原始数据
            ts = data
            if ts.ndim > 1:
                ts = ts.mean(axis=1)
            f, t, Zxx = stft(
                ts,
                fs=self.fs,
                nperseg=self.nperseg,
                noverlap=self.noverlap,
                boundary='zeros'
            )
            bg_specs.append(Zxx)

        self.background_stfts = bg_specs

        print(f"背景数据准备完成，共 {len(bg_specs)} 个样本")
        return np.array([Zxx.flatten() for Zxx in bg_specs]), f, t

    def compute_shap_for_sample(self, sample_idx, target_label):
        """计算单个样本的SHAP值"""
        # 准备测试样本（使用原始数据）
        sample_data = self.data[sample_idx]
        f, t, test_spec = self._get_stft(sample_data.numpy())
        test_spec_flat = test_spec.flatten()

        # 准备背景数据（传入命令行参数）
        background_data, _, _ = self.prepare_background_data(
            num_clusters_per_class=self.args.num_clusters_per_class,
            num_samples_per_class=self.args.num_samples_per_class
        )

        # 初始化KernelExplainer
        explainer = shap.KernelExplainer(
            model=self._model_predict,
            data=background_data,
            link="logit"
        )

        # 计算SHAP值
        shap_values = explainer.shap_values(
            X=test_spec_flat.reshape(1, -1),
            nsamples="auto",
            l1_reg="auto"
        )

        num_freq, num_time = test_spec.shape
        shap_2d = shap_values[0][:, target_label].reshape((num_freq, num_time))

        return f, t, test_spec, shap_2d

    def validate_stft_reconstruction(self, sample_idx=0):
        """验证STFT重构的准确性"""
        # 原始信号
        original_data = self.data[sample_idx].numpy()
        if original_data.ndim > 1:
            original_data = original_data.mean(axis=1)

        # STFT → ISTFT 重构
        f, t, Zxx = self._get_stft(original_data)
        _, reconstructed = istft(
            Zxx,
            fs=self.fs,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            boundary='zeros'
        )

        # 裁剪到相同长度
        min_len = min(len(original_data), len(reconstructed))
        original_data = original_data[:min_len]
        reconstructed = reconstructed[:min_len]

        # 计算重构误差
        mse = np.mean((original_data - reconstructed) ** 2)
        correlation = np.corrcoef(original_data, reconstructed)[0, 1]

        print(f"STFT重构验证 - 样本 {sample_idx}:")
        print(f"  MSE: {mse:.6f}")
        print(f"  相关系数: {correlation:.6f}")

        return mse < 0.01 and correlation > 0.95  # 返回是否通过验证




def plot_shap_heatmap(f, t, shap_2d, sample_idx, label, output_path, dataset_name, data_type='test', freq_ratio=0.66):
    """绘制SHAP热力图（只显示前N比例的低频分量，横坐标与时序图一致）"""
    plt.figure(figsize=(12, 6))
    # 获取原始时序的总时间步（需从数据中推导，与plot一致）
    nperseg = (len(f) - 1) * 2  # 从频率点数反推STFT窗口长度（f的长度= nperseg//2 +1）
    total_timesteps = len(t) * (nperseg - nperseg // 2) - (nperseg - nperseg // 2)  # 反推原始时序总时间步

    # 同步切片SHAP值
    num_freq_total = len(f)
    num_freq_keep = int(num_freq_total * freq_ratio)
    if num_freq_keep < 1:
        num_freq_keep = 1
    shap_2d_sliced = shap_2d[:num_freq_keep, :]
    f_sliced = f[:num_freq_keep]

    # 根据数据类型设置保存路径
    if data_type == 'train':
        save_dir = f"{output_path}/train/{dataset_name}_{label}"
    else:  # test
        save_dir = f"{output_path}/test/{dataset_name}_{label}"

    os.makedirs(save_dir, exist_ok=True)

    # 保存SHAP数值数据为npy文件
    shap_npy_path = f"{save_dir}/shap_matrix_sample{sample_idx}.npy"
    np.save(shap_npy_path, shap_2d_sliced)
    print(f"SHAP矩阵已保存: {shap_npy_path} (形状: {shap_2d_sliced.shape})")

    # 横坐标刻度与时序图（plot）完全一致
    n_ticks = 12
    tick_indices = np.linspace(0, total_timesteps - 1, n_ticks, dtype=int)
    tick_positions = tick_indices / total_timesteps * t[-1]
    tick_labels = [f"{idx}" for idx in tick_indices]

    vmax = np.max(np.abs(shap_2d_sliced))
    im = plt.pcolormesh(t, f_sliced, shap_2d_sliced, cmap='bwr', shading='auto',
                        vmin=-vmax, vmax=vmax)

    plt.colorbar(im, label='SHAP Value')
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Frequency [Hz]', fontsize=12)

    # 根据数据类型设置标题
    if data_type == 'train':
        plt.title(f'SHAP Heatmap - {dataset_name}, Sample {sample_idx}, Label: {label}',
                  fontsize=12)
    else:  # test
        plt.title(f'SHAP Heatmap - {dataset_name}, Sample {sample_idx}',
                  fontsize=12)

    plt.xticks(tick_positions, tick_labels)
    plt.tight_layout()

    plt.savefig(f"{save_dir}/shap_sample{sample_idx}.png",
                dpi=300, bbox_inches='tight')

    plt.close()

    print(f"SHAP热力图已保存: {save_dir}/shap_sample{sample_idx}.png")

    # 返回保存的文件路径
    return {
        'shap_npy_path': shap_npy_path
    }


def compute_shap_heatmap_new(model, data, labels, means, stds, sample_idx, output_path, dataset_name,
                             nperseg=64, fs=1.0, freq_ratio=0.66, num_samples_per_class=25,
                             num_clusters_per_class=7, data_type='test'):
    """使用新方法计算SHAP热力图"""
    print(f"开始计算样本 {sample_idx} 的SHAP值...")

    # 创建解释器，传入原始数据和均值和标准差
    explainer = STFTSHAPExplainer(
        args=type('args', (), {
            'fs': fs,
            'nperseg': nperseg,
            'noverlap': nperseg // 2,  # 统一使用nperseg//2
            'timesteps': data.shape[1],
            'num_bg_samples': 20,
            'device': device,
            'num_samples_per_class': num_samples_per_class,
            'num_clusters_per_class': num_clusters_per_class
        }),
        model=model,
        data=data,
        labels=labels
    )

    # 验证STFT重构质量
    print("验证STFT重构质量...")
    if explainer.validate_stft_reconstruction(sample_idx):
        print("STFT重构验证通过！")
    else:
        print("警告：STFT重构可能存在误差")

    # 确定目标标签
    if data_type == 'train':
        # 训练集使用真实标签
        target_label = labels[sample_idx].item()
        print(f"训练集样本 {sample_idx}，使用真实标签: {target_label}")
    else:
        # 测试集使用模型预测的标签
        sample_data = data[sample_idx]
        target_label = explainer.predict_label(sample_data)
        print(f"测试集样本 {sample_idx}，模型预测标签: {target_label} (真实标签: {labels[sample_idx].item()})")

    # 计算SHAP值
    f, t, test_spec, shap_2d = explainer.compute_shap_for_sample(sample_idx, target_label)

    # 绘制SHAP热力图
    shap_paths = plot_shap_heatmap(f, t, shap_2d, sample_idx, labels[sample_idx].item(), output_path, dataset_name,
                                   data_type=data_type, freq_ratio=freq_ratio)

    return shap_2d, shap_paths


def process_dataset(data_original, labels, args, model, data_type='test'):
    """处理数据集（训练集或测试集）"""
    print(f"\n处理{data_type}数据集...")

    # 数据归一化
    print("数据归一化...")
    data_normalized, means, stds = normalize_data(data_original)

    # 处理样本
    if args.sample_idx >= 0:
        # 处理单个样本
        print(f"处理单个样本 {args.sample_idx}...")

        # 绘制时间序列图
        plot_time_series(data_normalized, labels, args.sample_idx, args.output_path, args.dataset, data_type=data_type)

        # 绘制时频图
        f, t, Zxx = plot_time_frequency(data_normalized, args.sample_idx, labels, args.output_path, args.dataset,
                                        args.nperseg, freq_ratio=args.freq_ratio, data_type=data_type)

        # 计算SHAP热力图
        shap_heatmap, shap_paths = compute_shap_heatmap_new(model, data_original, labels, means, stds, args.sample_idx,
                                                            args.output_path, args.dataset, args.nperseg, args.fs,
                                                            freq_ratio=args.freq_ratio,
                                                            num_samples_per_class=args.num_samples_per_class,
                                                            num_clusters_per_class=args.num_clusters_per_class,
                                                            data_type=data_type)
        print(f"SHAP计算完成，热力图已保存")
    else:
        # 处理所有样本
        print(f"处理所有{data_type}样本...")
        if args.max_samples > 0:
            data_original = data_original[:args.max_samples]
            data_normalized = data_normalized[:args.max_samples]
            labels = labels[:args.max_samples]
            means = means[:args.max_samples]
            stds = stds[:args.max_samples]
            print(f"限制处理前 {args.max_samples} 个样本")

        # 处理所有样本的循环
        for sample_idx in range(len(data_original)):
            print(f"\n处理{data_type}样本 {sample_idx + 1}/{len(data_original)}...")

            try:
                # 绘制时间序列图
                plot_time_series(data_normalized, labels, sample_idx, args.output_path, args.dataset,
                                 data_type=data_type)

                # 绘制时频图
                f, t, Zxx = plot_time_frequency(data_normalized, sample_idx, labels, args.output_path, args.dataset,
                                                args.nperseg, freq_ratio=args.freq_ratio, data_type=data_type)

                # 计算SHAP热力图
                shap_heatmap, shap_paths = compute_shap_heatmap_new(model, data_original, labels, means, stds,
                                                                    sample_idx,
                                                                    args.output_path, args.dataset, args.nperseg,
                                                                    args.fs,
                                                                    freq_ratio=args.freq_ratio,
                                                                    num_samples_per_class=args.num_samples_per_class,
                                                                    num_clusters_per_class=args.num_clusters_per_class,
                                                                    data_type=data_type)

                print(f"{data_type}样本 {sample_idx} 处理完成")

            except Exception as e:
                print(f"处理{data_type}样本 {sample_idx} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue

    print(f"所有{data_type}结果已保存")


def main():
    # 参数解析
    parser = argparse.ArgumentParser(description='时间序列数据分析')
    parser.add_argument('--dataset', type=str, default='cincecgtorso',
                        choices=['toydata_final', 'mixedshapes', 'ACSF', 'Lightning',
                                 'computer', 'yoga', 'RFD', 'midair', 'UMD', 'forda',
                                 'fordb', 'strawberry', 'ECG200', 'cincecgtorso',
                                 'gunpointmalefemale', 'Freezer', 'blink', 'arrowhead',
                                 'EPG', 'EPG1', 'LKA', 'Blink', 'ShapeletSim', 'twopatterns'],
                        help='数据集名称')
    parser.add_argument('--output_path', type=str, default='./image',
                        help='输出文件路径')
    parser.add_argument('--sample_idx', type=int, default=-1,
                        help='要分析的样本索引 (-1表示处理所有样本)')
    parser.add_argument('--nperseg', type=int, default=60,
                        help='STFT窗口长度')
    parser.add_argument('--model_path', type=str,
                        default=r'C:\Users\34517\Desktop\zuhui\xITSC\classification_models\cincecgtorso\transformer\transformer.pt',
                        help='训练好的模型路径')
    parser.add_argument('--max_samples', type=int, default=-1,
                        help='最大处理样本数 (-1表示处理所有样本)')
    parser.add_argument('--num_clusters_per_class', type=int, default=7,
                        help='每个类别进行K-means聚类的簇数')
    parser.add_argument('--num_samples_per_class', type=int, default=7,
                        help='每个类别选择的背景样本数量')
    parser.add_argument('--num_classes', type=int, default=4,
                        help='分类类别数')

    parser.add_argument('--timesteps', type=int, default=1639,
                        help='时间序列长度')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Transformer层数')
    parser.add_argument('--d_model', type=int, default=64,
                        help='模型维度')
    parser.add_argument('--nhead', type=int, default=8,
                        help='注意力头数')
    parser.add_argument('--dim_feedforward', type=int, default=256,
                        help='前馈网络维度')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout率')
    parser.add_argument('--fs', type=float, default=1.0,
                        help='采样频率')
    parser.add_argument('--freq_ratio', type=float, default=1,
                        help='保留的频率分量比例（0-1，如0.66表示保留前2/3低频分量）')
    parser.add_argument('--process_train', action='store_true', default=False,
                        help='是否处理训练集')
    parser.add_argument('--process_test', action='store_true', default=True,
                        help='是否处理测试集')
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(f"{args.output_path}/train", exist_ok=True)
    os.makedirs(f"{args.output_path}/test", exist_ok=True)
    args.device = device

    # 加载模型
    print("加载训练好的模型...")
    model = load_transformer_model(args)

    # 处理训练集
    if args.process_train:
        print(f"\n加载训练数据集: {args.dataset}")
        train_data_original, train_labels = load_train(args.dataset)
        process_dataset(train_data_original, train_labels, args, model, data_type='train')

    # 处理测试集
    if args.process_test:
        print(f"\n加载测试数据集: {args.dataset}")
        test_data_original, test_labels = load_test(args.dataset)
        process_dataset(test_data_original, test_labels, args, model, data_type='test')

    print(f"\n所有处理完成!")
    print(f"训练集结果保存在: {args.output_path}/train")
    print(f"测试集结果保存在: {args.output_path}/test")


if __name__ == "__main__":
    main()