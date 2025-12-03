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
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter, label, binary_dilation
from itertools import combinations
import math

from models.models import TransformerModel

# 设置随机种子
random.seed(42)
torch.set_num_threads(32)
torch.manual_seed(911)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(data_name, args):
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


def sample_backgroundIdentification(f, t, original_spectrogram, original_signal, args):
    """按照您提供的背景识别方法计算背景值"""
    frequency_composition_abs = np.abs(original_spectrogram)
    measures = []
    for freq, freq_composition in zip(f, frequency_composition_abs):
        measures.append(np.mean(freq_composition) / np.std(freq_composition))
    max_value = max(measures)
    selected_frequency = measures.index(max_value)
    dummymatrix = np.zeros((len(f), len(t)))
    dummymatrix[selected_frequency, :] = 1

    background_frequency = original_spectrogram * dummymatrix
    background_frequency = torch.tensor(background_frequency)
    _, xrec = istft(background_frequency.numpy(), args.fs, nperseg=args.nperseg, noverlap=args.noverlap,
                    boundary='zeros')
    xrec = xrec[:original_signal.shape[0]]
    xrec = xrec.reshape(original_signal.shape)
    return xrec, background_frequency


def normalize_data(data):
    """将数据归一化到0-1范围 (使用整个数据集的全局最大/最小值)"""
    if isinstance(data, torch.Tensor):
        data_np = data.numpy()
    else:
        data_np = data

    # 计算整个数据集的全局最小值和最大值
    global_min = np.min(data_np)
    global_max = np.max(data_np)

    print(f"全局最小值: {global_min:.4f}, 全局最大值: {global_max:.4f}")

    # 避免除零错误
    if global_max - global_min == 0:
        print("警告: 数据集所有值相同，返回原数据")
        return data

    # 使用全局参数进行归一化
    normalized_data_np = (data_np - global_min) / (global_max - global_min)

    # 验证常数序列是否保持常数
    for i in range(min(5, len(data_np))):  # 检查前5个样本
        sample = data_np[i]
        if np.all(sample == sample[0]):  # 如果是常数序列
            normalized_sample = normalized_data_np[i]
            if not np.all(normalized_sample == normalized_sample[0]):
                print(f"警告: 样本 {i} 是常数序列但归一化后出现波动")
                print(
                    f"原始值: {sample[0]:.4f}, 归一化后范围: {normalized_sample.min():.4f} ~ {normalized_sample.max():.4f}")

    return torch.tensor(normalized_data_np, dtype=torch.float32)


def plot_time_series(data, labels, sample_idx, output_path, dataset_name):
    """绘制时间序列图"""
    plt.figure(figsize=(12, 6))
    sample = data[sample_idx].numpy()
    time_steps = np.arange(len(sample))

    n_ticks = 12
    tick_indices = np.linspace(0, len(sample) - 1, n_ticks, dtype=int)
    tick_labels = [f"{i}" for i in tick_indices]

    plt.plot(time_steps, sample, linewidth=2)
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Normalized Value', fontsize=12)
    plt.title(f'Time Series Plot - {dataset_name}\nSample {sample_idx}, Label: {labels[sample_idx].item()}',
              fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(tick_indices, tick_labels)
    plt.ylim(0, 1)

    plt.tight_layout()

    label_dir = f"{output_path}/{dataset_name}_{labels[sample_idx].item()}"
    os.makedirs(label_dir, exist_ok=True)
    plt.savefig(f"{label_dir}/plot_sample{sample_idx}.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_time_frequency(data, sample_idx, labels, output_path, dataset_name, nperseg=64):
    """绘制时频图"""
    plt.figure(figsize=(12, 8))
    sample = data[sample_idx].numpy()

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

    n_ticks = 12
    tick_indices = np.linspace(0, len(t) - 1, n_ticks, dtype=int)
    tick_positions = t[tick_indices]
    tick_labels = [f"{pos:.1f}" for pos in tick_positions]

    plt.pcolormesh(t, f, spec_magnitude, shading='auto', cmap='viridis')
    plt.colorbar(label='Signal Magnitude')
    plt.xlabel('Time (Normalized)', fontsize=12)
    plt.ylabel('Frequency (Normalized)', fontsize=12)
    plt.title(f'Time-Frequency Image - {dataset_name}\nSample {sample_idx}, Label: {labels[sample_idx].item()}',
              fontsize=14)
    plt.xticks(tick_positions, tick_labels, rotation=45, ha='right')
    plt.tight_layout()

    label_dir = f"{output_path}/{dataset_name}_{labels[sample_idx].item()}"
    os.makedirs(label_dir, exist_ok=True)
    plt.savefig(f"{label_dir}/STFT_sample{sample_idx}.png", dpi=300, bbox_inches='tight')
    plt.close()

    return f, t, Zxx


def detect_high_energy_regions_correct(Zxx, f, t, min_region_size=4):
    """
    正确的高能量区域检测方法
    检测大于均值的区域，然后合并相邻像素
    """
    magnitude = np.abs(Zxx)
    power = magnitude ** 2

    # 计算全局均值
    global_mean = np.mean(power)

    # 创建二值掩码：大于均值的区域
    binary_mask = power > global_mean

    # 标记连通区域
    labeled_array, num_features = label(binary_mask)

    regions = []
    for i in range(1, num_features + 1):
        # 获取当前区域的坐标
        region_coords = np.where(labeled_array == i)

        if len(region_coords[0]) < min_region_size:
            continue  # 跳过太小的区域

        time_indices = region_coords[1]
        freq_indices = region_coords[0]

        min_time_idx, max_time_idx = np.min(time_indices), np.max(time_indices)
        min_freq_idx, max_freq_idx = np.min(freq_indices), np.max(freq_indices)

        # 计算区域平均能量
        region_power = power[freq_indices, time_indices]
        avg_power = np.mean(region_power)
        total_power = np.sum(region_power)

        region_info = {
            'time_range': (min_time_idx, max_time_idx),
            'freq_range': (min_freq_idx, max_freq_idx),
            'avg_power': avg_power,
            'total_power': total_power,
            'size': len(region_coords[0]),
            'coords': (freq_indices, time_indices)
        }
        regions.append(region_info)

    # 按总能量排序
    regions.sort(key=lambda x: x['total_power'], reverse=True)

    print(f"检测到 {len(regions)} 个高能量区域")

    return regions


def create_low_energy_features(Zxx, high_energy_regions):
    """
    创建低能量像素块特征
    每个低能量像素作为一个单独的特征
    """
    magnitude = np.abs(Zxx)
    power = magnitude ** 2

    # 创建高能量区域掩码
    high_energy_mask = np.zeros(Zxx.shape, dtype=bool)
    for region in high_energy_regions:
        t_min, t_max = region['time_range']
        f_min, f_max = region['freq_range']
        high_energy_mask[f_min:f_max + 1, t_min:t_max + 1] = True

    # 低能量像素坐标
    low_energy_coords = np.where(~high_energy_mask)
    low_energy_features = []

    for i in range(len(low_energy_coords[0])):
        f_idx = low_energy_coords[0][i]
        t_idx = low_energy_coords[1][i]

        feature_info = {
            'type': 'low_energy_pixel',
            'coords': ([f_idx], [t_idx]),
            'time_range': (t_idx, t_idx),
            'freq_range': (f_idx, f_idx),
            'power': power[f_idx, t_idx]
        }
        low_energy_features.append(feature_info)

    print(f"创建了 {len(low_energy_features)} 个低能量像素特征")
    return low_energy_features


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


def _get_prediction_probability(model, sample, true_label):
    """获取模型对指定样本的预测概率"""
    with torch.no_grad():
        sample_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(device)
        if sample_tensor.ndim == 2:
            sample_tensor = sample_tensor.unsqueeze(-1)
        pred = torch.softmax(model(sample_tensor), dim=1)
        return pred[0, true_label].item()


def _replace_all_features_with_background(sample, all_features, background_Zxx, Zxx_shape, nperseg, fs):
    """将所有特征替换为背景值"""
    return _replace_specific_features_with_background(sample, all_features, [], background_Zxx, Zxx_shape, nperseg, fs)


def _replace_specific_features_with_background(sample, all_features, features_to_keep, background_Zxx, Zxx_shape,
                                               nperseg, fs):
    """
    只保留指定的特征，其他特征替换为背景值
    features_to_keep: 要保留的特征索引列表
    """
    # 计算样本的STFT
    f, t, Zxx_sample = stft(
        sample,
        fs=fs,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        window='hann',
        boundary='zeros'
    )

    # 创建掩蔽矩阵（初始化为背景值）
    Zxx_masked = background_Zxx.copy()

    # 只保留指定的特征
    for feature_idx in features_to_keep:
        feature = all_features[feature_idx]
        freq_indices, time_indices = feature['coords']

        # 确保索引在有效范围内
        valid_freq = [f for f in freq_indices if f < Zxx_sample.shape[0]]
        valid_time = [t for t in time_indices if t < Zxx_sample.shape[1]]

        if valid_freq and valid_time:
            # 创建网格索引
            freq_grid, time_grid = np.meshgrid(valid_freq, valid_time, indexing='ij')
            Zxx_masked[freq_grid, time_grid] = Zxx_sample[freq_grid, time_grid]

    # 重构时域信号
    _, sample_reconstructed = istft(
        Zxx_masked,
        fs=fs,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        boundary='zeros'
    )

    # 调整长度匹配原始样本
    if len(sample_reconstructed) > len(sample):
        sample_reconstructed = sample_reconstructed[:len(sample)]
    elif len(sample_reconstructed) < len(sample):
        sample_reconstructed = np.pad(
            sample_reconstructed,
            (0, len(sample) - len(sample_reconstructed)),
            mode='constant',
            constant_values=0
        )

    return sample_reconstructed


def compute_shap_values_correct(model, sample, true_label, all_features, background_Zxx, f, t, Zxx_shape, nperseg, fs):
    """
    正确计算SHAP值，严格按照SHAP定义
    """
    # 特征总数
    n_features = len(all_features)
    print(f"开始计算 {n_features} 个特征的SHAP值...")

    # 计算原始预测概率（所有特征都存在）
    original_prob = _get_prediction_probability(model, sample, true_label)
    print(f"原始预测概率 (所有特征): {original_prob:.4f}")

    # 计算基准预测概率（所有特征都替换为背景值）
    baseline_sample = _replace_all_features_with_background(
        sample, all_features, background_Zxx, Zxx_shape, nperseg, fs
    )
    baseline_prob = _get_prediction_probability(model, baseline_sample, true_label)
    print(f"基准预测概率 (无特征): {baseline_prob:.4f}")

    # 初始化SHAP值
    shap_values = np.zeros(n_features)

    # 对于每个特征，计算其SHAP值
    for target_feature_idx in range(n_features):
        print(f"计算特征 {target_feature_idx + 1}/{n_features} 的SHAP值...")

        total_weighted_contribution = 0.0

        # 获取不包含目标特征的所有特征索引
        other_features = [i for i in range(n_features) if i != target_feature_idx]

        # 遍历所有可能的子集大小
        for subset_size in range(len(other_features) + 1):
            # 获取所有大小为subset_size的子集
            for subset_indices in combinations(other_features, subset_size):
                # 子集S（不包含目标特征i）
                subset_S = list(subset_indices)

                # 子集S ∪ {i}（包含目标特征i）
                subset_S_with_i = subset_S + [target_feature_idx]

                # 计算f(S) - 子集S的预测
                sample_S = _replace_specific_features_with_background(
                    sample, all_features, subset_S, background_Zxx, Zxx_shape, nperseg, fs
                )
                prob_S = _get_prediction_probability(model, sample_S, true_label)

                # 计算f(S ∪ {i}) - 子集S加上特征i的预测
                sample_S_with_i = _replace_specific_features_with_background(
                    sample, all_features, subset_S_with_i, background_Zxx, Zxx_shape, nperseg, fs
                )
                prob_S_with_i = _get_prediction_probability(model, sample_S_with_i, true_label)

                # 边际贡献 = f(S ∪ {i}) - f(S)
                marginal_contribution = prob_S_with_i - prob_S

                # 计算权重：|S|! * (n - |S| - 1)! / n!
                weight = math.factorial(len(subset_S)) * math.factorial(
                    n_features - len(subset_S) - 1) / math.factorial(n_features)

                total_weighted_contribution += weight * marginal_contribution

        shap_values[target_feature_idx] = total_weighted_contribution
        print(f"  特征 {target_feature_idx + 1}: SHAP值 = {shap_values[target_feature_idx]:.4f}")

    # 验证SHAP值性质
    sum_shap = np.sum(shap_values)
    expected_sum = original_prob - baseline_prob
    print(f"SHAP值总和: {sum_shap:.4f}")
    print(f"期望总和 (原始-基准): {expected_sum:.4f}")
    print(f"差异: {abs(sum_shap - expected_sum):.4f}")

    return shap_values, original_prob, baseline_prob


def _create_detailed_shap_heatmap(shap_values, all_features, Zxx_shape):
    """创建详细的SHAP热力图"""
    shap_heatmap = np.zeros(Zxx_shape, dtype=float)

    for i, feature in enumerate(all_features):
        freq_indices, time_indices = feature['coords']

        # 确保索引在有效范围内
        valid_freq = [f for f in freq_indices if f < Zxx_shape[0]]
        valid_time = [t for t in time_indices if t < Zxx_shape[1]]

        if valid_freq and valid_time:
            # 创建网格索引
            freq_grid, time_grid = np.meshgrid(valid_freq, valid_time, indexing='ij')
            shap_heatmap[freq_grid, time_grid] = shap_values[i]

    return shap_heatmap


def _plot_correct_shap_heatmap(shap_heatmap, f, t, sample_idx, label, output_path, dataset_name,
                               high_energy_regions, low_energy_features, shap_values,
                               original_prob, baseline_prob):
    """绘制正确的SHAP热力图"""
    plt.figure(figsize=(15, 10))

    # 计算颜色范围
    shap_min = np.min(shap_values)
    shap_max = np.max(shap_values)
    vmax = max(abs(shap_min), abs(shap_max))
    vmin = -vmax

    # 绘制热力图
    im = plt.pcolormesh(t, f, shap_heatmap, cmap='bwr', shading='gouraud',
                        vmin=vmin, vmax=vmax)

    plt.colorbar(im, label='SHAP Value (Feature Importance)')
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Frequency [Hz]', fontsize=12)

    # 添加统计信息
    total_positive = np.sum([v for v in shap_values if v > 0])
    total_negative = np.sum([v for v in shap_values if v < 0])
    n_high_energy = len(high_energy_regions)
    n_low_energy = len(low_energy_features)

    plt.title(f'Correct SHAP Heatmap - {dataset_name}\n'
              f'Sample {sample_idx}, Label: {label}\n'
              f'Baseline: {baseline_prob:.4f}, Original: {original_prob:.4f}\n'
              f'High-energy regions: {n_high_energy}, Low-energy pixels: {n_low_energy}\n'
              f'Total +SHAP: {total_positive:.4f}, Total -SHAP: {total_negative:.4f}',
              fontsize=12)

    # 设置横坐标刻度
    n_ticks = min(12, len(t))
    tick_indices = np.linspace(0, len(t) - 1, n_ticks, dtype=int)
    tick_positions = t[tick_indices]
    tick_labels = [f"{pos:.1f}" for pos in tick_positions]
    plt.xticks(tick_positions, tick_labels)

    # 添加高能量区域边界
    for region_idx, region in enumerate(high_energy_regions):
        t_min, t_max = region['time_range']
        f_min, f_max = region['freq_range']

        # 绘制矩形边界
        rect = plt.Rectangle(
            (t[t_min], f[f_min]),
            t[t_max] - t[t_min],
            f[f_max] - f[f_min],
            fill=False,
            edgecolor='green',
            linewidth=2,
            linestyle='-'
        )
        plt.gca().add_patch(rect)

        # 在区域中心添加SHAP值
        center_x = t[t_min] + (t[t_max] - t[t_min]) / 2
        center_y = f[f_min] + (f[f_max] - f[f_min]) / 2
        shap_val = shap_values[region_idx]

        plt.text(center_x, center_y, f'R{region_idx + 1}: {shap_val:.3f}',
                 ha='center', va='center', fontsize=8,
                 fontweight='bold', color='green',
                 bbox=dict(boxstyle="round,pad=0.2",
                           facecolor='white', alpha=0.8))

    plt.tight_layout()

    # 保存图像
    label_dir = f"{output_path}/{dataset_name}_{label}"
    os.makedirs(label_dir, exist_ok=True)
    plt.savefig(f"{label_dir}/correct_shap_sample{sample_idx}.png",
                dpi=300, bbox_inches='tight')
    plt.close()

    # 打印详细统计
    print(f"\n样本 {sample_idx} 详细SHAP统计:")
    print(f"基准概率: {baseline_prob:.4f}")
    print(f"原始概率: {original_prob:.4f}")
    print(f"SHAP值总和: {np.sum(shap_values):.4f}")
    print(f"验证: 基准 + SHAP总和 = {baseline_prob + np.sum(shap_values):.4f}")

    # 高能量区域SHAP值
    print("\n高能量区域SHAP值:")
    for i in range(len(high_energy_regions)):
        print(f"  区域 {i + 1}: SHAP = {shap_values[i]:.4f}")

    # 低能量像素统计（由于数量多，只统计前10个）
    print(f"\n低能量像素SHAP值 (前10个):")
    for i in range(len(high_energy_regions), min(len(high_energy_regions) + 10, len(shap_values))):
        print(f"  像素 {i - len(high_energy_regions) + 1}: SHAP = {shap_values[i]:.4f}")


def compute_shap_heatmap_correct(model, data, labels, sample_idx, f, t, Zxx, output_path, dataset_name, nperseg=64,
                                 fs=1.0):
    """正确计算SHAP热力图"""
    print(f"开始计算样本 {sample_idx} 的正确SHAP值...")

    sample = data[sample_idx].numpy()
    true_label = labels[sample_idx].item()

    # 1. 检测高能量区域
    high_energy_regions = detect_high_energy_regions_correct(Zxx, f, t)

    # 2. 创建低能量像素特征
    low_energy_features = create_low_energy_features(Zxx, high_energy_regions)

    # 3. 组合所有特征
    all_features = high_energy_regions + low_energy_features

    # 4. 计算背景值
    background_signal, background_Zxx = sample_backgroundIdentification(f, t, Zxx, sample,
                                                                        type('args', (), {'fs': fs, 'nperseg': nperseg,
                                                                                          'noverlap': nperseg // 2}))

    # 5. 计算SHAP值
    shap_values, original_prob, baseline_prob = compute_shap_values_correct(
        model, sample, true_label, all_features, background_Zxx.numpy(),
        f, t, Zxx.shape, nperseg, fs
    )

    # 6. 创建SHAP热力图
    shap_heatmap = _create_detailed_shap_heatmap(shap_values, all_features, Zxx.shape)

    # 7. 绘制SHAP热力图
    _plot_correct_shap_heatmap(shap_heatmap, f, t, sample_idx, true_label, output_path,
                               dataset_name, high_energy_regions, low_energy_features,
                               shap_values, original_prob, baseline_prob)

    return shap_heatmap


# 在main函数中替换原来的SHAP计算调用
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
    parser.add_argument('--sample_idx', type=int, default=-1,  # 修改默认值为0，处理第一个样本
                        help='要分析的样本索引 (-1表示处理所有样本)')
    parser.add_argument('--nperseg', type=int, default=16,
                        help='STFT窗口长度')
    parser.add_argument('--model_path', type=str,
                        default=r'C:\Users\34517\Desktop\zuhui\Time_is_not_Enough-main\classification_models\cincecgtorso\transformer\transformer.pt',
                        help='训练好的模型路径')
    parser.add_argument('--max_samples', type=int, default=-1,
                        help='最大处理样本数 (-1表示处理所有样本)')

    parser.add_argument('--num_classes', type=int, default=4,
                        help='分类类别数')
    parser.add_argument('--num_bg_samples', type=int, default=20,
                        help='SHAP背景样本数量')
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
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_path, exist_ok=True)
    args.device = device

    # 加载数据
    print(f"加载数据集: {args.dataset}")
    data, labels = load_data(args.dataset, args)

    # 数据归一化
    print("数据归一化...")
    data_normalized = normalize_data(data)

    # 加载模型
    print("加载训练好的模型...")
    model = load_transformer_model(args)

    # 处理样本
    if args.sample_idx >= 0:
        # 处理单个样本
        print(f"处理单个样本 {args.sample_idx}...")

        # 绘制时间序列图
        plot_time_series(data_normalized, labels, args.sample_idx, args.output_path, args.dataset)

        # 绘制时频图
        f, t, Zxx = plot_time_frequency(data_normalized, args.sample_idx, labels, args.output_path, args.dataset,
                                        args.nperseg)

        # 计算SHAP热力图
        shap_heatmap = compute_shap_heatmap_correct(model, data_normalized, labels, args.sample_idx, f, t, Zxx,
                                                    args.output_path, args.dataset, args.nperseg, fs=1.0)
        print(f"SHAP计算完成，热力图已保存")
    else:
        # 处理所有样本
        print("处理所有样本...")
        if args.max_samples > 0:
            data_normalized = data_normalized[:args.max_samples]
            labels = labels[:args.max_samples]
            print(f"限制处理前 {args.max_samples} 个样本")

        # 这里添加处理所有样本的循环
        for sample_idx in range(len(data_normalized)):
            print(f"\n处理样本 {sample_idx + 1}/{len(data_normalized)}...")

            try:
                # 绘制时间序列图
                plot_time_series(data_normalized, labels, sample_idx, args.output_path, args.dataset)

                # 绘制时频图
                f, t, Zxx = plot_time_frequency(data_normalized, sample_idx, labels, args.output_path, args.dataset,
                                                args.nperseg)

                # 计算SHAP热力图
                shap_heatmap = compute_shap_heatmap_correct(model, data_normalized, labels, sample_idx, f, t, Zxx,
                                                            args.output_path, args.dataset, args.nperseg, fs=1.0)
                print(f"样本 {sample_idx} 处理完成")

            except Exception as e:
                print(f"处理样本 {sample_idx} 时出错: {e}")
                continue

    print(f"所有结果已保存到: {args.output_path}")


if __name__ == "__main__":
    main()