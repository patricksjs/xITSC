import os
import json
import torch
import numpy as np
import random
from collections import defaultdict
from torch import nn
from torch.utils.data import DataLoader
from data.dataloader import load_test
import argparse
from models.models import TransformerModel, ResNet1d, BiLSTMModel, BasicBlock1d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def calculate_class_accuracy(true_labels, predicted_labels, num_classes):
    """
    计算每个类别的准确率
    """
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    class_accuracy = {}

    for true, pred in zip(true_labels, predicted_labels):
        class_total[true] += 1
        if true == pred:
            class_correct[true] += 1

    for class_id in range(num_classes):
        if class_total[class_id] > 0:
            class_accuracy[class_id] = class_correct[class_id] / class_total[class_id]
        else:
            class_accuracy[class_id] = 0.0

    return class_accuracy


def generate_prediction_json(args, output_path):
    # 修改这里：只接收2个返回值
    data, labels = load_test(args.dataset)

    # 手动创建数据集
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, data, labels):
            self.data = data
            self.labels = labels

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]

    ds = SimpleDataset(data, labels)
    data_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    # 初始化模型（根据模型类型选择）
    if args.model_type == 'transformer':
        net = TransformerModel(args, num_classes=args.num_classes).to(device)
    elif args.model_type == 'bilstm':
        net = BiLSTMModel(args, num_classes=args.num_classes).to(device)
    elif args.model_type == 'resnet':
        # 添加ResNet的特定参数
        if not hasattr(args, 'inplanes'):
            args.inplanes = 64
        if not hasattr(args, 'use_transformer'):
            args.use_transformer = False
        net = ResNet1d(BasicBlock1d, [3, 4, 6, 3], args, num_classes=args.num_classes).to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    # 加载预训练模型权重
    print(f"加载模型: {args.classification_model}")
    try:
        from torch.serialization import safe_globals
        with safe_globals([argparse.Namespace]):
            checkpoint = torch.load(args.classification_model, map_location=device, weights_only=False)
    except Exception as e:
        print(f"使用兼容模式加载模型: {e}")
        checkpoint = torch.load(args.classification_model, map_location=device, weights_only=False)

    # 检查checkpoint的键
    print(f"Checkpoint keys: {checkpoint.keys()}")

    # 尝试加载模型权重
    if 'model_state_dict' in checkpoint:
        model_weights = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        model_weights = checkpoint['model']
    elif 'state_dict' in checkpoint:
        model_weights = checkpoint['state_dict']
    else:
        model_weights = checkpoint

    # 处理权重名称不匹配的问题
    if args.model_type == 'transformer':
        # 对于Transformer模型，检查initial_linear层的尺寸
        if 'initial_linear.weight' in model_weights:
            expected_input_size = net.initial_linear.in_features if net.initial_linear else None
            actual_input_size = model_weights['initial_linear.weight'].size(1)
            if expected_input_size and expected_input_size != actual_input_size:
                print(f"调整initial_linear输入尺寸: {expected_input_size} -> {actual_input_size}")
                net.initial_linear = nn.Linear(actual_input_size, args.d_model).to(device)

    # 尝试加载权重
    try:
        net.load_state_dict(model_weights, strict=False)
        print("模型权重加载成功（strict=False）")
    except Exception as e:
        print(f"加载失败，尝试手动加载权重: {e}")
        # 手动加载兼容的权重
        net_state_dict = net.state_dict()
        loaded_keys = []
        for name, param in model_weights.items():
            if name in net_state_dict:
                if net_state_dict[name].shape == param.shape:
                    net_state_dict[name].copy_(param)
                    loaded_keys.append(name)
                else:
                    print(f"尺寸不匹配跳过: {name}, 模型: {net_state_dict[name].shape}, 权重: {param.shape}")
            else:
                print(f"键不匹配跳过: {name}")
        print(f"成功加载 {len(loaded_keys)}/{len(net_state_dict)} 个参数")

    net.eval()
    wrong_sample_ids = []

    results = []
    correct_count = 0
    total_count = len(ds)

    # 用于收集所有预测结果，以便计算每个类别的准确率
    all_true_labels = []
    all_predicted_labels = []

    with torch.no_grad():
        for batch_idx, (batch_data, batch_labels) in enumerate(data_loader):
            # 准备数据
            batch_data = batch_data.unsqueeze(1).float().to(device)  # 添加通道维度
            batch_labels = batch_labels.to(device)

            # 模型预测
            outputs = net(batch_data)

            # 获取预测结果
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            predicted_labels = torch.argmax(outputs, dim=1).cpu().numpy()
            true_labels = batch_labels.cpu().numpy()

            # 收集所有标签用于计算类别准确率
            all_true_labels.extend(true_labels.tolist())
            all_predicted_labels.extend(predicted_labels.tolist())

            # 计算每个样本的结果
            batch_start_idx = batch_idx * args.batch_size
            for i in range(len(true_labels)):
                sample_idx = batch_start_idx + i
                true_label = int(true_labels[i])
                predicted_label = int(predicted_labels[i])
                prob = probabilities[i].tolist()

                # 统计正确性
                if predicted_label == true_label:
                    correct_count += 1
                else:
                    wrong_sample_ids.append(sample_idx)

                # 构建结果字典（暂时不包含accuracy字段，会在后续计算后添加）
                result = {
                    "id": sample_idx,
                    "label": true_label,
                    "predicted": predicted_label,
                    "probabilities": prob,
                    "logits": outputs[i].cpu().numpy().tolist(),
                    # accuracy字段将在计算类别准确率后添加
                }
                results.append(result)

            # 打印进度
            if (batch_idx + 1) % 10 == 0:
                print(f"处理完批次 {batch_idx + 1}/{len(data_loader)}")

    # 计算每个类别的准确率
    class_accuracy = calculate_class_accuracy(all_true_labels, all_predicted_labels, args.num_classes)

    # 创建格式化的Accuracy字典
    accuracy_dict = {}
    for class_id, acc in sorted(class_accuracy.items()):
        key = f"label {class_id}"
        accuracy_dict[key] = acc

    print("\n" + "=" * 50)
    print("每个类别的准确率:")
    for key, acc in accuracy_dict.items():
        print(f"  {key}: {acc:.4f}")
    print("=" * 50 + "\n")

    # 为每个样本添加Accuracy字段
    for result in results:
        # 注意：这里的Accuracy是字典格式，包含所有label的准确率
        result["Accuracy"] = accuracy_dict.copy()

    # 计算整体准确率
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    print("=" * 50)
    print(f"预测完成！")
    print(f"总样本数：{total_count}")
    print(f"正确样本数：{correct_count}")
    print(f"预测准确率：{accuracy:.4f}（{correct_count}/{total_count}）")
    print("=" * 50)
    print(f"错误分类样本数量：{len(wrong_sample_ids)}")
    if wrong_sample_ids:
        print(f"错误分类样本ID：{wrong_sample_ids[:]}")

    # 保存结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"结果已保存至 {output_path}，共 {len(results)} 个样本")

    # 打印样本示例
    if results:
        print("\n示例样本输出:")
        print(json.dumps(results[0], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--classification_model', type=str,
                        default=r"C:\Users\34517\Desktop\zuhui\xITSC\classification_models\computer\transformer\transformer.pt",
                        help='trained classifier model for testing')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--dataset', type=str, default='computer', help="Dataset to process")
    parser.add_argument('--model_type', type=str, default="transformer", choices=['resnet', 'transformer', 'bilstm'])
    parser.add_argument('--num_classes', type=int, default=2)

    # Transformer模型参数
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--dim_feedforward', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--timesteps', type=int, default=720)

    # ResNet模型参数
    parser.add_argument('--inplanes', type=int, default=64, help='ResNet initial channels')
    parser.add_argument('--use_transformer', action='store_true', help='Use transformer in ResNet')

    # 频谱图参数
    parser.add_argument('--fs', type=int, default=100)
    parser.add_argument('--noverlap', type=int, default=4)
    parser.add_argument('--nperseg', type=int, default=8)

    # 输出文件路径
    parser.add_argument('--output_json', type=str, default='computer_transformer.json',
                        help='Path to save the computer-1-shap JSON file')

    args = parser.parse_args()
    generate_prediction_json(args, args.output_json)