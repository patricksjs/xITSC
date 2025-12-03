import os
import json
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from data.data_loader import load_data
import argparse
from models.models import TransformerModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def generate_prediction_json(args, output_path):
    # 修改这里：只接收2个返回值
    data, labels = load_data(args.dataset)

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

    # 计算频谱参数（如果需要）
    num_freq = args.nperseg // 2 + 1
    num_slices = data.shape[1] // (args.nperseg - args.noverlap) + 1

    # 初始化模型
    net = TransformerModel(args, num_classes=args.num_classes).to(device)

    # 其余代码保持不变...
    # 加载预训练模型权重
    print(f"加载模型: {args.classification_model}")
    try:
        from torch.serialization import safe_globals
        with safe_globals([argparse.Namespace]):
            checkpoint = torch.load(args.classification_model, map_location=device, weights_only=False)
    except Exception as e:
        print(f"使用兼容模式加载模型: {e}")
        checkpoint = torch.load(args.classification_model, map_location=device, weights_only=False)

    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    results = []
    correct_count = 0
    total_count = len(ds)

    with torch.no_grad():
        for sample_idx in range(total_count):
            # 修改这里：只接收2个值
            data_sample, true_label = ds[sample_idx]

            # 数据预处理
            data_sample = data_sample.unsqueeze(0).unsqueeze(1).float().to(device)
            true_label = true_label.item() if torch.is_tensor(true_label) else true_label
            labels_tensor = torch.tensor([true_label]).long().to(device)

            # 模型预测
            outputs = net(data_sample)
            logits = outputs.cpu().numpy()[0]
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            _, predicted_label = torch.max(outputs, 1)
            predicted_label = predicted_label.item()

            # 统计正确性
            if predicted_label == true_label:
                correct_count += 1

            # 构建结果字典
            result = {
                "id": sample_idx,
                "label": true_label,
                "result": probabilities.tolist(),
                "predicted": predicted_label,
                "logits": logits.tolist()
            }
            results.append(result)

            # 打印进度
            if (sample_idx + 1) % (args.batch_size * 10) == 0:
                print(f"处理完 {sample_idx + 1}/{total_count} 样本")

    # 计算准确率
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    print("=" * 50)
    print(f"预测完成！")
    print(f"总样本数：{total_count}")
    print(f"正确样本数：{correct_count}")
    print(f"预测准确率：{accuracy:.4f}（{correct_count}/{total_count}）")
    print("=" * 50)

    # 保存结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"结果已保存至 {output_path}，共 {len(results)} 个样本")


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
    parser.add_argument('--dropout', type=int, default=0.2)
    parser.add_argument('--timesteps', type=int, default=720)

    # 频谱图参数
    parser.add_argument('--fs', type=int, default=100)
    parser.add_argument('--noverlap', type=int, default=4)
    parser.add_argument('--nperseg', type=int, default=8)

    # 输出文件路径
    parser.add_argument('--output_json', type=str, default='computer_transformer.json',
                        help='Path to save the computer-1-shap JSON file')

    args = parser.parse_args()
    generate_prediction_json(args, args.output_json)