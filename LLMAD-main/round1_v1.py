import os
import matplotlib.pyplot as plt
import torch
from collections import defaultdict
from openai import OpenAI

from dataloader_v2 import TimeSeriesDataset
from dataloader_v2 import load_data

# 初始化 OpenAI 客户端
OPENAI_API_KEY = "sk-9BZTN28uw5bGVKSub25xPCymSqovhctKlcR3GQcW4vnzEVNe"
client = OpenAI(api_key=OPENAI_API_KEY, base_url="https://api.chatanywhere.tech/v1")
dataset_name = "Computers"
domain = ("This dataset were taken from data recorded as part of government sponsored study called Powering the "
          "Nation. The intention was to collect behavioural data about how consumers use electricity within the home "
          "to help reduce the UK's carbon footprint. The data contains readings from 251 households, sampled in "
          "two-minute intervals over a month. Each series is length 720 (24 hours of readings taken every 2 minutes). "
          "Classes are Desktop(Class 0) and Laptop(Class 1)")
num = 5


def get_class_to_indices(dataset):
    """构建类别 -> 索引列表的映射"""
    class_to_idxs = defaultdict(list)
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        if torch.is_tensor(label):
            label = label.item()
        class_to_idxs[int(label)].append(idx)
    return class_to_idxs


def plot_and_save_series(data, idx, label, save_path):
    """绘制单个时间序列并保存"""
    if torch.is_tensor(data):
        data = data.cpu().numpy()
    plt.figure(figsize=(8, 2))
    if data.ndim == 1:
        plt.plot(data, color='black')
    else:
        for c in range(data.shape[0]):
            plt.plot(data[c])
    plt.title(f"Idx: {idx} | Class: {label}")
    plt.axis('off')  # 去掉坐标轴，让模型专注波形
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def encode_image_to_base64(image_path):
    import base64
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def generate_class_description(class_label, image_paths):
    """调用 GPT-4o Vision 生成类别描述"""
    content = [
        {
            "type": "text",
            "text": (
                "### Task Description\n"
                "You are an expert in time series analysis. You are given a time series classification task with the"
                f"""{dataset_name} dataset. {domain}."""
                f"""You will be provided with {num} time series samples from Class {class_label}"""
                "Your first task is to analyze the significant pattern of this class."
                "### Analysis Task\n"
                f"""Describe the common patterns, shapes, trends, or characteristics of this class (Class {class_label})."""
                "Focus on similarity in periodicity, shape, spikes, smoothness, oscillations, amplitude, rate of change, etc."
                "Ignore absolute value scale unless it clearly distinguishes classes."
                "Break the series into meaningful segments if applicable."
                "Keep the description concise in 200 words."
                "### Answer Format\n"
                f"-- pattern of class {class_label} --: "
            )
        }
    ]
    for path in image_paths:
        base64_image = encode_image_to_base64(path)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{base64_image}"}
        })

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.2,
        messages=[
            {"role": "user", "content": content}
        ]
    )
    return response.choices[0].message.content.strip()


def analyze_classes_with_gpt4v(train_dataset, selected_positions,
                               output_dir="gpt4v_class_analysis"):
    os.makedirs(output_dir, exist_ok=True)

    class_to_idxs = get_class_to_indices(train_dataset)
    class_descriptions = {}

    for class_label, all_idxs in sorted(class_to_idxs.items()):
        print(f"\nProcessing Class {class_label} (total samples: {len(all_idxs)})")

        # 选取指定位置的索引（注意：positions 是在该类内部的顺序）
        selected_samples = []
        image_paths = []

        for pos in selected_positions:
            if pos < len(all_idxs):
                global_idx = all_idxs[pos]
                data, _ = train_dataset[global_idx]
                img_path = os.path.join(output_dir, f"class{class_label}_pos{pos}_idx{global_idx}.png")
                plot_and_save_series(data, global_idx, class_label, img_path)
                selected_samples.append(global_idx)
                image_paths.append(img_path)
            else:
                print(f"  Warning: Position {pos} exceeds class size ({len(all_idxs)}). Skipped.")

        if not image_paths:
            print(f"  No valid samples for Class {class_label}. Skipping.")
            continue

        # 调用 GPT-4o
        print(f"  Sending {len(image_paths)} images to GPT-4o...")
        try:
            desc = generate_class_description(class_label, image_paths)
            class_descriptions[class_label] = desc
            print(f"  ✅ Description: {desc}")




        except Exception as e:
            print(f"  ❌ Error: {e}")
            class_descriptions[class_label] = "Error generating description."

    # 保存所有描述到文本文件
    # with open(os.path.join(output_dir, "class_descriptions.txt"), "w", encoding="utf-8") as f:
    #     for cls, desc in class_descriptions.items():
    #         f.write(f"Class {cls}:\n{desc}\n{'-' * 50}\n")
    print(f"\n✅ All descriptions saved to {output_dir}/class_descriptions.txt")
    return class_descriptions





if __name__ == '__main__':
    # 假设你已有 train_dataset
    data_splits = load_data("computers")
    train_dataset = TimeSeriesDataset(*data_splits['train'])

    # 运行分析
    descriptions = analyze_classes_with_gpt4v(
        train_dataset,
        selected_positions=[5, 10, 15, 20, 25],
        output_dir="gpt4v_computers_analysis"
    )
