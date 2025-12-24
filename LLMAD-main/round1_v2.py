import os
import torch
from collections import defaultdict
from openai import OpenAI
import base64

from dataloader_v2 import load_data, TimeSeriesDataset

# 初始化 OpenAI 客户端
OPENAI_API_KEY = "sk-VdhI38Ualpqsc1yxyYN3Is82AWrNOtvhMjgxUihamFZSAf7k"
client = OpenAI(api_key=OPENAI_API_KEY, base_url="https://api.chatanywhere.tech/v1")
dataset_name = "BME"
# domain = ("This dataset were taken from data recorded as part of government sponsored study called Powering the "
#           "Nation. The intention was to collect behavioural data about how consumers use electricity within the home "
#           "to help reduce the UK's carbon footprint. The data contains readings from 251 households, sampled in "
#           "two-minute intervals over a month. Each series is length 720 (24 hours of readings taken every 2 minutes). "
#           "Classes are Desktop(Class 0) and Laptop(Class 1)")
domain = ("BME (Begin-Middle-End) is a synthetic univariate data set with three classes: one class is characterized "
          "by a small positive bell arising at the initial period (Begin), one does not have any bell (Middle), "
          "one has a positive bell arising at the final period (End)."
          "All series are constituted by a central plate. The central plates may be positive or negative. The "
          "discriminant is the presence or absence of a positive peak, or at the beginning of series or at the end."
          "Class 0: Begin, Class 1: Middle, Class 2: End")

num = 5
num_compare = 5


def get_class_to_indices(dataset):
    class_to_idxs = defaultdict(list)
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        if torch.is_tensor(label):
            label = label.item()
        class_to_idxs[int(label)].append(idx)
    return class_to_idxs


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def pre_generate_comparison_images(class_to_idxs, positions):
    """
    为每个类预生成用于对比的图像（pos 5,10,15），返回 class_label -> [img_paths]
    """
    class_image_map = {}
    for class_label, idxs in class_to_idxs.items():
        img_paths = []
        for pos in positions[class_label]:
            img_path = os.path.join(f"plots/{dataset_name}_train/class{class_label}/{pos}.png")

            img_paths.append(img_path)
        class_image_map[class_label] = img_paths
    return class_image_map


# === 修改：生成带对比的描述 ===
def generate_class_description_with_comparison(
        class_label,
        own_image_paths,
        other_class_images,  # dict: {other_label: [img_paths]}
):
    # === Step 1: 生成自身模式描述 ===
    prompt1 = ("### Task Description\n"
               "You are an expert in time series analysis. You are given a time series classification task with the"
               f"""{dataset_name} dataset. {domain}."""
               f"""You will be provided with {num} time series samples from Class {class_label}"""
               "Your first task is to analyze the significant pattern of this class."
               "### Analysis Task\n"
               f"""Describe the common patterns, shapes, trends, or characteristics of this class (Class {class_label})."""
               "Focus on similarity in periodicity, shape, spikes, smoothness, oscillations, amplitude, rate of change, etc."
               "Describe the overall characteristics of the data first and then break the series into meaningful segments to describe local features if applicable."
               "Keep the description concise in 200 words."
               "### Answer Format\n"
               f"-- pattern of class {class_label} --: ")

    content1 = [{"type": "text", "text": prompt1}]
    for p in own_image_paths:
        content1.append(
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image_to_base64(p)}"}})

    resp1 = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": content1}],
        temperature=0.2
    )
    self_desc = resp1.choices[0].message.content.strip()

    # === Step 2: 对比分析（当前类 vs 所有其他类）===
    all_other_labels = list(other_class_images.keys())
    print(all_other_labels)
    if not all_other_labels:
        return self_desc + "\n\n(No other classes for comparison.)"

    prompt2 = f"""### Task Description\n
                You are an expert in time series analysis. You are shown:
                - First, {num_compare} samples from Class {class_label} (your target class).
                - Then, {num_compare} samples from each other classes: {', '.join([f'Class {lbl}' for lbl in all_other_labels])}.
                Compare and summarize the significant differences between Class {class_label} and the other classes.
                Focus on distinction in periodicity, shape, spikes, smoothness, oscillations, amplitude, rate of change, etc: e.g., "Class {class_label} has sharp spikes, while others are smooth
                Keep the difference of each category concise in 200 words.
                ### Answer Format\n
                -- Difference to class xx --:"""

    content2 = [{"type": "text", "text": prompt2}]
    # # 先加自己的图
    # for p in own_image_paths:
    #     content2.append(
    #         {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image_to_base64(p)}"}})
    # 再加其他类的图（每类最多3张）
    for other_lbl, paths in other_class_images.items():
        for p in paths:  # 避免太多图
            content2.append(
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image_to_base64(p)}"}})

    resp2 = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": content2}],
        temperature=0.2
    )
    comp_desc = resp2.choices[0].message.content.strip()

    return f"**Pattern of Class {class_label}:**\n{self_desc}\n\n**Distinctive from other classes:**\n{comp_desc}"


# === 主函数：分析所有类并做对比 ===
def analyze_classes_with_gpt4v(train_dataset,
                               selected_positions,
                               compare_positions,
                               output_dir="gpt4v_class_analysis"):
    os.makedirs(output_dir, exist_ok=True)

    class_to_idxs = get_class_to_indices(train_dataset)
    sorted_classes = sorted(class_to_idxs.keys())

    # 预生成所有类用于对比的图像（pos 5,10,15）
    print("🔄 Pre-generating comparison images for all classes...")
    compare_image_map = pre_generate_comparison_images(
        class_to_idxs, compare_positions
    )

    class_descriptions = {}

    for class_label in sorted_classes:
        print(f"\n🔍 Processing Class {class_label}")

        # 获取自身用于初始描述的图像（pos 5,10,15,20,25）
        own_images = []
        for pos in selected_positions[class_label]:
            img_path = os.path.join(f"plots/{dataset_name}_train/class{class_label}/{pos}.png")

            own_images.append(img_path)

        if not own_images:
            print(f"  ⚠️ No valid samples for Class {class_label}. Skipping.")
            class_descriptions[class_label] = "No samples."
            continue

        # 构建其他类的图像字典（用于对比）
        other_class_imgs = {
            lbl: paths for lbl, paths in compare_image_map.items()
        }

        # 调用增强版描述生成
        try:
            full_desc = generate_class_description_with_comparison(
                class_label=class_label,
                own_image_paths=own_images,
                other_class_images=other_class_imgs
            )
            class_descriptions[class_label] = full_desc
            print(f"  ✅ Generated description for Class {class_label}")
        except Exception as e:
            print(f"  ❌ Error for Class {class_label}: {e}")
            class_descriptions[class_label] = f"Error: {e}"

        # 保存结果
        out_file = os.path.join(output_dir, f"{dataset_name}_{class_label}_initial.txt")
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(class_descriptions[class_label] + "\n")

    return class_descriptions


if __name__ == "__main__":
    # 假设你已有 train_dataset
    data_splits = load_data(dataset_name)
    train_dataset = TimeSeriesDataset(*data_splits['train'])
    selected_positions = data_splits['representative_indices']
    # 假设你有 train_dataset
    descriptions = analyze_classes_with_gpt4v(
        train_dataset=train_dataset,
        selected_positions=[selected_positions[0], selected_positions[1], selected_positions[2]],  # 用于自身描述
        compare_positions=[selected_positions[0][:2], selected_positions[1][:2], selected_positions[2][:2]],  # 用于对比分析
        output_dir="log")
