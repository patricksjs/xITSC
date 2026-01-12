import os
import time

import torch
from collections import defaultdict
from openai import OpenAI
import base64
import json
from dataloader_v2 import load_data, TimeSeriesDataset

# 初始化 OpenAI 客户端
OPENAI_API_KEY = "sk-1YNR0jY6LEWEXSMS7abadhNcpHsblvlxDhyZ62GRY16ZTYVV"
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
def generate_class_description(
        class_label,
        own_image_paths,
        top_k
):
    # === Step 1: 生成自身模式描述 ===
    prompt1 = ("### Task Description\n"
               "You are an expert in image pattern recognition. You are given a time series classification task with "
               f"""the {dataset_name} dataset.\n {domain}.\n"""
               f"""You will be provided with {top_k} time series samples from Class {class_label}"""
               "Your task is to analyze and identify significant patterns within these images, summarizing your "
               "findings concisely."
               "### Analysis Task\n"
               f"""Find the common patterns or characteristics among the images."""
               "Focus on similarity in cycle, shape, trend, spikes, oscillations, amplitude, "
               "rate of change, etc. and their location abd frequency."
               "You cam describe the global characteristics of the data, or break the series into meaningful "
               "segments to find the local common features if applicable.\n"
               "Use a bullet-point list for the description, with each point highlighting a distinct pattern or "
               "feature."
               "Do not mention the feature of specific image."
               "Do not exceed 5 points and keep each point less than 50 word."
               "### Answer Format\n"
               "• [Point 1]"
               "• [Point 2]"
               "• [Point 3]"
               "... (continue as needed, with each point in a bullet-point format)")

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

    return f"\n{self_desc}\n"


def analyze_classes_with_gpt4v(train_dataset,
                               selected_positions,
                               top_k,
                               output_dir):
    os.makedirs(output_dir, exist_ok=True)

    class_to_idxs = get_class_to_indices(train_dataset)
    sorted_classes = sorted(class_to_idxs.keys())
    print(sorted_classes)

    class_descriptions = {}

    for class_label in sorted_classes:
        print(f"\n🔍 Processing Class {class_label}")

        clusters = selected_positions[class_label]  # list of clusters, each cluster is a list of image indices
        descriptions_for_class = []

        for i, cluster in enumerate(clusters):
            own_images = []
            for pos in cluster:
                img_path = os.path.join(f"plots/{dataset_name}_train/class{class_label}/{pos}.png")
                if os.path.exists(img_path):
                    own_images.append(img_path)
                else:
                    print(f"  ⚠️ Image not found: {img_path}")

            try:
                full_desc = generate_class_description(
                    class_label=class_label,
                    own_image_paths=own_images,
                    top_k=top_k
                )
                descriptions_for_class.append(full_desc)
                print(f"  ✅ Generated description for Class {class_label}, Cluster {i}")
            except Exception as e:
                error_msg = f"Error in cluster {i}: {e}"
                print(f"  ❌ {error_msg}")

        # Save all descriptions for this class into one file
        class_descriptions[class_label] = descriptions_for_class
        # 使用JSON格式保存，以便于读取为列表
        out_file = os.path.join(output_dir, f"{dataset_name}_{class_label}_initial.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(descriptions_for_class, f, ensure_ascii=False, indent=2)

    return class_descriptions


def summarize_common_patterns_across_classes(
        output_dir,
        train_dataset
):
    class_to_idxs = get_class_to_indices(train_dataset)
    class_labels = sorted(class_to_idxs.keys())

    for label in class_labels:
        file_path = os.path.join(output_dir, f"{dataset_name}_{label}_initial.json")
        if not os.path.exists(file_path):
            print(f"⚠️ Warning: {file_path} not found. Skipping class {label}.")
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            descs = json.load(f)

        # Step 3: 构建 prompt
        prompt = ("### Task Description:\n"
                  "You are an expert in pattern recognition and time series analysis.\n"
                  f"Below are the per-cluster descriptions for class {label} in the '{dataset_name}' dataset.\n {domain} \n"
                  "Your task is to identify **common patterns or shared characteristics** across different clusters.\n"
                  "Focus on similarity in cycle, shape, trend, spikes, oscillations, amplitude, "
                  "rate of change, etc. and their location abd frequency."
                  "Also note if certain features are consistently absent across clusters.\n"
                  "Keep the summary concise in 200 words.\n"
                  "### Each Cluster Descriptions:\n")

        for i, desc in enumerate(descs):
            if isinstance(desc, str) and desc.strip():
                prompt += f"{desc.strip()}\n"

        prompt += "\n### Common Patterns Summary:\n"

        common_summary = gpt_chat(prompt)

        out_file = os.path.join(output_dir, f"{dataset_name}_{label}_initial_common.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(common_summary, f, ensure_ascii=False, indent=2)


def gpt_chat(content, max_retries=3):
    """发送聊天请求（支持本地图片的Base64编码）"""

    # print("p", conversation)
    retry_count = 0
    while retry_count < max_retries:
        try:
            if isinstance(content, list):
                user_message = {"role": "user", "content": content}
            else:
                user_message = {"role": "user", "content": content}

            response = client.chat.completions.create(
                model="deepseek-r1",
                temperature=0.2,
                messages=[user_message],
                stream=False
            )
            return response.choices[0].message.content

        except Exception as e:
            error_msg = str(e)[:250]
            print(f"API请求失败 (尝试 {retry_count + 1}/{max_retries}): {error_msg}")
            retry_count += 1
            if retry_count < max_retries:
                time.sleep(5)
    print("已达到最大重试次数，请求失败。")
    return None


def common(train_dataset,
           selected_positions,
           output_dir):
    class_to_idxs = get_class_to_indices(train_dataset)
    sorted_classes = sorted(class_to_idxs.keys())

    for class_label in sorted_classes:
        print(f"\n🔍 Processing Class {class_label}")
        file_path = os.path.join(output_dir, f"{dataset_name}_{class_label}_initial.json")
        if not os.path.exists(file_path):
            print(f"⚠️ Warning: {file_path} not found. Skipping class {class_label}.")
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            descs = json.load(f)

        own_images = []
        for pos in selected_positions[class_label]:
            img_path = os.path.join(f"plots/{dataset_name}_train/class{class_label}/{pos[0]}.png")

            own_images.append(img_path)

            prompt = ("### Task Description:\n"
                      "You are an expert in pattern recognition and time series analysis.\n"
                      f"Below are the per-cluster descriptions for class {class_label} in the '{dataset_name}' "
                      f"dataset.\n {domain} \n"
                      "Your task is to identify **common patterns or shared characteristics** across different "
                      "clusters.\n"
                      "Focus on similarity in cycle, shape, trend, spikes, oscillations, amplitude, "
                      "rate of change, etc. and their location abd frequency."
                      "Also note if certain features are consistently absent across clusters.\n"
                      "Keep the summary concise in 200 words.\n"
                      "You will be provided one image from each cluster to help you recognize the common pattern.\n"
                      "### Each Cluster Descriptions:\n")

            for i, desc in enumerate(descs):
                if isinstance(desc, str) and desc.strip():
                    prompt += f"{desc.strip()}\n"

            content1 = [{"type": "text", "text": prompt}]

            for p in own_images:
                content1.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image_to_base64(p)}"}})

            resp1 = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": content1}],
                temperature=0.2
            )
            self_desc = resp1.choices[0].message.content.strip()

            out_file = os.path.join(output_dir, f"{dataset_name}_{class_label}_initial_common.json")
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(self_desc, f, ensure_ascii=False, indent=2)


def common2(class_label, selected_positions, output_dir):
    print(f"\n🔍 Processing Class {class_label}")
    file_path = os.path.join(output_dir, f"{dataset_name}_{class_label}_initial.json")
    with open(file_path, "r", encoding="utf-8") as f:
        descs = json.load(f)

    own_images = []
    for pos in selected_positions[class_label]:
        img_path = os.path.join(f"plots/{dataset_name}_train/class{class_label}/{pos[0]}.png")

        own_images.append(img_path)

        prompt = ("### Task Description:\n"
                  "You are an expert in pattern recognition and time series analysis.\n"
                  f"Below are the per-cluster descriptions for class {class_label} in the '{dataset_name}' "
                  f"dataset.\n {domain} \n"
                  "Your task is to identify **common patterns or shared characteristics** across different "
                  "clusters.\n"
                  "Focus on similarity in cycle, shape, trend, spikes, oscillations, amplitude, "
                  "rate of change, etc. and their location abd frequency."
                  "Also note if certain features are consistently absent across clusters.\n"
                  "Keep the summary concise in 200 words.\n"
                  "You will be provided one image from each cluster to help you recognize the common pattern.\n"
                  "### Each Cluster Descriptions:\n")

        for i, desc in enumerate(descs):
            if isinstance(desc, str) and desc.strip():
                prompt += f"{desc.strip()}\n"

        content1 = [{"type": "text", "text": prompt}]

        for p in own_images:
            content1.append(
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image_to_base64(p)}"}})

        resp1 = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": content1}],
            temperature=0.2
        )
        self_desc = resp1.choices[0].message.content.strip()

        out_file = os.path.join(output_dir, f"{dataset_name}_{class_label}_initial_common.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(self_desc, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    k = 6
    cluster = 1
    data_splits = load_data(dataset_name, n_clusters=cluster, top_k=k)
    train_dataset = TimeSeriesDataset(*data_splits['train'])
    selected_positions = data_splits['representative_indices']
    print(selected_positions)
    descriptions = analyze_classes_with_gpt4v(
        train_dataset=train_dataset,
        selected_positions=selected_positions,  # 用于自身描述
        top_k=k,
        output_dir="log")

    # summarize_common_patterns_across_classes(
    #     output_dir="log",
    #     train_dataset=train_dataset
    # )

    common(train_dataset=train_dataset,
           selected_positions=selected_positions,
           output_dir="log")
