import os
import random
import time

import torch
from collections import defaultdict
from openai import OpenAI
import base64
import json
from dataloader_v2 import load_data, TimeSeriesDataset

# 初始化 OpenAI 客户端
OPENAI_API_KEY = "sk-jqE4GfEhpVfwQeo08T4W1jbWXdyGlBTUdoIxSsx3h2njtJ3p"
client = OpenAI(api_key=OPENAI_API_KEY, base_url="https://api.chatanywhere.tech/v1")
dataset_name = "ArrowHead"
category = 3
# domain = ("This dataset were taken from data recorded as part of government sponsored study called Powering the "
# "Nation. The intention was to collect behavioural data about how consumers use electricity within the home " "to
# help reduce the UK's carbon footprint. The data contains readings from 251 households, sampled in " "two-minute
# intervals over a month. Each series is length 720 (24 hours of readings taken every 2 minutes). " "Classes are
# Desktop(Class 0) and Laptop(Class 1)")


# domain = ("BME (Begin-Middle-End) is a synthetic univariate data set with three classes: one class is characterized "
#           "by a small positive bell arising at the initial period (Begin), one does not have any bell (Middle), "
#           "one has a positive bell arising at the final period (End)."
#           "All series are constituted by a central plate. The central plates may be positive or negative. The "
#           "discriminant is the presence or absence of a positive peak, or at the beginning of series or at the end."
#           "Class 0: Begin, Class 1: Middle, Class 2: End")

domain = ("The arrowhead data consists of outlines of the images of arrowheads. The shapes of the projectile points "
          "are converted into a time series using the angle-based method. The classification of projectile points is "
          "an important topic in anthropology. The classes are based on shape distinctions such as the presence and "
          "location of a notch in the arrow. The three classes are called Avonlea(class 0), Clovis(class 1) and Mix("
          "class 2)")


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


def generate_new_features(image_path, class_label, existing_features):
    """
    Ask GPT-4V to describe ONLY NEW features in this image that are not already in existing_features.
    """
    existing_text = "\n".join(f"• {f}" for f in existing_features) if existing_features else "None yet."

    # prompt = (
    #     f"You are an expert in time series analysis for the {dataset_name} dataset. {domain}\n"
    #     f"### Previously Identified Features for Class {class_label}:\n{existing_text}\n"
    #     "### Current Task:\n"
    #     "Analyze the provided time series image and identify ONLY NEW visual patterns or characteristics "
    #     "that are NOT already listed above.\n"
    #     "Focus on feature such as trend, cycle, shape, symmetry, skewness, kurtosis, stationarity, "
    #     "amplitude, rate of change, spikes, etc. and their location and frequency of occurrence."
    #     "You can segment the time series for analysis."
    #     "### Answer Format:\n"
    #     "Return a star-point list (*) with each point describing ONE distinct feature.\n"
    #     "Keep each point under 50 words.\n"
    #     "If the image is largely similar to a previous feature, you can make some modifications instead of"
    #     "adding new features."
    #     "Only output the star points (should include the previously identified features)."
    # )
    prompt = (
        f"You are an expert in time series analysis for the {dataset_name} dataset. {domain}\n"
        f"### Previously Identified Features for Class {class_label}:\n{existing_text}\n"
        "### Current Task:\n"
        "Analyze the THREE provided time series images together and identify ONLY NEW COMMON visual patterns "
        "or characteristics among them that are NOT already listed above.\n"
        "Focus on feature such as trend, cycle, shape, symmetry, stationarity, "
        "amplitude, rate of change, spikes, etc. and their location and frequency of occurrence."
        "You may compare across the three images to find common patterns.\n"
        "You can segment the time series for analysis."
        "### Answer Format:\n"
        "Return a star-point list (*) with each point describing ONE distinct NEW COMMON feature.\n"
        "Keep each point under 50 words. Do NOT repeat any existing feature, even in different words.\n"
        "If no new features are found, return exactly: 'No new'\n"
        # "DO NOT mention specific dataset index like(the final sample).\n"
        # "DO NOT mention the sample like (in all samples).\n"
        "Describe the feature first and then the period."
        "Only output the star points or the 'No new' message.\n"
        "Examples:\n"
        "* Repeated wave-like structures appearing frequently, particularly noticeable between the 150th and 200th "
        "time steps.\n"
        "* The amplitude of oscillations increases markedly after the initial peak."
    )

    content = [{"type": "text", "text": prompt}]
    for img_path in image_path:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encode_image_to_base64(img_path)}"}
        })

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": content}],
        temperature=0.2
    )
    text = resp.choices[0].message.content.strip()

    if text == "No new":
        return []

    # Parse bullet points
    lines = [line.strip() for line in text.split('\n') if line.strip().startswith('*')]
    new_features = [line[1:].strip() for line in lines]
    return new_features


def get_feature_list(train_dataset,
                     random_indices,
                     output_dir,
                     images_per_round=2):
    os.makedirs(output_dir, exist_ok=True)

    class_to_idxs = get_class_to_indices(train_dataset)
    sorted_classes = sorted(class_to_idxs.keys())
    print("Classes:", sorted_classes)

    all_class_features = {}

    # 遍历每个类别
    for class_idx, class_label in enumerate(sorted_classes):
        print(f"\n🔍 Processing Class {class_label}")

        img_indices = random_indices[class_idx]  # list of indices for this class
        accumulated_features = []  # will grow incrementally

        for round_i in range(len(random_indices[0])):
            print(f"  🔄 Round {round_i + 1}")

            # Randomly sample images_per_round images (with replacement if needed, but better without)
            if len(img_indices) < images_per_round:
                selected = img_indices  # use all if not enough
            else:
                selected = random.sample(img_indices, images_per_round)

            image_paths = [
                os.path.join(f"plots/{dataset_name}_train/class{class_label}/{pos}.png")
                for pos in selected
            ]

            # Check existence
            valid_paths = []
            for p in image_paths:
                if not os.path.exists(p):
                    print(f"    ⚠️ Image not found: {p}")
                else:
                    valid_paths.append(p)

            if not valid_paths:
                print("    ❌ No valid images in this round.")
                continue

            try:
                new_feats = generate_new_features(
                    image_path=valid_paths,
                    class_label=class_label,
                    existing_features=accumulated_features
                )

                if new_feats:
                    accumulated_features.extend(new_feats)
                    print(f"    ➕ Added {len(new_feats)} new feature(s):")
                    for nf in new_feats:
                        print(f"      • {nf[:70]}...")
                else:
                    print("    ➖ No new features.")

            except Exception as e:
                print(f"    ❌ Error in round {round_i + 1}: {e}")
                continue

            # Save final features for this class
        all_class_features[class_label] = accumulated_features
        out_file = os.path.join(output_dir, f"{dataset_name}_{class_label}_features.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(accumulated_features, f, ensure_ascii=False, indent=2)

        print(f"  ✅ Class {class_label}: total {len(accumulated_features)} unique features saved.")

    return all_class_features


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


def check_feature_existence(image_path, feature):
    """
    Ask GPT-4V if the specified feature exists in the given image.
    Returns True if the feature is identified, otherwise False.
    """
    prompt = (
        f"You are an expert in time series analysis for the {dataset_name} dataset. {domain}\n\n"
        "### Task:\n"
        f"Check if the following feature EXISTS in the provided time series image:\n{feature}\n"
        "Respond with 'Yes' if the feature is present, or 'No' if not.\n"
        "Only output 'Yes' or 'No'."
    )
    # prompt = (
    #     f"You are an expert in time series analysis for the {dataset_name} dataset. {domain}\n\n"
    #     "### Task:\n"
    #     f"Check if the following feature EXISTS in the provided time series image:\n{feature}\n"
    #     "Respond with 'Yes' if the feature is present, or 'Part' if it partially exists, or 'No' if not.\n"
    #     "Only output 'Yes' or 'Part' or 'No'."
    # )

    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image_to_base64(image_path)}"}}
    ]

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": content}],
        temperature=0.2
    )
    answer = resp.choices[0].message.content.strip()
    print(answer)
    return answer


def analyze_test_samples(class_label, test_indices, output_dir):
    file_path = os.path.join(output_dir, f"{dataset_name}/feature_list/{dataset_name}_{class_label}_features.json")
    with open(file_path, "r", encoding="utf-8") as f:
        features = json.load(f)

    print(f"\n🔍 Analyzing Class {class_label}")
    stats = [0] * len(features)  # Initialize occurrence counts for this class's features

    for sample_idx in test_indices.get(class_label, []):
        img_path = os.path.join(f"plots/{dataset_name}_train_nolabel/class{class_label}/{sample_idx}.png")

        if not os.path.exists(img_path):
            print(f"  ⚠️ Image not found: {img_path}")
            continue

        for i, feature in enumerate(features):
            answer = check_feature_existence(img_path, feature)
            if ('yes' == answer
                    or 'Yes' == answer
                    or 'YES' == answer
                    or 'part' == answer
                    or 'Part' == answer
                    or 'PART' == answer):
                stats[i] += 1

    print(f"  ✅ Class {class_label}: Stats saved. {stats}")

    return stats


if __name__ == "__main__":
    k = 2
    cluster = 3
    r = 8
    data_splits = load_data(dataset_name, n_clusters=cluster, top_k=k, random=r)
    train_dataset = TimeSeriesDataset(*data_splits['train'])
    selected_positions = data_splits['representative_indices']
    print(selected_positions)
    random_indices = data_splits['random_indices']
    print(random_indices)
    # descriptions = analyze_classes_with_gpt4v(
    #     train_dataset=train_dataset,
    #     selected_positions=selected_positions,  # 用于自身描述
    #     top_k=k,
    #     output_dir="log")

    get_feature_list(
        train_dataset=train_dataset,
        random_indices=random_indices,
        output_dir=f"log/{dataset_name}/feature_list"
    )
    # for i in range(category):
    #     stats = analyze_test_samples(i, selected_positions, "log")
    #     print(stats)
