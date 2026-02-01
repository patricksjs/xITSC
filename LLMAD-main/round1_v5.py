import os
import random
import time
from pathlib import Path

import torch
from collections import defaultdict
from openai import OpenAI
import base64
import json
from dataloader_v2 import load_data, TimeSeriesDataset, cluster_per_class, random_choice

# 初始化 OpenAI 客户端
OPENAI_API_KEY = "sk-J3azhkbBoUT2YQC8Sl2KLsqBnKC5LcamDvTxWco3ZglWdcgJ"
client = OpenAI(api_key=OPENAI_API_KEY,
                base_url="https://api.chatanywhere.tech/v1",
                # base_url="https://api.chatanywhere.org/v1"
                )
dataset_name = "Trace"
category = 2
# domain = ("This dataset were taken from data recorded as part of government sponsored study called Powering the "
#           "Nation. The intention was to collect behavioural data about how consumers use electricity within the home "
#           "to help reduce the UK's carbon footprint. The data contains readings from 251 households, sampled in "
#           "two-minute intervals over a month. Each series is length 720 (24 hours of readings taken every 2 minutes). "
#           "Classes are Desktop(Class 0) and Laptop(Class 1)")


# domain = ("BME (Begin-Middle-End) is a synthetic univariate data set with three classes: "
#           # "one class is characterized "
#           # "by a small positive bell arising at the initial period (Begin), one does not have any bell (Middle), "
#           # "one has a positive bell arising at the final period (End)."
#           "All series are constituted by a central plate. The central plates may be positive or negative. The "
#           "discriminant is the presence or absence of a positive peak, or at the beginning of series or at the end."
#           "Class 0: Begin, Class 1: Middle, Class 2: End")

# domain = ("The arrowhead data consists of outlines of the images of arrowheads. The shapes of the projectile points "
#           "are converted into a time series using the angle-based method. The classification of projectile points is "
#           "an important topic in anthropology. The classes are based on shape distinctions such as the presence and "
#           "location of a notch in the arrow. The three classes are called Avonlea(class 0), Clovis(class 1) and Mix("
#           "class 2)")

# domain = ("The data was collected using a tri-axial accelerometer on the dominant wrist "
#           "whilst conducting 4 different activities: SEIZURE MIMICKING(class 0) with seating after the mimicked "
#           "seizure, WALKING(class 1) with different paces and"
#           "gestures, RUNNING(class 2) with running a 40 meters long corridor, SAWING(class 3) with a saw and "
#           "during 30 seconds."
#           "The sampling frequency was 16 Hz. The activities lasted about 13 seconds")
# domain = ("These dataset were taken from data recorded as part of government sponsored study called Powering the "
#           "Nation. The intention was to collect behavioural data about how consumers use electricity within the home "
#           "to help reduce the UK's carbon footprint. The data contains readings from 251 households, sampled in "
#           "two-minute intervals over a month. Each series is length 720 (24 hours of readings taken every 2 minutes). "
#           "There are 7 classes.")
domain = ("It is a synthetic dataset designed to simulate instrumentation failures in a nuclear power plant, "
          "with 4 classes")
# domain = ("This is a dataset with phase-aligned starlight curves of length 1,024, whose class has been determined by "
#           "an expert.There are 3 classes.")
# domain = ("The dataset has 3 classes. Data from each class is standard normal noise plus an offset term which differs "
#           "for each class.")

# domain = ("The two classes are a normal heartbeat(class 0) and a Myocardial Infarction(class 1). The electrocardiogram "
#           "of a normal heartbeat and that of a myocardial infarction are quite different, mainly reflected in the "
#           "following core features:\n "
#           "Normal heartbeat: The ST segment is straight and coincides with the baseline. T waves are usually upright "
#           "and symmetrical.\n"
#           "Myocardial infarction: The ST segment will show obvious abnormalities. The most common two situations are: "
#           "one is the arched back upward type elevation, and the other is significant depression. The T wave becomes "
#           "abnormally sharp (acute phase) or symmetrically inverted (necrotic phase).")



def get_class_to_indices(dataset):
    class_to_idxs = defaultdict(list)
    for idx in range(len(dataset)):
        _, label, _ = dataset[idx]
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
        f"You are an expert in time series analysis for the {dataset_name} dataset. "
        f"### Dataset Information:\n{domain}\n"
        f"### Previously Identified Features for Class {class_label}:\n{existing_text}\n"
        "### Current Task:\n"
        "Analyze the provided time series images together and identify NEW COMMON visual patterns "
        "or characteristics among them that are NOT already listed above.\n"
        "You need to take into account both normal and myocardial infarction ECG.\n"
        # "Focus on feature such as trend, cycle, shape, symmetry, stationarity, "
        # "amplitude, rate of change, spikes, etc. and their location and frequency of occurrence."
        "You can compare across the images to find common patterns below.\n"
        "-- trend: [e.g. Overall trend direction: upward trend, downward trend, no trend; Trend stability: stable "
        "trend, gradual trend, sudden change trend, oscillating trend]\n"
        "-- cycle: [e.g. frequency and stability; period value]\n"
        "-- shape: [e.g. Overall contour: smooth curves, serrated curves, stepped curves, pulsed curves; Local "
        "features: number and location of inflection points]\n"
        "-- symmetry:\n"
        "-- stationarity: [e.g. irregular oscillation, baseline feature, plateau feature, rate of change]\n"
        "-- amplitude: [e.g. peak-trough difference; stable or oscillating]\n"
        "-- spike and valley:[e.g. sharp or wide; small hills or high peaks; peak and valley values; duration; "
        "frequency; the time of occurrence.]\n"
        "You can summarize the global patterns or segment the time series to analyze the local patterns.\n."
        "### Answer Format:\n"
        "Return a star-point list (*) with each point describing ONE distinct NEW feature.\n"
        "Keep each point under 50 words. Do NOT repeat any existing feature, even in different words.\n"
        "If no new features are found, return exactly: 'No new'\n"
        "Add less than 3 features. \n"
        # "Avoid vague descriptions.\n "
        "Only output the star points or the 'No new' message.\n"
        "### Example:\n"
        "The early P waves were exceptionally sharp and high, with values reaching 3 to 4."
    )

    content = [{"type": "text", "text": prompt}]
    for img_path in image_path:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encode_image_to_base64(img_path)}"}
        })

    resp = client.chat.completions.create(
        model=gpt_model,
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


def get_feature_list(label_tensor,
                     selected_indices,
                     output_dir,
                     round,
                     images_per_round):
    os.makedirs(output_dir, exist_ok=True)
    sorted_classes = torch.unique(label_tensor)
    print("Classes:", sorted_classes)

    all_class_features = {}

    # 遍历每个类别
    for class_idx, class_label in enumerate(sorted_classes):
        print(f"\n🔍 Processing Class {class_label}")
        img_indices = selected_indices[class_idx]  # list of indices for this class
        accumulated_features = []  # will grow incrementally
        for cluster in img_indices:
            for round_i in range(round):
                print(f"  🔄 Round {round_i + 1}")

                # Randomly sample images_per_round images (with replacement if needed, but better without)
                if len(cluster) < images_per_round:
                    selected = cluster  # use all if not enough
                else:
                    selected = random.sample(cluster, images_per_round)
                    print(selected)
                image_paths = [
                    os.path.join(f"plots/{dataset_name}_train_nolabel/class{class_label}/{pos}.png")
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

                    if len(cluster) < images_per_round:
                        break
                except Exception as e:
                    print(f"    ❌ Error in round {round_i + 1}: {e}")
                    continue

            # Save final features for this class
        # all_class_features[class_label] = accumulated_features
        # out_file = os.path.join(output_dir, f"{dataset_name}_{class_label}_features.json")
        # with open(out_file, "w", encoding="utf-8") as f:
        #     json.dump(accumulated_features, f, ensure_ascii=False, indent=2)
        #
        # print(f"  ✅ Class {class_label}: total {len(accumulated_features)} unique features saved.")

        # 创建新的字典，其中每个特征作为键，"common feature" 作为值
        features_with_common_label = {feature: "common feature" for feature in accumulated_features}

        out_file = os.path.join(output_dir, f"{dataset_name}_{class_label}_{gpt_model}_features.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(features_with_common_label, f, ensure_ascii=False, indent=2)

        print(
            f"  ✅ Class {class_label}: total {len(accumulated_features)} unique features saved.")
    return all_class_features


def summarize_common_patterns_across_classes(label_tensor, output_dir):
    sorted_classes = torch.unique(label_tensor)
    print("Classes:", sorted_classes)

    # 第一步：收集所有类别的 common features（用于后续排除）
    all_common_features = set()
    class_features_data = {}

    for label in sorted_classes:
        file_path = os.path.join(output_dir, f"{dataset_name}/feature_list/{dataset_name}_{label}_features.json")
        if not os.path.exists(file_path):
            print(f"⚠️ Warning: {file_path} not found. Skipping class {label}.")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            features_dict = json.load(f)

        # 提取当前类的 common features
        common_in_class = {key for key, val in features_dict.items() if val == "common feature"}
        all_common_features.update(common_in_class)

        # 保存原始数据供后续使用
        class_features_data[label.item()] = common_in_class

    # 第二步：为每个类别生成不含公共特征的关键特征描述
    final_summaries = []

    for label in sorted_classes:
        prompt = (
            f"You are an expert in time series analysis for the {dataset_name} dataset. {domain}\n"
            "Each category has several characteristics.\n"
            f"### class 0 feature:\n {class_features_data[0]}\n"
            f"### class 1 feature:\n {class_features_data[1]}\n"
            "### Current Task:\n"
            "Conduct a horizontal comparison between different label features.\n"
            f"Analyze and screen out the key features of class {label} as classification criteria."
            "Try to avoid including common features in the classification criteria, but retain as many features as "
            "possible."
            "DO NOT modify the original features."
            "### Answer Format:\n"
            f"Return a star-point list (*) with each point describing ONE KEY feature of the class {label}.\n"
        )

        common_summary = gpt_chat(prompt)

        # 清理输出格式，确保是纯文本
        lines = [line.strip() for line in common_summary.split('\n') if line.strip().startswith('*')]
        new_features = [line[1:].strip() for line in lines]

        # 读取原始 JSON 文件以获取 "part feature"
        original_file_path = os.path.join(output_dir,
                                          f"{dataset_name}/feature_list/{dataset_name}_{label}_features.json")
        with open(original_file_path, "r", encoding="utf-8") as f:
            original_features_dict = json.load(f)

        # 提取所有 val == "part feature" 的键
        part_features = {k: v for k, v in original_features_dict.items() if v == "partial-sample feature"}

        # 构建新字典：
        # - new_features 中的项：值设为 "common feature"
        # - part_features 保留原样
        updated_features = {}

        # 先加入 new_features（作为关键特征，按要求标记为 "common feature"）
        for feat in new_features:
            updated_features[feat] = "common feature"

        # 再加入原有的 "part feature"
        updated_features.update(part_features)

        # 保存到新 JSON 文件
        new_json_path = os.path.join(output_dir,
                                     f"{dataset_name}/feature_list/{dataset_name}_{label}_key_features.json")
        with open(new_json_path, "w", encoding="utf-8") as f:
            json.dump(updated_features, f, ensure_ascii=False, indent=2)

        print(f"✅ Saved key features for class {label} to {new_json_path}")


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
    # prompt = (
    #     f"You are an expert in time series analysis for the {dataset_name} dataset. {domain}\n\n"
    #     "### Task:\n"
    #     f"Check if the following feature EXISTS in the provided time series image:\n{feature}\n"
    #     "Respond with 'Yes' if the feature is present, or 'No' if not.\n"
    #     "Only output 'Yes' or 'No'."
    # )
    prompt = (
        f"You are an expert in time series analysis for the {dataset_name} dataset. {domain}\n\n"
        "### Task:\n"
        f"Check if the following feature EXISTS in the provided time series image:\n{feature}\n"
        "Respond with 'Yes' if the feature is present, or 'Part' if it partially exists, or 'No' if not.\n"
        "Only output 'Yes' or 'Part' or 'No'."
    )

    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image_to_base64(image_path)}"}}
    ]

    resp = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[{"role": "user", "content": content}],
        temperature=0.2
    )
    answer = resp.choices[0].message.content.strip()
    print(answer)
    return answer


def analyze_test_samples(test_indices, output_dir):
    for class_label in range(category):
        file_path = os.path.join(output_dir, f"{dataset_name}/feature_list/{dataset_name}_{class_label}_{gpt_model}_features.json")
        temp_path = os.path.join(output_dir, f"{dataset_name}/feature_list/{dataset_name}_{class_label}_{gpt_model}_temp.json")
        with open(file_path, "r", encoding="utf-8") as f:
            features_dict = json.load(f)
            features = list(features_dict.keys())

        print(f"\n🔍 Analyzing Class {class_label}")
        stats = [0] * len(features)  # Initialize occurrence counts for this class's features

        for sample_idx in test_indices[class_label]:
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

        print(f"  ✅ Class {class_label}: {stats} ")

        threshold_low = len(test_indices[class_label]) / 3  # 1/3
        threshold_high = 2 * len(test_indices[class_label]) / 3  # 2/3

        for i, feature in enumerate(features):
            count = stats[i]
            if count >= threshold_high:
                features_dict[feature] = "common feature"
            elif count <= threshold_low:
                features_dict[feature] = "delete feature"  # both <1/3 and [1/3, 2/3)
                print(features_dict[feature])
            else:
                features_dict[feature] = "partial-sample feature"
        # Save updated dict back to file
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(features_dict, f, ensure_ascii=False, indent=2)
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(features_dict, f, ensure_ascii=False, indent=2)
        print(f"Stats summary: {stats}")


if __name__ == "__main__":
    k = 5
    cluster = 2
    r = 6
    gpt_model = "gemini-2.5-pro"  # gemini-2.5-flash grok-4-fast qwen3-235b-a22b claude-sonnet-4-5-20250929 o4-mini
    data_train_tensor, labels_train_tensor, _ = load_data(dataset_name, n_clusters=cluster, top_k=k, random=r)['train']
    folder_path = Path(f"plots/{dataset_name}_train_nolabel")
    selected_indices = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".png"):
                try:
                    idx_str = file.split('.')[0]  # 移除 .png
                    idx = int(idx_str)
                    selected_indices.append(idx)
                except ValueError:
                    print(f"Invalid filename: {file}, skipping")
    selected_indices = sorted(selected_indices)  # 排序方便调试
    selected_positions = cluster_per_class(data_tensor=data_train_tensor, labels_tensor=labels_train_tensor, selected_indice=selected_indices, n_clusters=cluster, top_k=k)

    data_splits = load_data(dataset_name, n_clusters=cluster, top_k=k, random=r)
    train_dataset = TimeSeriesDataset(*data_splits['train'])
    # selected_positions = data_splits['representative_indices']
    # print(selected_positions)
    # random_indices = data_splits['random_indices']
    random_indices = random_choice(labels_train_tensor, selected_indices, r)
    # print(random_indices)

    get_feature_list(
        label_tensor=labels_train_tensor,
        selected_indices=selected_positions,
        output_dir=f"log/{dataset_name}/feature_list",
        round=2,
        images_per_round=3
    )
    analyze_test_samples(test_indices=random_indices, output_dir="log")
    # summarize_common_patterns_across_classes(labels_train_tensor, "log")
