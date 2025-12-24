import base64
import os
import re

import openai
import pandas as pd
import torch
from openai import OpenAI
import json
import time
import numpy as np
from dataloader_v2 import load_data, TimeSeriesDataset

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from round1_v3 import generate_class_description, common2, encode_image_to_base64

gpt_model = "gpt-4o"

OPENAI_API_KEY = "sk-1YNR0jY6LEWEXSMS7abadhNcpHsblvlxDhyZ62GRY16ZTYVV"

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://api.chatanywhere.tech/v1"
)
categories = 3

subset_description = (
    "This dataset were taken from data recorded as part of government sponsored study called Powering the "
    "Nation. The intention was to collect behavioural data about how consumers use electricity within the home "
    "to help reduce the UK's carbon footprint. The data contains readings from 251 households, sampled in "
    "two-minute intervals over a month. Each series is length 720 (24 hours of readings taken every 2 minutes). "
    "Classes are Desktop(Class 0) and Laptop(Class 1)")
# subset_description = (
#     "BME (Begin-Middle-End) is a synthetic univariate data set with three classes: one class is characterized "
#     "by a small positive bell arising at the initial period (Begin), one does not have any bell (Middle), "
#     "one has a positive bell arising at the final period (End)."
#     "All series are constituted by a central plate. The central plates may be positive or negative. The "
#     "discriminant is the presence or absence of a positive peak, or at the beginning of series or at the end."
#     "Class 0: Begin, Class 1: Middle, Class 2: End")
dataset_name = "BME"


def gpt_chat_vision(text_prompt, image_b64, model="gpt-4o"):
    """
    调用 GPT-4o Vision，输入文本 + 单张图像
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}"
                        }
                    }
                ]
            }
        ],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def gpt_chat(content, conversation):
    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            response = client.chat.completions.create(
                model=gpt_model,
                temperature=0.2,
                messages=conversation + [{"role": "user", "content": content}],
                stream=False
            )
            return response.choices[0].message.content

        except Exception as e:
            print(f"API请求失败 (尝试 {retry_count + 1}/{max_retries}): {e}")
            retry_count += 1
            if retry_count < max_retries:
                time.sleep(5)

    print("已达到最大重试次数，请求失败。")
    return None


def test(test_num, test_class):
    """
    从指定数据集的测试集中，随机选取 test_num 个属于 test_class 的样本，用于测试。

    Args:
        data_name (str): 数据集名称，如 "computers"
        initial_prompt (str): 初始提示词
        test_num (int): 要测试的样本数量
        test_class (int): 目标类别标签（0-indexed）
    """
    # 1. 加载数据（自动处理 train/test/all）
    data_splits = load_data(dataset_name)
    data_test_tensor, labels_test_tensor = data_splits['train']

    # 2. 找出测试集中属于 test_class 的样本索引
    indices = (labels_test_tensor == test_class).nonzero(as_tuple=True)[0]

    if len(indices) == 0:
        raise ValueError(f"No samples found in test set for class {test_class} in dataset '{dataset_name}'")

    if test_num > len(indices):
        print(f"⚠️ Requested {test_num} samples, but only {len(indices)} available for class {test_class}. Using all.")
        test_num = len(indices)

    # 3. 随机打乱并选取 test_num 个
    selected_indices = indices[torch.randperm(len(indices))[:test_num]]

    # 4. 调用测试循环（注意：传入的是整个 test data tensor + 选中的索引）
    test_cycle(test_num, selected_indices, test_class)


def plot_series_to_base64(idx, test_class):
    with open(f"plots/{dataset_name}_train_nolabel/class{test_class}/{idx}.png", "rb") as f:
        b64 = base64.b64encode(f.read()).decode('utf-8')

    return b64


def test_cycle(test_num, selected_indices, test_class, cycle=2):
    for i in range(test_num):
        correct = 0
        while correct != 2 * cycle:
            sample = plot_series_to_base64(selected_indices[i], test_class)
            initial_prompt = load_prompt()
            test_prompt = ("### Task Description:\n"
                           f"You are an expert in time series analysis. {subset_description}.\n I am trying to "
                           f"write a prompt used to generate a"
                           "description for the data, where the generated description can enhance the zero-shot "
                           "classification of the data."
                           "There are 2 descriptions to describe the data, each class has common features and "
                           "specific features:\n "
                           f"**** Text 0 ****: {initial_prompt[0]}\n"
                           f"**** Text 1 ****: {initial_prompt[1]}\n"
                           "Which description better describes the following image? Provide your analysis and the "
                           "choice you believe is better.\n"
                           "### Answer in json Format:\n"
                           "{"
                           "    \"analyze\":[Your analysis],\n"
                           "    \"answer\":[text 0 / text 1]\n"
                           "}"
                           )
            # print("class 0", initial_prompt[0])
            # print("class 1", initial_prompt[1])
            input_prompt = test_prompt

            answer = gpt_chat_vision(input_prompt, sample)

            error = answer.split('answer')[0]
            choose_content = answer.split('answer')[1]

            numbers = re.findall(r'\d+', choose_content)

            last_number = int(numbers[0])

            if last_number == test_class:
                print(f"sample {i},index {selected_indices[i]},choose {last_number},gt {test_class}")
                correct = 2 * cycle
            else:
                print(f"sample {i},index {selected_indices[i]},choose {last_number},gt {test_class}")
                true_prompt = initial_prompt[test_class]
                false_prompt = initial_prompt[last_number]
                feature, analysis, corret_type, error_type, top3_indices = reflect(sample, selected_indices[i],
                                                                                   true_prompt,
                                                                                   false_prompt, error,
                                                                                   test_class, last_number)

                if correct < cycle:
                    modify(
                        feature=feature,
                        analysis=analysis,
                        true_label=test_class,
                        misclassified_image_b64=sample,
                        type=corret_type,
                        index=selected_indices[i]
                    )
                    correct += 1
                    print(
                        f"Updated class {test_class} (correct class) based on sample {i}, model chose class {last_number}")
                else:
                    modify_wrong(
                        feature=feature,
                        analysis=analysis,
                        true_label=test_class,
                        wrong_label=last_number,
                        misclassified_image_b64=sample,
                        type=error_type,
                        index=top3_indices[0]
                    )
                    correct += 1
                    print(
                        f"Updated class {last_number} (wrong class) based on sample {i}, model chose class {last_number}")


def reflect(mis_img, current_data_index, true_prompt, false_prompt, error_string, true_label, wrong_label):
    # 加载训练数据
    file_path = f'data/{dataset_name}/{dataset_name}_TRAIN.txt'
    df = pd.read_csv(file_path, header=None, sep='\s+')
    labels = df.iloc[:, 0].values  # 第一列为标签
    data = df.iloc[:, 1:].values  # 其余为时间序列
    # 转为 tensor
    labels_tensor = torch.tensor(labels, dtype=torch.long) - 1  # 假设标签从1开始
    data_tensor = torch.tensor(data, dtype=torch.float32)
    # 获取当前样本的时间序列（注意：current_data_index 是原始数据中的索引）
    current_series = data_tensor[current_data_index].cpu().numpy()

    wrong_class_indices = (labels_tensor == wrong_label).nonzero(as_tuple=True)[0]
    # 计算 DTW 距离并排序
    distances = []
    for idx in wrong_class_indices:
        series = data_tensor[idx].cpu().numpy()

        def scalar_euclidean(u, v):
            return abs(u - v)

        dist, _ = fastdtw(current_series.flatten(), series.flatten(), dist=scalar_euclidean)
        distances.append((dist, idx))

    # 取距离最近的3个
    distances.sort(key=lambda x: x[0])
    top3_indices = [idx for _, idx in distances[:3]]
    print(top3_indices)

    neighbor_imgs = []
    img1_b64 = plot_series_to_base64(top3_indices[0], wrong_label)
    neighbor_imgs.append(img1_b64)
    img2_b64 = plot_series_to_base64(top3_indices[1], wrong_label)
    neighbor_imgs.append(img2_b64)
    img3_b64 = plot_series_to_base64(top3_indices[2], wrong_label)
    neighbor_imgs.append(img3_b64)
    feature, analysis, corret_type, error_type = analyze_misclassification(
        misclassified_image_b64=mis_img,
        nearest_same_class_images_b64=neighbor_imgs,
        correct_description=true_prompt,
        wrong_description=false_prompt,
        true_label=true_label,
        wrong_label=wrong_label
    )
    return feature, analysis, corret_type, error_type, top3_indices

    # modify_prompt = f'''My current data is "{current_data}"\n 3 sample with same label is {sample1}\n{sample2}\n{
    # sample3}\n 3 sample with wrong label is {sample4}\n{sample5}\n{sample6}\n My current prompt is: {
    # current_prompt}\n But this prompt descriptions that are too simple, similar and vague, making it difficult to
    # distinguish which description correctly matches the class and leading to the wrong description being chosen for
    # the following examples {error_string} Give a reasons why the prompt could have gotten these examples
    # wrong.Modify the Class {true_label} prompt but do not change its structure.\n### Answer Format:


def analyze_misclassification(
        misclassified_image_b64,
        nearest_same_class_images_b64,  # list of 3 base64 strings
        correct_description,
        wrong_description,
        true_label,
        wrong_label
):
    # 构建多模态消息内容（兼容 OpenAI API 格式）
    content = [
        {
            "type": "text",
            "text": (
                f"You are an expert in time series analysis and prompt engineering.\n{subset_description}\n"
                "The following time series image was **misclassified**:\n"
                f"- True class: {true_label}\n"
                "**Description:**\n"
                f"{correct_description}\n\n"
                f"- Predicted class: {wrong_label}\n\n"
                "**Description:**\n"
                f"{wrong_description}\n\n"
                "The 2 description is too simple, similar and vague, making it "
                "difficult to distinguish which description correctly matches the image and leading to the wrong"
                "description being chosen for the following examples.\n"
                "### Task Description:\n"
                "1. Compare the visual patterns between the classes. "
                "2. Explain why the wrong description might appear more "
                "plausible for the misclassified sample. Based the wrong description, select the type that is most "
                f"similar to the image in the error category (class {wrong_label})\n"
                f"3. Determine which type's majority of features in correct category (class {true_label}) this image"
                f" is most consistent with. If none of type match, answer \"none\""
                f"4. Identify what key characteristics/differences are missing or underemphasized in the "
                "correct description.\n"
                "### Answer in json Format:\n"
                "{"
                "   \"explanation\":,\n"
                "   \"error type\":[most similar to the image in the error category],\n"
                "   \"correct type\":[a number or none],\n"
                "   \"key characteristics\":[]\n"
                "}"
            )
        },
        {
            "type": "text",
            "text": f"Below are visualizations:\nMisclassified Sample (True Class is {true_label} but Predicted Class is {wrong_label})\n):"
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{misclassified_image_b64}"}
        },
        {
            "type": "text",
            "text": f"Three most similar samples from misclassified class (class {wrong_label}):"
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{nearest_same_class_images_b64[0]}"}
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{nearest_same_class_images_b64[1]}"}
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{nearest_same_class_images_b64[2]}"}
        }
    ]

    # 调用 OpenAI 兼容的多模态 API
    messages = [{"role": "user", "content": content}]

    response = client.chat.completions.create(
        model="gpt-4o",  # 或 gpt-4o-mini, claude-3-5-sonnet 等
        messages=messages,
        temperature=0.2
    )

    answer = response.choices[0].message.content

    # 解析输出
    try:
        analysis = answer.split("explanation")[1].split("error type")[0].strip()
        et = answer.split("error type")[1].split("correct type")[0].strip().lower()
        print("ERROR TYPE", et)
        numbers = re.findall(r'\d+', et)
        error_type = numbers[0]
        ct = answer.split("correct type")[1].split('key characteristics')[0].strip().lower()
        print("CORRECT TYPE", ct)
        if not ct or "none" in ct:
            correct_type = "none"
        else:
            numbers = re.findall(r'\d+', ct)
            correct_type = numbers[0] if numbers else "none"
        feature = answer.split("key characteristics")[1].strip()
    except Exception as e:
        print("⚠️ Parsing failed:", e)
        print("Raw answer:", answer)
        # 退化：返回原描述
        return correct_description, answer, "none", 0

    # print("✅ Analysis:", analysis)
    # print("✨ Feature:", feature)

    return feature, analysis, correct_type, error_type


def modify(feature, analysis, true_label, misclassified_image_b64, type, index):
    """
    基于错分图像、遗漏特征和错误分析，改进该类别的文本 prompt。

    Args:
        current_prompt (str): 当前用于 true_label 的描述
        feature (str): 被当前 prompt 忽略的关键时间序列特征
        analysis (str): 模型分类错误的原因
        true_label (int): 正确类别
        misclassified_image_b64 (str): 被错分样本的 base64 图像

    Returns:
        str: 改进后的 prompt
    """
    file_path2 = f"log/{dataset_name}_{true_label}_initial.json"
    with open(file_path2, 'r', encoding='utf-8') as file2:
        contents2 = list(json.load(file2))
    if type == "none":
        # 加载训练数据
        file_path = f'data/{dataset_name}/{dataset_name}_TRAIN.txt'
        df = pd.read_csv(file_path, header=None, sep='\s+')
        labels = df.iloc[:, 0].values  # 第一列为标签
        data = df.iloc[:, 1:].values  # 其余为时间序列
        # 转为 tensor
        labels_tensor = torch.tensor(labels, dtype=torch.long) - 1  # 假设标签从1开始
        data_tensor = torch.tensor(data, dtype=torch.float32)
        # 获取当前样本的时间序列（注意：current_data_index 是原始数据中的索引）
        current_series = data_tensor[index].cpu().numpy()

        true_class_indices = (labels_tensor == true_label).nonzero(as_tuple=True)[0]
        # 计算 DTW 距离并排序
        distances = []
        for idx in true_class_indices:
            series = data_tensor[idx].cpu().numpy()

            def scalar_euclidean(u, v):
                return abs(u - v)

            dist, _ = fastdtw(current_series.flatten(), series.flatten(), dist=scalar_euclidean)
            distances.append((dist, idx))

        # 取距离最近的3个
        distances.sort(key=lambda x: x[0])
        top3_indices = [idx for _, idx in distances[1:5]]
        print(top3_indices)

        own_images = []
        for pos in top3_indices:
            img_path = os.path.join(f"plots/{dataset_name}_train/class{true_label}/{pos}.png")
            if os.path.exists(img_path):
                own_images.append(img_path)
            else:
                print(f"  ⚠️ Image not found: {img_path}")

        desc = generate_class_description(
            class_label=true_label,
            own_image_paths=own_images,
            top_k=5
        )
        contents2.append(desc)
        print("NEW:", desc)

    else:
        # 构建多模态输入内容（兼容 OpenAI API）
        type = int(type)

        current_prompt = contents2[type]
        print("CURRENT PROMPT:", current_prompt)
        content = [
            {
                "type": "text",
                "text": (
                    f"You are an expert in time series analysis and prompt design for zero-shot classification."
                    f"\n{subset_description}\n"
                    f"### Task Description\n"
                    f"The image belongs to class {true_label}, but the model makes a wrong prediction since insufficient description."
                    f"Improve the textual description of type {type} in class {true_label} so that it correctly "
                    f"describes the time series shown in the image below.\n"
                    f"### Current Description:\n"
                    f'"{current_prompt}"\n\n'
                    f"### The reasons for the model's incorrect classification :\n"
                    f"{analysis}\n\n"
                    f"### Missing Key Features :\n"
                    f"{feature}\n\n"
                    f"### Requirements:\n"
                    f"- Add or modify statements on the textual description.\n"
                    f"- Incorporate the missing key features above.\n"
                    f"- DO NOT change the structure of the origin description (bullet-point list).\n"
                    f"- Only output the modified description and do not output any other information"
                )
            },
            {
                "type": "text",
                "text": f"Misclassified Sample (True Class: {true_label}):"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{misclassified_image_b64}"
                }
            }
        ]

        # 调用多模态大模型（如 GPT-4o）
        messages = [{"role": "user", "content": content}]

        response = client.chat.completions.create(
            model="gpt-4o",  # 或 gpt-4o-mini
            messages=messages,
            temperature=0.2
        )
        answer = response.choices[0].message.content

        # 解析输出
        try:
            improved = answer.strip()
        except (IndexError, AttributeError):
            print("Raw response:", answer)
            return current_prompt

        print("✨ Improved prompt for class", true_label, ":", improved)
        # 替换列表中的第 type_value 项
        contents2[type] = improved

    # 将更新后的列表写回文件
    with open(file_path2, 'w', encoding='utf-8') as file2:
        json.dump(contents2, file2, ensure_ascii=False, indent=2)

    k = 6
    cluster = 6
    data_splits = load_data(dataset_name, n_clusters=cluster, top_k=k)
    selected_positions = data_splits['representative_indices']
    common2(true_label, selected_positions, "log")


def modify_wrong(feature, analysis, true_label, wrong_label, misclassified_image_b64, type, index):
    img_path = os.path.join(f"plots/{dataset_name}_train/class{wrong_label}/{index}.png")

    file_path2 = f"log/{dataset_name}_{wrong_label}_initial.json"
    with open(file_path2, 'r', encoding='utf-8') as file2:
        contents2 = list(json.load(file2))

    # 构建多模态输入内容（兼容 OpenAI API）
    type = int(type)

    current_prompt = contents2[type]
    print("CURRENT PROMPT:", current_prompt)
    content = [
        {
            "type": "text",
            "text": (
                f"You are an expert in time series analysis and prompt design for zero-shot classification."
                f"\n{subset_description}\n"
                f"### Task Description\n"
                f"The image belongs to class {true_label}, but the model chooses type {type} in class {wrong_label}\n."
                f"You need to improve the textual description of type {type} in class {wrong_label} so that it "
                f"DO NOT match the misclassified time series\n"
                f"I will provide 1 image belongs to type {type} in class {wrong_label} but similar to the "
                f"misclassified image to help you distinguish.\n"
                f"### Current Description of type {type} in class {wrong_label}:\n"
                f'"{current_prompt}"\n\n'
                f"### The reasons for the model's incorrect classification :\n"
                f"{analysis}\n\n"
                f"### Requirements:\n"
                f"- Add or modify statements on the textual description.\n"
                f"- Incorporate the missing key features above.\n"
                f"- DO NOT change the structure of the origin description (bullet-point list).\n"
                f"- Only output the modified description and do not output any other information"
            )
        },
        {
            "type": "text",
            "text": f"Misclassified Sample (True Class: {true_label}):"
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{misclassified_image_b64}"
            }
        },
        {
            "type": "text",
            "text": f"Similar Sample (True Class: {wrong_label}):"
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{encode_image_to_base64(img_path)}"
            }
        }
    ]

    # 调用多模态大模型（如 GPT-4o）
    messages = [{"role": "user", "content": content}]

    response = client.chat.completions.create(
        model="gpt-4o",  # 或 gpt-4o-mini
        messages=messages,
        temperature=0.2
    )
    answer = response.choices[0].message.content

    # 解析输出
    try:
        improved = answer.strip()
    except (IndexError, AttributeError):
        print("Raw response:", answer)
        return current_prompt

    print("✨ Improved prompt for class", wrong_label, ":", improved)
    # 替换列表中的第 type_value 项
    contents2[type] = improved

    # 将更新后的列表写回文件
    with open(file_path2, 'w', encoding='utf-8') as file2:
        json.dump(contents2, file2, ensure_ascii=False, indent=2)

    k = 6
    cluster = 6
    data_splits = load_data(dataset_name, n_clusters=cluster, top_k=k)
    selected_positions = data_splits['representative_indices']
    common2(wrong_label, selected_positions, "log")


def load_prompt():
    class_feature = []

    for n in range(categories):
        feature = f"\n** The common feature of class {n}: **\n"
        file_path = f"log/{dataset_name}_{n}_initial_common.json"

        with open(file_path, 'r', encoding='utf-8') as file:
            contents = json.load(file)

        feature += contents
        file_path2 = f"log/{dataset_name}_{n}_initial.json"
        with open(file_path2, 'r', encoding='utf-8') as file2:
            contents2 = list(json.load(file2))
            contents3 = "\n".join(
                f"The type {i} time series of class {n} is: {item}"
                for i, item in enumerate(contents2)
            )

        feature += contents3
        class_feature.append(feature)
    return class_feature


if __name__ == "__main__":
    test(10, 1)
