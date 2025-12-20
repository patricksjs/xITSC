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
from dataloader_v2 import load_data

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

gpt_model = "gpt-4o"

OPENAI_API_KEY = "sk-VdhI38Ualpqsc1yxyYN3Is82AWrNOtvhMjgxUihamFZSAf7k"

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://api.chatanywhere.tech/v1"
)
categories = 3

# subset_description = (
#     "This dataset were taken from data recorded as part of government sponsored study called Powering the "
#     "Nation. The intention was to collect behavioural data about how consumers use electricity within the home "
#     "to help reduce the UK's carbon footprint. The data contains readings from 251 households, sampled in "
#     "two-minute intervals over a month. Each series is length 720 (24 hours of readings taken every 2 minutes). "
#     "Classes are Desktop(Class 0) and Laptop(Class 1)")
subset_description = ("BME (Begin-Middle-End) is a synthetic univariate data set with three classes: one class is characterized "
          "by a small positive bell arising at the initial period (Begin), one does not have any bell (Middle), "
          "one has a positive bell arising at the final period (End)."
          "All series are constituted by a central plate. The central plates may be positive or negative. The "
          "discriminant is the presence or absence of a positive peak, or at the beginning of series or at the end."
          "Class 0: Begin, Class 1: Middle, Class 2: End")
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


def test(initial_prompt, test_num, test_class):
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
    final_description = test_cycle(initial_prompt, test_num, selected_indices, test_class)
    print(final_description)


def plot_series_to_base64(idx, test_class):
    with open(f"plots/{dataset_name}_train_nolabel/class{test_class}/{idx}.png", "rb") as f:
        b64 = base64.b64encode(f.read()).decode('utf-8')

    return b64


def test_cycle(initial_prompt, test_num, selected_indices, test_class, max_cycles=20):
    cycle_count = 0
    while cycle_count < max_cycles:
        updated = False
        print("cycle count:", cycle_count)
        correct = 0
        for i in range(test_num):
            sample = plot_series_to_base64(selected_indices[i], test_class)

            test_prompt = f"""\n### Task Description:
You are an expert in time series analysis. {subset_description}
I am trying to write a prompt used to generate a description for the data, where the generated description can enhance the zero-shot classification of the data. 
There are 2 descriptions to describe the data:
** Text 0 **: {initial_prompt[0]}
** Text 1 **: {initial_prompt[1]}
** Text 2 **: {initial_prompt[2]}
Which description better describes the following image? Provide your analysis and the choice you believe is better.
### Answer Format:
-- analyze --:
-- answer --: text 0 or text 1 or text 2"""
            # print("class 0", initial_prompt[0])
            # print("class 1", initial_prompt[1])
            input_prompt = test_prompt

            answer = gpt_chat_vision(input_prompt, sample)

            error = answer.split('answer')[0]
            choose_content = answer.split('answer')[1]
            print("prediction:", choose_content)
            numbers = re.findall(r'\d+', choose_content)

            last_number = int(numbers[-1])

            if last_number == test_class:
                print(f"sample {i},index {selected_indices[i]},choose {last_number},gt {test_class}")
                correct = correct + 1
            else:
                print(f"sample {i},index {selected_indices[i]},choose {last_number},gt {test_class}")
                true_prompt = initial_prompt[test_class]
                false_prompt = initial_prompt[last_number]
                feature, analysis = reflect(sample, selected_indices[i], true_prompt, false_prompt, error,
                                            test_class, last_number)
                initial_prompt[test_class] = modify(
                    current_prompt=true_prompt,
                    feature=feature,
                    analysis=analysis,
                    true_label=test_class,
                    misclassified_image_b64=sample  # ← 关键：传入错分图像
                )
                updated = True
                print(
                    f"Updated class {test_class} (correct class) based on sample {i}, model chose class {last_number}")

        cycle_count += 1
        if not updated:
            print("No updates in this cycle. Stopping.")
            print(initial_prompt[test_class])
            break
        if correct >= 0.7 * test_num:
            print(f"Correct: {correct} / {test_num}")
            break
    return initial_prompt


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

    # 找出所有同类样本的索引（排除自己）
    # same_class_indices = (labels_tensor == true_label).nonzero(as_tuple=True)[0]
    # same_class_indices = same_class_indices[same_class_indices != current_data_index]
    wrong_class_indices = (labels_tensor == wrong_label).nonzero(as_tuple=True)[0]

    # if len(same_class_indices) == 0:
    #     # 退化处理：随机选3个同类（理论上不会发生）
    #     same_class_indices = (labels_tensor == true_label).nonzero(as_tuple=True)[0]

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
    feature, analysis = analyze_misclassification(
        misclassified_image_b64=mis_img,
        nearest_same_class_images_b64=neighbor_imgs,
        correct_description=true_prompt,
        wrong_description=false_prompt,
        true_label=true_label,
        wrong_label=wrong_label
    )
    return feature, analysis

    #     modify_prompt = f'''My current data is "{current_data}"\n
    # 3 sample with same label is {sample1}\n{sample2}\n{sample3}\n
    # 3 sample with wrong label is {sample4}\n{sample5}\n{sample6}\n
    # My current prompt is: {current_prompt}\n
    # But this prompt descriptions that are too simple, similar and vague, making it difficult to distinguish which description correctly matches the class and leading to the wrong description being chosen for the following examples {error_string}
    # Give a reasons why the prompt could have gotten these examples wrong.Modify the Class {true_label} prompt but do not change its structure.\n### Answer Format:


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
                "### Task Description:\n"
                "1. The 2 description is too simple, similar and vague, making it "
                "difficult to distinguish which description correctly matches the image and leading to the wrong"
                "description being chosen for the following examples.\n"
                "2. Compare the visual patterns in the images. Explain why the wrong description might appear more "
                "plausible for the misclassified sample.\n"
                "3. Identify what key characteristics of class {true_label} are missing or underemphasized in the "
                "correct description.\n"
                "### Answer Format:\n"
                "-- Explanation -- :[]\n"
                "-- Key Characteristics -- :[]\n"
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
        analysis = answer.split("Explanation")[1].split("Key Characteristics")[0].strip()
        feature = answer.split("Key Characteristics")[1].strip()
    except Exception as e:
        print("⚠️ Parsing failed:", e)
        print("Raw answer:", answer)
        # 退化：返回原描述
        return correct_description, answer

    print("✅ Analysis:", analysis)
    print("✨ Feature:", feature)

    return feature, analysis


def modify(current_prompt, feature, analysis, true_label, misclassified_image_b64):
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
    # 构建多模态输入内容（兼容 OpenAI API）
    content = [
        {
            "type": "text",
            "text": (
                f"You are an expert in time series analysis and prompt design for zero-shot classification.\n{subset_description}\n"
                f"### Task Description\n"
                f"Improve the textual description for **Class {true_label}** so that it correctly describes the time "
                f"series shown in the image below.\n\n"
                f"### Current Description of class {true_label}:\n"
                f'"{current_prompt}"\n\n'
                f"### Missing Key Features :\n"
                f"{feature}\n\n"
                # f"### The reasons for the model's incorrect classification :\n"
                # f"{analysis}\n\n"
                f"### Requirements:\n"
                f"- Add or modify statements on the textual description instead of deleting sentence.\n"
                f"- Incorporate the missing features above.\n"
                f"- Avoid vague terms like 'varies' or 'complex'.\n"
                f"- DO NOT change the structure of the origin description.\n\n"
                f"### Answer Format:\n"
                f"** Pattern of Class {true_label} **:\n"
                f"** Difference to other class **:\n"
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
    return improved


if __name__ == "__main__":
    class_feature = []

    for n in range(categories):
        file_path = f"log/{dataset_name}_{n}_initial.txt"

        with open(file_path, 'r') as file:
            contents = file.read()

            class_feature.append(contents)

    test(class_feature, 5, 2)
