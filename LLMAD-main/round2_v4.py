import base64
import os
import re

import openai
import pandas as pd
import torch
from openai import OpenAI
import json
import time
from dataloader_v2 import load_data
from fastdtw import fastdtw
from round1_v3 import encode_image_to_base64

gpt_model = "gpt-5-nano"

OPENAI_API_KEY = "sk-J3azhkbBoUT2YQC8Sl2KLsqBnKC5LcamDvTxWco3ZglWdcgJ"

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://api.chatanywhere.tech/v1"
)
categories = 2

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
# subset_description = (
#     "The arrowhead data consists of outlines of the images of arrowheads. The shapes of the projectile points "
#     "are converted into a time series using the angle-based method. The classification of projectile points is "
#     "an important topic in anthropology. The classes are based on shape distinctions such as the presence and "
#     "location of a notch in the arrow. The three classes are called Avonlea(class 0), Clovis(class 1) and Mix("
#     "class 2)")
dataset_name = "Computers"


def gpt_chat_vision(text_prompt, image_b64, model):
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


def test(test_num, selected_indices, test_class, temp):
    correct_num = 0
    wrong_num = 0
    wrong_indices = []
    for i in range(test_num):

        sample = plot_series_to_base64(selected_indices[i], test_class)
        initial_prompt = load_prompt(temp)
        test_prompt = (f"You are an expert in time series analysis. {subset_description}.\n "
                       "There are 2 descriptions to describe the data:\n "
                       f"**** Text 0 ****\n: {initial_prompt[0]}\n"
                       f"**** Text 1 ****\n: {initial_prompt[1]}\n"
                       # f"**** Text 2 ****: {initial_prompt[2]}\n"
                       "### Task Description:\n"
                       "You will be provided a time series image."
                       "Think step by step:\n"
                       "-- Analyze the Time Series Pattern. Focus on global or local feature such as trend, "
                       "cycle, shape, symmetry, stationarity, amplitude, rate of change, spikes, etc.\n"
                       "-- Compare and match the features to each descriptions. Pay attention to the common "
                       "feature and the partial-sample features.\n"
                       "-- Based on above analysis, choose the description that "
                       "better describes the following image.\n"
                       "### Answer in json Format:\n"
                       "{"
                       "    \"analysis\":[Your reasoning process],\n"
                       "    \"answer\":[text 0 / text 1]\n"
                       "}"
                       )
        # print("class 0", initial_prompt[0])
        # print("class 1", initial_prompt[1])
        input_prompt = test_prompt

        answer = gpt_chat_vision(input_prompt, sample, model="gpt-5-nano")

        error = answer.split('answer')[0]
        choose_content = answer.split('answer')[1]

        numbers = re.findall(r'\d+', choose_content)

        last_number = int(numbers[0])

        if last_number == test_class:
            correct_num += 1
        else:
            wrong_indices.append(selected_indices[i])
            wrong_num += 1
        print(f"sample {i},index {selected_indices[i]},choose {last_number},gt {test_class}")

    return correct_num, wrong_num, wrong_indices


def improve(test_num, test_class):
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
    correct_num, wrong_num, wrong_indices = test(test_num, selected_indices, test_class, False)
    if correct_num >= test_num:
        return

    # 4. 调用测试循环（注意：传入的是整个 test data tensor + 选中的索引）
    test_cycle(correct_num, wrong_num, wrong_indices, selected_indices, test_class, max_cycle=2)


def plot_series_to_base64(idx, test_class):
    with open(f"plots/{dataset_name}_train_nolabel/class{test_class}/{idx}.png", "rb") as f:
        b64 = base64.b64encode(f.read()).decode('utf-8')

    return b64


def test_cycle(correct_num, wrong_num, selected_indices, all_indices, test_class, max_cycle):
    for i in range(wrong_num):
        curr_cycle = 0
        while curr_cycle != 2 * max_cycle:
            sample = plot_series_to_base64(selected_indices[i], test_class)
            initial_prompt = load_prompt(temp=True)
            test_prompt = (f"You are an expert in time series analysis. {subset_description}.\n "
                           "There are 2 descriptions to describe the data:\n "
                           f"**** Text 0 ****\n: {initial_prompt[0]}\n"
                           f"**** Text 1 ****\n: {initial_prompt[1]}\n"
                           # f"**** Text 2 ****: {initial_prompt[2]}\n"
                           "### Task Description:\n"
                           "You will be provided a time series image."
                           "Think step by step:\n"
                           "-- Analyze the Time Series Pattern. Focus on global or local feature such as trend, "
                           "cycle, shape, symmetry, stationarity, amplitude, rate of change, spikes, etc.\n"
                           "-- Compare and match the features to each descriptions. Pay attention to the common "
                           "feature and the partial-sample features.\n"
                           "-- Based on above analysis, choose the description that "
                           "better describes the following image.\n"
                           "### Answer in json Format:\n"
                           "{"
                           "    \"analysis\":[Your reasoning process],\n"
                           "    \"answer\":[text 0 / text 1]\n"
                           "}"
                           )
            # print("class 0", initial_prompt[0])
            # print("class 1", initial_prompt[1])
            input_prompt = test_prompt

            answer = gpt_chat_vision(input_prompt, sample, model="gpt-5-nano")

            error = answer.split('answer')[0]
            choose_content = answer.split('answer')[1]

            numbers = re.findall(r'\d+', choose_content)

            last_number = int(numbers[0])

            if last_number == test_class and curr_cycle == 0:
                print(f"sample {i},index {selected_indices[i]},choose {last_number},gt {test_class}")
                with open(f"log/{dataset_name}/feature_list/{dataset_name}_{test_class}_temp.json", "r",
                          encoding="utf-8") as f_source:
                    source_data = json.load(f_source)  # 加载为字典

                # 2. 覆盖写入目标JSON（清空原有内容，写入新内容）
                with open(f"log/{dataset_name}/feature_list/{dataset_name}_{test_class}_key_features.json", "w",
                          encoding="utf-8") as f_target:
                    json.dump(source_data, f_target, ensure_ascii=False, indent=2)
                curr_cycle = 2 * max_cycle

            elif last_number == test_class and curr_cycle != 0:
                correct_num, _, _ = test(len(all_indices), all_indices, test_class, True)



            else:
                print(f"sample {i},index {selected_indices[i]},choose {last_number},gt {test_class}")
                true_prompt = initial_prompt[test_class]
                false_prompt = initial_prompt[last_number]
                feature, analysis, suggestion, top3_indices = reflect(sample, selected_indices[i],
                                                                      true_prompt,
                                                                      false_prompt, error,
                                                                      test_class, last_number)

                if curr_cycle < max_cycle:
                    modify(
                        feature=feature,
                        analysis=analysis,
                        suggestion=suggestion,
                        true_label=test_class,
                        misclassified_image_b64=sample,
                        correct=curr_cycle
                    )
                    curr_cycle += 1
                    print(
                        f"Updated class {test_class} (correct class) based on sample {i}, model chose class {last_number}")
                # else:
                #     modify_wrong(
                #         feature=feature,
                #         analysis=analysis,
                #         true_label=test_class,
                #         wrong_label=last_number,
                #         misclassified_image_b64=sample,
                #         index=top3_indices[0]
                #     )
                #     correct += 1
                #     print(
                #         f"Updated class {last_number} (wrong class) based on sample {i}, model chose class {last_number}")


def reflect(mis_img, current_data_index, true_prompt, false_prompt, error_string, true_label, wrong_label):
    # 加载训练数据
    file_path = f'data/{dataset_name}/{dataset_name}_TRAIN.txt'
    df = pd.read_csv(file_path, header=None, sep='\s+')
    labels = df.iloc[:, 0].values - 1  # 第一列为标签
    data = df.iloc[:, 1:].values  # 其余为时间序列
    # 转为 tensor
    labels_tensor = torch.tensor(labels, dtype=torch.long)  # 假设标签从1开始
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
    feature, analysis, suggestion = analyze_misclassification(
        misclassified_image_b64=mis_img,
        nearest_same_class_images_b64=neighbor_imgs,
        correct_description=true_prompt,
        wrong_description=false_prompt,
        true_label=true_label,
        wrong_label=wrong_label
    )
    return feature, analysis, suggestion, top3_indices

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
                "**Correct Description:**\n"
                f"{correct_description}\n\n"
                f"- Predicted class: {wrong_label}\n\n"
                "**Wrong Description:**\n"
                f"{wrong_description}\n\n"
                "The 2 description is too simple, similar and vague, making it "
                "difficult to distinguish which description correctly matches the image and leading to the wrong"
                "description being chosen for the following examples.\n"
                "### Task Description:\n"
                "1. Compare the visual patterns between the classes. "
                "2. Explain why the wrong description might appear more "
                "plausible for the misclassified sample.\n"
                "3. Identify what characteristics/differences are missing or underemphasized or overemphasized in the "
                f"correct description(class {true_label}). \n"
                "4. Give add/modify suggestions to the correct description based on the above "
                "characteristics/differences. Your suggestion should not be to delete or deny the original features, "
                "but to modify or further refine them\n"
                "### Answer in json Format:\n"
                "{"
                "   \"explanation\":,\n"
                "   \"characteristics\":\n"
                "   \"suggestions\":,\n"
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
        model="gpt-5-nano",  # 或 gpt-4o-mini, claude-3-5-sonnet 等
        messages=messages,
        temperature=0.2
    )

    answer = response.choices[0].message.content

    analysis = answer.split("explanation")[1].split("characteristics")[0].strip()
    feature = answer.split("characteristics")[1].split("suggestions")[0].strip()
    # feature = ""
    suggestion = answer.split("suggestions")[1].strip()
    # suggestion = ""
    # print("✅ Analysis:", analysis)
    # print("✨ Feature:", feature)

    return feature, analysis, suggestion


# def modify(feature, analysis, suggestion, true_label, misclassified_image_b64):
#
#
#     initial_prompt = load_prompt()
#     current_prompt = initial_prompt[true_label]
#     print("CURRENT PROMPT:", current_prompt)
#     common_feature = current_prompt.split('** The feature that appear')[0]
#     content = [
#         {
#             "type": "text",
#             "text": (
#                 f"You are an expert in time series analysis and prompt design for zero-shot classification."
#                 f"The dataset is called \"{dataset_name}\".\n"
#                 f"{subset_description}\n"
#                 f"### Task Description\n"
#                 f"The image belongs to class {true_label}, but the model makes a wrong prediction since insufficient "
#                 f"description."
#                 f"Improve the textual description of class {true_label} so that it correctly "
#                 f"describes the time series shown in the image below.\n"
#                 f"### Current Description:\n"
#                 f'"{common_feature}"\n\n'
#                 f"### The reasons for the model's incorrect classification :\n"
#                 f"{analysis}\n\n"
#                 # f"### Missing Features :\n"
#                 # f"{feature}\n\n"
#                 f"### Add/Modify Suggestions :\n"
#                 f"{suggestion}\n\n"
#                 f"### Requirements:\n"
#                 "- Analyze the features of the image.\n"
#                 # "and check whether Missing Features they really exist in the image before modification"
#                 f"- Add or modify statements on the description. \n"
#                 # f"Incorporate the missing features above.\n"
#                 f"- Only output the description and do not output any other information. Use the star point"
#                 f"(*) to separate feature.\n"
#                 "DO NOT DELETE the original feature, but you can modify.\n"
#                 # "Make least modifications as possible.\n"
#                 "Check if there is any specific feature in the image but do not exist in the common feature. You"
#                 "can only add one or zero feature.\n"
#                 f"### Answer Format:\n"
#                 "common feature\n"
#                 "* [...]\n"
#                 "* [...]\n"
#                 "specific feature\n"
#                 "* [...]\n"
#             )
#         },
#         {
#             "type": "image_url",
#             "image_url": {
#                 "url": f"data:image/png;base64,{misclassified_image_b64}"
#             }
#         }
#     ]
#
#     # 调用多模态大模型（如 GPT-4o）
#     messages = [{"role": "user", "content": content}]
#
#     response = client.chat.completions.create(
#         model="gpt-5-nano",  # 或 gpt-4o-mini
#         messages=messages,
#         temperature=0.2
#     )
#     answer = response.choices[0].message.content
#
#     improved = answer.strip()
#
#     print("✨ Improved prompt for class", true_label, ":", improved)
#     # 替换列表中的第 type_value 项
#     result = {}
#     common_match = re.search(
#         r'^common\s+feature\s*$(.*?)(?=^\s*(?:non-common\s+feature|common\s+feature|$))',
#         improved,
#         re.IGNORECASE | re.MULTILINE | re.DOTALL
#     )
#
#     non_common_match = re.search(
#         r'non-common\s+feature',
#         improved,
#         re.IGNORECASE | re.MULTILINE | re.DOTALL
#     )
#
#     # 提取 * 开头的行
#     def extract_starred_items(block_text):
#         if not block_text:
#             return []
#         # 找出所有以 * 开头的行（允许前面有空格）
#         return [
#             line.strip()[2:].strip()  # 去掉 "* "
#             for line in block_text.split('\n')
#             if line.strip().startswith('*')
#         ]
#
#     # 处理 part feature
#     if common_match:
#         for desc in extract_starred_items(common_match.group(1)):
#             result[desc] = "common feature"
#
#     # 处理 non-common feature
#     if non_common_match:
#         for desc in extract_starred_items(non_common_match.group(1)):
#             result[desc] = "non-common feature"
#
#     with open(f"log/{dataset_name}/feature_list/{dataset_name}_{true_label}_temp.json", "w", encoding="utf-8") as f:
#         json.dump(result, f, ensure_ascii=False, indent=2)


def modify(feature, analysis, suggestion, true_label, misclassified_image_b64, correct):
    initial_prompt = load_prompt(correct)
    current_prompt = initial_prompt[true_label]
    print("CURRENT PROMPT:", current_prompt)
    # common_feature = current_prompt.split('** The feature that appear')[0]
    content = [
        {
            "type": "text",
            "text": (
                f"You are an expert in time series analysis and prompt design for zero-shot classification."
                f"The dataset is called \"{dataset_name}\".\n"
                f"{subset_description}\n"
                f"### Task Description\n"
                f"The image belongs to class {true_label}, but the model makes a wrong prediction since insufficient "
                f"description."
                f"Improve the textual description of class {true_label} so that it correctly "
                f"describes the time series shown in the image below.\n"
                f"### Current Description:\n"
                f'"{current_prompt}"\n\n'
                f"### The reasons for the model's incorrect classification :\n"
                f"{analysis}\n\n"
                f"### missing/underemphasized/overemphasized features:\n"
                f"{feature}\n\n"
                f"### Add/Modify Suggestions :\n"
                f"{suggestion}\n\n"
                f"### Requirements:\n"
                # "- Analyze the features of the image.\n"
                # "and check whether Missing Features they really exist in the image before modification"
                f"- Add or modify statements on the description. DO NOT DELETE the feature, but you can modify or add.\n"
                f"- Incorporate the missing/underemphasized/overemphasized features above.\n"
                f"- Only output the description and do not output any other information. Use the star point"
                f"(*) to separate feature.\n"
                "DO NOT use absolute words like \"all\" \"none\". You can use e.g. \"in most series, xxx\" \"in some "
                "series, xxx\""
                # "Make least modifications as possible.\n"
                # "Check if there is any specific feature in the image but do not exist in the common feature. You"
                # "can only add one or zero feature.\n"
                f"### Answer Format:\n"
                "common feature\n"
                "* [...]\n"
                "partial-sample feature\n"
                "* [...]\n"
            )
        }
    ]

    # 调用多模态大模型（如 GPT-4o）
    messages = [{"role": "user", "content": content}]

    response = client.chat.completions.create(
        model="deepseek-r1",  # 或 gpt-4o-mini
        messages=messages,
        temperature=0.2
    )
    answer = response.choices[0].message.content

    improved = answer.strip()

    print("✨ Improved prompt for class", true_label, ":", improved)
    pattern = r'(common feature|partial-sample feature)\s*([\s\S]*?)(?=(common feature|partial-sample feature|$))'
    matches = re.findall(pattern, improved)

    feature_dict = {}

    for match in matches:
        category = match[0].strip()
        # 提取该类别下的所有特征项（以*开头的行）
        items = re.findall(r'\*\s*(.*?)(?=\s*\*|$)', match[1], re.DOTALL)

        for item in items:
            # 清理特征文本（去除多余空白、换行）
            cleaned_item = re.sub(r'\s+', ' ', item).strip()
            if cleaned_item:  # 跳过空行
                feature_dict[cleaned_item] = category
    with open(f"log/{dataset_name}/feature_list/{dataset_name}_{true_label}_temp.json", "w", encoding="utf-8") as f:
        json.dump(feature_dict, f, ensure_ascii=False, indent=2)


def modify_wrong(feature, analysis, true_label, wrong_label, misclassified_image_b64, index):
    img_path = os.path.join(f"plots/{dataset_name}_train/class{wrong_label}/{index}.png")

    initial_prompt = load_prompt(1)
    current_prompt = initial_prompt[true_label]
    print("CURRENT PROMPT:", current_prompt)

    print("CURRENT PROMPT:", current_prompt)
    content = [
        {
            "type": "text",
            "text": (
                f"You are an expert in time series analysis and prompt design for zero-shot classification."
                f"\n{subset_description}\n"
                f"### Task Description\n"
                f"The image belongs to class {true_label}, but the model chooses class {wrong_label}\n."
                f"I will provide 1 image belongs to class {wrong_label} but similar to the "
                f"misclassified image to help you distinguish.\n"
                "You need to find the difference between 2 images, and"
                f"improve the textual description of class {wrong_label} so that it "
                f"DO NOT match the misclassified time series\n"
                f"### Current Description of class {wrong_label}:\n"
                f'"{current_prompt}"\n\n'
                f"### The reasons for the model's incorrect classification :\n"
                f"{analysis}\n\n"
                f"### Requirements:\n"
                f"- Add or modify statements on the textual description.\n"
                f"- DO NOT change the structure of the origin description.\n"
                f"- Only output the modified description and do not output any other information."
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
        model="gpt-5-nano",  # 或 gpt-4o-mini
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
    current_prompt = improved

    # 将更新后的列表写回文件
    with open(file_path2, 'w', encoding='utf-8') as file2:
        json.dump(current_prompt, file2, ensure_ascii=False, indent=2)

    # k = 6
    # cluster = 6
    # data_splits = load_data(dataset_name, n_clusters=cluster, top_k=k)
    # selected_positions = data_splits['representative_indices']
    # common2(wrong_label, selected_positions, "log")


def load_prompt(temp):
    class_feature = []

    for n in range(categories):
        # feature = f"\n** The common feature of class {n}: **\n"
        feature = ""
        if not temp:
            file_path = f"log/{dataset_name}/feature_list/{dataset_name}_{n}_key_features.json"
        else:
            file_path = f"log/{dataset_name}/feature_list/{dataset_name}_{n}_temp.json"
        with open(file_path, 'r', encoding='utf-8') as file:
            features_dict = json.load(file)

        # 提取 common feature 和 part feature 的键
        common_feats = [k for k, v in features_dict.items() if v == "common feature"]
        part_feats = [k for k, v in features_dict.items() if v == "partial-sample feature"]

        # 格式化为字符串（每项一行，缩进两个空格）
        common_str = "\n".join(f"  - {feat}" for feat in common_feats) if common_feats else "  None"
        part_str = "\n".join(f"  - {feat}" for feat in part_feats) if part_feats else "  None"
        feature_text = (
            f"*  The feature that appear at most sample of class {n}(common feature):\n{common_str}\n"
            f"** The feature that appear at some sample of class {n}(partial-sample features):\n{part_str}\n"
        )
        # file_path2 = f"log/{dataset_name}_{n}_initial.json"
        # with open(file_path2, 'r', encoding='utf-8') as file2:
        #     contents2 = list(json.load(file2))
        #     contents3 = "\n".join(
        #         f"The type {i} time series of class {n} is: {item}"
        #         for i, item in enumerate(contents2)
        #     )
        #
        # feature += contents3
        class_feature.append(feature_text)
    return class_feature


if __name__ == "__main__":
    improve(10, 0)
