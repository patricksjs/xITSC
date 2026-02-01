import base64
import os
import re
from pathlib import Path
from openai import OpenAI
import json
from dataloader_v2 import load_data, random_choice
from fastdtw import fastdtw

OPENAI_API_KEY = "sk-BgMrhX3ZCx1TOeUFhOXIcu8iJfz8ahXbWR56zFQPC6pdTj0k"

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
# subset_description = ("The data was collected using a tri-axial accelerometer on the dominant wrist "
#           "whilst conducting 4 different activities: SEIZURE MIMICKING(class 0) with seating after the mimicked "
#           "seizure, WALKING(class 1) with different paces and"
#           "gestures, RUNNING(class 2) with running a 40 meters long corridor, SAWING(class 3) with a saw and "
#           "during 30 seconds."
#           "The sampling frequency was 16 Hz. The activities lasted about 13 seconds")
# subset_description = ("These dataset were taken from data recorded as part of government sponsored study called "
#                       "Powering the"
#                       "Nation. The intention was to collect behavioural data about how consumers use electricity "
#                       "within the home"
#                       "to help reduce the UK's carbon footprint. The data contains readings from 251 households, "
#                       "sampled in"
#                       "two-minute intervals over a month. Each series is length 720 (24 hours of readings taken every "
#                       "2 minutes). There are 7 classes.")
# subset_description = (
#     "The dataset are electrooculography signal (EOG). EOG is measurements of the electrical potential between "
#     "electrodes placed at points close to the eyes. The EOG recording device is BlueGain, a commercial "
#     "biomedical amplifier. The sampling rate was 1.0KHz. This "
#     "dataset includes 6 participants eye-writing 7 types of Japanese Katakana strokes. There are 7 classes.")
# subset_description = ("It is a synthetic dataset designed to simulate instrumentation failures in a nuclear power plant, "
#           "with 4 classes")
subset_description = (
    "This is a dataset with phase-aligned starlight curves of length 1,024, whose class has been determined by "
    "an expert.There are 3 classes.")
# subset_description = ("The dataset has 3 classes. Data from each class is standard normal noise plus an offset term which differs "
#           "for each class.")

# subset_description = ("The two classes are a normal heartbeat(class 0) and a Myocardial Infarction(class 1). The electrocardiogram "
#           "of a normal heartbeat and that of a myocardial infarction are quite different, mainly reflected in the "
#           "following core features:\n "
#           "Normal heartbeat: The ST segment is straight and coincides with the baseline. T waves are usually upright "
#           "and symmetrical.\n"
#           "Myocardial infarction: The ST segment will show obvious abnormalities. The most common two situations are: "
#           "one is the arched back upward type elevation, and the other is significant depression. The T wave becomes "
#           "abnormally sharp (acute phase) or symmetrically inverted (necrotic phase).")
dataset_name = "StarLightCurves"


def gpt_chat_vision(text_prompt, image_b64, model):
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


def test(selected_indices, test_class, temp):
    correct_num = 0
    wrong_indices = []
    wrong_choice = []
    wrong_reason = []
    for i in range(len(selected_indices)):

        sample = plot_series_to_base64(selected_indices[i], test_class)
        initial_prompt = load_prompt(temp)
        test_prompt = (f"You are an expert in time series analysis. {subset_description}.\n "
                       f"There are {categories} descriptions to describe the data:\n "
                       f"**** Text 0 ****\n: {initial_prompt[0]}\n"
                       f"**** Text 1 ****: {initial_prompt[1]}\n"
                       f"**** Text 2 ****: {initial_prompt[2]}\n"
                       # f"**** Text 3 ****: {initial_prompt[3]}\n"
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
                       "    \"answer\":[text 0 / text 1 / text 2]\n"
                       "}"
                       )
        # print("class 0", initial_prompt[0])
        # print("class 1", initial_prompt[1])
        input_prompt = test_prompt

        answer = gpt_chat_vision(input_prompt, sample, model=gpt_model)

        reason = answer.split('answer')[0]
        choose_content = answer.split('answer')[1]

        numbers = re.findall(r'\d+', choose_content)

        last_number = int(numbers[0])

        if last_number == test_class:
            correct_num += 1
        else:
            wrong_indices.append(selected_indices[i])
            wrong_choice.append(last_number)
            wrong_reason.append(reason)
        print(f"sample {i},index {selected_indices[i]},choose {last_number},gt {test_class}")

    return correct_num, wrong_indices, wrong_choice, wrong_reason


# def improve(max_num, wrong_indices, wrong_choice, wrong_reason, all_indices, cycle, test_class):
#     curr_max_num = max_num
#     for i in range(len(wrong_indices)):
#         correct_cycle = 0
#         while correct_cycle < cycle and curr_max_num < len(all_indices) * 0.8:
#             after_num, wrong_indices2, wrong_choice2, wrong_reason2 = test_cycle_correct(wrong_indices[i],
#                                                                                          wrong_choice[i],
#                                                                                          wrong_reason[i], all_indices,
#                                                                                          test_class)
#             print(f"modify {wrong_indices[i]} after correct cycle {correct_cycle}: {after_num} correct")
#             if after_num > curr_max_num:
#                 print(after_num, curr_max_num)
#                 overwrite(test_class)
#                 curr_max_num = after_num
#                 if after_num == len(all_indices):
#                     return
#                 improve(curr_max_num, wrong_indices2, wrong_choice2, wrong_reason2, all_indices, cycle, test_class)
#                 break
#
#             if after_num < curr_max_num / 2:
#                 print(after_num, curr_max_num)
#                 break
#             correct_cycle += 1
#
#         overwrite_back(test_class)


def improve(max_num, wrong_indices, wrong_choice, wrong_reason, all_indices, max_iterations, test_class, num):
    """批量修改并迭代优化"""
    curr_max_num = max_num
    iteration = 0

    # 记录修改历史，避免重复修改
    modification_history = set()

    while iteration < max_iterations and curr_max_num < len(all_indices) * 0.8:
        print(f"\n=== 迭代 {iteration + 1} ===")

        # 收集所有错误样本的修改建议
        candidate_modifications = []

        for i in range(len(wrong_indices)):
            # if wrong_indices[i] in modification_history:
            #     continue  # 跳过已经修改过的样本

            print(f"分析错误样本 {wrong_indices[i]}...")
            candidate = generate_candidate_modification(
                wrong_indices[i], wrong_choice[i], wrong_reason[i],
                all_indices, test_class, random_indices, num
            )
            if candidate:
                candidate_modifications.append(candidate)

        if not candidate_modifications:
            print("没有更多的候选修改")
            break

        # 测试所有候选修改，选择最佳的一个
        best_candidate = None
        best_score = curr_max_num

        for i, candidate in enumerate(candidate_modifications):
            print(f"测试候选修改 {i + 1}/{len(candidate_modifications)}...")

            # 应用候选修改
            for n in range(num):
                new_description = apply_candidate_modification(candidate, test_class, n)

                # 测试修改后的效果
                after_num, new_wrong_indices, new_wrong_choice, new_wrong_reason = test(
                    selected_indices=all_indices, test_class=test_class, temp=True
                )

                print(f"候选修改 {i + 1} 效果: {after_num}/{len(all_indices)} (之前: {curr_max_num}/{len(all_indices)})")

                # 记录最佳候选
                if after_num > best_score:
                    best_score = after_num
                    best_candidate = {
                        'candidate': candidate,
                        'score': after_num,
                        'wrong_indices': new_wrong_indices,
                        'wrong_choice': new_wrong_choice,
                        'wrong_reason': new_wrong_reason,
                        'desc': new_description
                    }

                # 恢复原始状态测试下一个候选
                # overwrite_back(test_class)

        # 应用最佳候选修改
        if best_candidate and best_score > curr_max_num:
            print(f"应用最佳候选修改，准确率从 {curr_max_num} 提升到 {best_score}")
            # apply_candidate_modification(best_candidate['candidate'], test_class)
            # overwrite(test_class)  # 永久保存最佳修改
            with open(f"log/{dataset_name}/feature_list_v2/{dataset_name}_{test_class}_{gpt_model}_3round_features.json", "w",
                      encoding="utf-8") as f_target:
                json.dump(best_candidate['desc'], f_target, ensure_ascii=False, indent=2)

            curr_max_num = best_score
            wrong_indices = best_candidate['wrong_indices']
            wrong_choice = best_candidate['wrong_choice']
            wrong_reason = best_candidate['wrong_reason']

            # 记录已修改的样本
            modification_history.add(best_candidate['candidate']['wrong_indice'])

            # # 如果已经达到完美，提前结束
            # if curr_max_num == len(all_indices):
            #     print("达到完美准确率，提前结束迭代")
            #     break
        else:
            print("没有找到能提升准确率的修改，结束迭代")
            break

        iteration += 1

    print(f"迭代结束，最终准确率: {curr_max_num}/{len(all_indices)}")


def generate_candidate_modification(wrong_indice, wrong_choice, wrong_reason, all_indices, test_class, selected, num):
    """生成候选修改"""
    sample = plot_series_to_base64(wrong_indice, test_class)
    initial_prompt = load_prompt(temp=True)

    print(f"为错误样本 {wrong_indice} 生成修改建议 (预测: {wrong_choice}, 真实: {test_class})")

    true_prompt = initial_prompt[test_class]
    false_prompt = initial_prompt[wrong_choice]

    result, top3_indices = reflect(
        sample, wrong_indice, true_prompt, false_prompt,
        wrong_reason, test_class, wrong_choice, isCorrect=True, selected=selected, num=num
    )

    return {
        'wrong_indice': wrong_indice,
        'wrong_choice': wrong_choice,
        'reflect': result,
        'sample_b64': sample
    }


def apply_candidate_modification(candidate, test_class, n):
    """应用候选修改"""
    new_description = modify_correct(
        feature=candidate['reflect'][n][0],
        analysis=candidate['reflect'][n][1],
        suggestion=candidate['reflect'][n][2],
        true_label=test_class,
        misclassified_image_b64=candidate['sample_b64']
    )
    print(f"应用对样本 {candidate['wrong_indice']} 的修改")
    return new_description


def plot_series_to_base64(idx, test_class):
    with open(f"plots/{dataset_name}_train_nolabel/class{test_class}/{idx}.png", "rb") as f:
        b64 = base64.b64encode(f.read()).decode('utf-8')

    return b64


def test_cycle_correct(wrong_indice, wrong_choice, wrong_reason, all_indices, test_class):
    sample = plot_series_to_base64(wrong_indice, test_class)
    initial_prompt = load_prompt(temp=True)

    print(f"index {wrong_indice},choose {wrong_choice},gt {test_class}")
    true_prompt = initial_prompt[test_class]
    false_prompt = initial_prompt[wrong_choice]
    feature, analysis, suggestion, top3_indices = reflect(sample, wrong_indice,
                                                          true_prompt,
                                                          false_prompt, wrong_reason,
                                                          test_class, wrong_choice, isCorrect=True,
                                                          selected=random_indices)
    new_desc = modify_correct(
        feature=feature,
        analysis=analysis,
        suggestion=suggestion,
        true_label=test_class,
        misclassified_image_b64=sample
    )
    print(
        f"Updated class {test_class} (correct class) based on sample {wrong_indice}, model chose class {wrong_choice}")

    after_num2, wrong_indices2, wrong_choice2, wrong_reason2 = test(selected_indices=all_indices, test_class=test_class,
                                                                    temp=True)

    return after_num2, wrong_indices2, wrong_choice2, wrong_reason2


def reflect(mis_img, current_data_index, true_prompt, false_prompt, error_string, true_label, wrong_label, isCorrect,
            selected, num):
    data, labels, _ = load_data(data_name=dataset_name)['train']

    # 获取当前样本的时间序列（注意：current_data_index 是原始数据中的索引）
    current_series = data[current_data_index]
    wrong_class_indices = (labels == wrong_label).nonzero()
    # wrong_class_indices = (labels == wrong_label).nonzero(as_tuple=True)[0]
    wrong_set = set(wrong_class_indices[0].tolist())
    select_set = set(selected[wrong_label])
    intersection_list = sorted(list(wrong_set & select_set))
    wrong_class_indices = intersection_list
    # 计算 DTW 距离并排序
    distances = []
    for idx in wrong_class_indices:
        series = data[idx]

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
    result = analyze_misclassification(
        misclassified_image_b64=mis_img,
        nearest_same_class_images_b64=neighbor_imgs,
        correct_description=true_prompt,
        wrong_description=false_prompt,
        true_label=true_label,
        wrong_label=wrong_label,
        isCorrect=isCorrect,
        num=num
    )
    return result, top3_indices


def analyze_misclassification(
        misclassified_image_b64,
        nearest_same_class_images_b64,  # list of 3 base64 strings
        correct_description,
        wrong_description,
        true_label,
        wrong_label,
        isCorrect,
        num
):
    if isCorrect:
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
                    f"1. I will give you 3 images from wrong class(class {wrong_label}). Compare the time patterns "
                    f"between the misclassified image and these 3 images."
                    "2. Explain why the wrong description might appear more "
                    "plausible for the misclassified sample.\n"
                    "3. Identify what characteristics/differences are missing or underemphasized or overemphasized in "
                    f"the correct description(class {true_label}). \n"
                    "4. Give add/modify suggestions to the correct description based on the above characteristics"
                    "/differences. Your suggestion should not delete or deny the original feature. "
                    "but modify or further refine them.\n"
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
                "text": f"Three samples from wrong class (class {wrong_label}):"
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

    else:
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
                    f"I will give you 3 images from wrong class(class {wrong_label}). You need to find the difference "
                    f"between these images and the misclassified images. And then"
                    f"improve the wrong description of class {wrong_label} so that it "
                    f"DO NOT match the misclassified time series.\n"
                    "### Task Description:\n"
                    f"1. Compare the time patterns between the misclassified image and 3 images of class {wrong_label}."
                    "2. Explain why the wrong description might appear more "
                    "plausible for the misclassified sample.\n"
                    "3. Identify what characteristics/differences are vague or overemphasized in "
                    f"the wrong description(class {wrong_label}). \n"
                    "4. Give add/modify suggestions to the wrong description based on the above characteristics"
                    "/differences. Your suggestion should not delete or deny the original feature. "
                    "but modify or further refine them.\n"
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
                "text": f"Three samples from wrong class (class {wrong_label}):"
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

    # # 调用 OpenAI 兼容的多模态 API
    # messages = [{"role": "user", "content": content}]
    #
    # response = client.chat.completions.create(
    #     model="gpt-5-mini",
    #     messages=messages,
    #     temperature=0.2
    # )
    #
    # answer = response.choices[0].message.content
    #
    # analysis = answer.split("explanation")[1].split("characteristics")[0].strip()
    # feature = answer.split("characteristics")[1].split("suggestions")[0].strip()
    # suggestion = answer.split("suggestions")[1].strip()

    results = []
    for _ in range(num):
        messages = [{"role": "user", "content": content}]
        response = client.chat.completions.create(
            model=gpt_model,
            messages=messages,
            temperature=0.2
        )
        answer = response.choices[0].message.content

        try:
            analysis = answer.split("explanation")[1].split("characteristics")[0].strip()
            feature = answer.split("characteristics")[1].split("suggestions")[0].strip()
            suggestion = answer.split("suggestions")[1].strip()
            results.append((feature, analysis, suggestion))
        except (IndexError, AttributeError) as e:
            # 如果解析失败，可选择跳过或抛出错误
            raise ValueError(f"Failed to parse response: {answer}") from e

    return results

    # return feature, analysis, suggestion


def modify_correct(feature, analysis, suggestion, true_label, misclassified_image_b64):
    initial_prompt = load_prompt(temp=False)
    current_prompt = initial_prompt[true_label]
    print("CURRENT PROMPT:", current_prompt)
    content = [
        {
            "type": "text",
            "text": (
                f"You are an expert in time series analysis and prompt design for zero-shot classification."
                f"The dataset is called \"{dataset_name}\".\n"
                f"{subset_description}\n"
                f"### Task Description\n"
                f"A image belongs to class {true_label}, but the model makes a wrong prediction since insufficient "
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
                f"- Add or modify statements on the description. DO NOT DELETE any feature, but you can modify or add "
                f"on it.\n"
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

    messages = [{"role": "user", "content": content}]

    response = client.chat.completions.create(
        model=gpt_model,
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
    with open(f"log/{dataset_name}/feature_list_v2/{dataset_name}_{true_label}_{gpt_model}_3round_temp.json", "w",
              encoding="utf-8") as f:
        json.dump(feature_dict, f, ensure_ascii=False, indent=2)
    return feature_dict


def modify_wrong(feature, analysis, suggestion, wrong_label, misclassified_image_b64):
    initial_prompt = load_prompt(temp=False)
    current_prompt = initial_prompt[wrong_label]
    print("CURRENT PROMPT:", current_prompt)
    content = [
        {
            "type": "text",
            "text": (
                f"You are an expert in time series analysis and prompt design for zero-shot classification."
                f"The dataset is called \"{dataset_name}\".\n"
                f"{subset_description}\n"
                f"### Task Description\n"
                f"The model makes a wrong prediction since insufficient description."
                "You need to improve the wrong description so that it "
                f"DO NOT match the misclassified time series\n"
                f"### Wrong Description:\n"
                f'"{current_prompt}"\n\n'""
                f"### The reasons for the model's incorrect classification :\n"
                f"{analysis}\n\n"
                f"### vague/overemphasized features in wrong description:\n"
                f"{feature}\n\n"
                f"### Add/Modify Suggestions :\n"
                f"{suggestion}\n\n"
                f"### Requirements:\n"
                # "- Analyze the features of the image.\n"
                # "and check whether Missing Features they really exist in the image before modification"
                f"- Add or modify statements on the wrong description. DO NOT DELETE any feature, "
                "but you can modify or add on it.\n"
                f"- Incorporate the vague/overemphasized features above.\n"
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

    messages = [{"role": "user", "content": content}]

    response = client.chat.completions.create(
        model="deepseek-r1",
        messages=messages,
        temperature=0.2
    )
    answer = response.choices[0].message.content

    improved = answer.strip()

    print("✨ Improved prompt for class", wrong_label, ":", improved)
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
    with open(f"log/{dataset_name}/feature_list_v2/{dataset_name}_{wrong_label}_{gpt_model}_3round_temp.json", "w",
              encoding="utf-8") as f:
        json.dump(feature_dict, f, ensure_ascii=False, indent=2)


def load_prompt(temp, dataset=dataset_name):
    root_dir = f"plots/{dataset}_train_nolabel"
    class_dirs = sorted([d for d in os.listdir(root_dir) if d.startswith("class")])
    class_labels = [int(cls[5:]) for cls in class_dirs]
    class_feature = []

    for n in class_labels:
        # feature = f"\n** The common feature of class {n}: **\n"
        feature = ""
        if not temp:
            file_path = f"log/{dataset}/feature_list_v2/{dataset}_{n}_{gpt_model}_3round_features.json"
        else:
            file_path = f"log/{dataset}/feature_list_v2/{dataset}_{n}_{gpt_model}_3round_temp.json"
        with open(file_path, 'r', encoding='utf-8') as file:
            features_dict = json.load(file)

        # 提取 common feature 和 part feature 的键
        common_feats = [k for k, v in features_dict.items() if v == "common feature"]
        part_feats = [k for k, v in features_dict.items() if v == "partial-sample feature"]

        # 格式化为字符串（每项一行，缩进两个空格）
        common_str = "\n".join(f"  - {feat}" for feat in common_feats) if common_feats else "  None"
        part_str = "\n".join(f"  - {feat}" for feat in part_feats) if part_feats else "  None"
        feature_text = (
            f"** The feature that appear at most sample of class {n}(common feature):\n{common_str}\n"
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


def overwrite(n):
    with open(f"log/{dataset_name}/feature_list_v2/{dataset_name}_{n}_{gpt_model}_3round_temp.json", "r",
              encoding="utf-8") as f_source:
        source_data = json.load(f_source)  # 加载为字典

    # 2. 覆盖写入目标JSON（清空原有内容，写入新内容）
    with open(f"log/{dataset_name}/feature_list_v2/{dataset_name}_{n}_{gpt_model}_3round_features.json", "w",
              encoding="utf-8") as f_target:
        json.dump(source_data, f_target, ensure_ascii=False, indent=2)


def overwrite_back(c):
    for n in range(c):
        with open(f"log/{dataset_name}/feature_list_v2/{dataset_name}_{n}_{gpt_model}_3round_features.json", "r",
                  encoding="utf-8") as f_source:
            source_data = json.load(f_source)  # 加载为字典

        # 2. 覆盖写入目标JSON（清空原有内容，写入新内容）
        with open(f"log/{dataset_name}/feature_list_v2/{dataset_name}_{n}_{gpt_model}_3round_temp.json", "w",
                  encoding="utf-8") as f_target:
            json.dump(source_data, f_target, ensure_ascii=False, indent=2)


def get_select(folder_path):
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
    return selected_indices


if __name__ == "__main__":
    gpt_model = "gpt-5-mini"
    test_class = 2
    test_num = 10

    data_train_tensor, labels_train_tensor, _ = load_data(dataset_name)['train']
    folder_path = Path(f"plots/{dataset_name}_train_nolabel")
    exist_indice = get_select(folder_path)
    random_indices = random_choice(labels_train_tensor, exist_indice, test_num)

    overwrite_back(categories)

    correct_num, wrong_indices, wrong_choice, wrong_reason = test(random_indices[test_class], test_class, False)
    if correct_num < test_num:
        improve(max_num=correct_num, wrong_indices=wrong_indices, wrong_choice=wrong_choice,
                wrong_reason=wrong_reason, all_indices=random_indices[test_class], max_iterations=3,
                test_class=test_class, num=1)
    else:
        overwrite(test_class)
        print("no modify")
