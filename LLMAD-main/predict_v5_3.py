import json
import os
import base64
import re
from pathlib import Path
from openai import OpenAI
from sklearn.metrics import accuracy_score, classification_report
import time
import numpy as np
from fastdtw import fastdtw

from dataloader_v2 import load_data
from round2_v5 import get_select

# subset_description = ( "This dataset were taken from data recorded as part of government sponsored study called
# Powering the " "Nation. The intention was to collect behavioural data about how consumers use electricity within
# the home " "to help reduce the UK's carbon footprint. The data contains readings from 251 households, sampled in "
# "two-minute intervals over a month. Each series is length 720 (24 hours of readings taken every 2 minutes). "
# "Classes are Desktop(Class 0) and Laptop(Class 1)")

# subset_description = ("The data was collected using a tri-axial
# accelerometer on the dominant wrist " "whilst conducting 4 different activities: SEIZURE MIMICKING(class 0) with
# seating after the mimicked " "seizure, WALKING(class 1) with different paces and" "gestures, RUNNING(class 2) with
# running a 40 meters long corridor, SAWING(class 3) with a saw and " "during 30 seconds." "The sampling frequency
# was 16 Hz. The activities lasted about 13 seconds")

# subset_description = ("These dataset were taken from data"
# "recorded as part of government sponsored study called Powering the Nation. The intention was to collect"
# "behavioural data about how consumers use electricity within the home to help reduce the UK's carbon footprint."
# "The data contains readings from 251 households, sampled in two-minute intervals over a month. Each series is"
# "length 720 (24 hours of readings taken every 2 minutes). There are 7 classes.")

# subset_description = ( "BME (Begin-Middle-End) is a synthetic univariate data set with three classes: one class is
# characterized " "by a small positive bell arising at the initial period (Begin), one does not have any bell (
# Middle), " "one has a positive bell arising at the final period (End)." "All series are constituted by a central
# plate. The central plates may be positive or negative. The " "discriminant is the presence or absence of a positive
# peak, or at the beginning of series or at the end." "Class 0: Begin, Class 1: Middle, Class 2: End")

subset_description = ("The arrowhead data consists of outlines of the images of arrowheads. The shapes of the "
                      "projectile points are converted into a time series using the angle-based method. The classification of "
                      "projectile points is an important topic in anthropology. The classes are based on shape distinctions such as the"
                      "presence and location of a notch in the arrow. The three classes are called Avonlea(class 0), Clovis(class 1) "
                      "and Mix(class 2)")

# subset_description = ( "The dataset are electrooculography signal (EOG). EOG is measurements
# of the electrical potential between " "electrodes placed at points close to the eyes. The EOG recording device is
# BlueGain, a commercial " "biomedical amplifier. The sampling rate was 1.0KHz. This " "dataset includes 6
# participants eye-writing 7 types of Japanese Katakana strokes. There are 7 classes.")

# subset_description = ("The
# dataset has 3 classes. Data from each class is standard normal noise plus an offset term which differs " "for each
# class.")

# subset_description = ("It is a synthetic dataset designed to simulate instrumentation failures in a"
# " nuclear power plant, with 4 classes.")

# subset_description = ( "This is a dataset with phase-aligned starlight
# curves of length 1,024, whose class has been determined by " "an expert.There are 3 classes.")

# subset_description = ("The two classes are a normal heartbeat(class 0) and a Myocardial Infarction(class 1). The
# electrocardiogram of a normal heartbeat and that of a myocardial infarction are quite different,
# mainly reflected in the " "following core features:\n " "Normal heartbeat: The ST segment is straight and coincides
# with the baseline. T waves are usually upright " "and symmetrical.\n" "Myocardial infarction: The ST segment will
# show obvious abnormalities. The most common two situations are: " "one is the arched back upward type elevation,
# and the other is significant depression. The T wave becomes " "abnormally sharp (acute phase) or symmetrically
# inverted (necrotic phase).")

dataset_name = "ArrowHead"
OPENAI_API_KEY = "sk-BgMrhX3ZCx1TOeUFhOXIcu8iJfz8ahXbWR56zFQPC6pdTj0k"
client = OpenAI(api_key=OPENAI_API_KEY,
                # base_url="https://api.chatanywhere.tech/v1",
                base_url="https://api.chatanywhere.org/v1"
                )
root_dir = f"plots/{dataset_name}_test_nolabel"
class_dirs = sorted([d for d in os.listdir(root_dir) if d.startswith("class")])
class_labels = [int(cls[5:]) for cls in class_dirs]
class_to_label = {cls: cls[5:] for cls in class_dirs}


def load_prompt(dataset=dataset_name):
    root_dir = f"plots/{dataset}_train_nolabel"
    class_dirs = sorted([d for d in os.listdir(root_dir) if d.startswith("class")])
    class_labels = [int(cls[5:]) for cls in class_dirs]
    class_feature = []

    for n in class_labels:
        file_path = f"log/{dataset}/feature_list/{dataset}_{n}_{gpt_model}_features.json"
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
        class_feature.append(feature_text)
    return class_feature


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def plot_series_to_base64(idx, test_class):
    with open(f"plots/{dataset_name}_train_nolabel/class{int(test_class)}/{idx}.png", "rb") as f:
        b64 = base64.b64encode(f.read()).decode('utf-8')

    return b64


def predict():
    # R1_PROMPT = ("###Task Description\n"
    #              "You are an expert in time series analysis. "
    #              "You are given a time series classification task with the Computers dataset"
    #              f"\n{subset_description}\n"
    #              "your task is to perform the time series classification task on the new data sample."
    #              "You will use your analysis of time series plot patterns, the dataset description, and a textual "
    #              "description of each category.\n"
    #              f"** Class 0 description **: {class_feature[0]}\n"
    #              f"** Class 1 description **: {class_feature[1]}\n"
    #              f"** Class 2 description **: {class_feature[2]}\n"
    #              "### Classification Task\n"
    #              "Please think step by step:\n"
    #              "– Analyze the Time Series Pattern: [Focus on similarity in periodicity, shape, spikes, "
    #              "smoothness, oscillations, amplitude, rate of change, etc.]\n"
    #              # "– Make a Preliminary Prediction: [Based on your analysis of the time series pattern and the "
    #              # "dataset description, make an initial classification decision.]\n"
    #              "- Compare the image patterns with each type of description and observe whether the patterns are "
    #              "more compatible with a certain class of description.\n"
    #              # "– Review Alternative Classifications: [Consider if there are any other plausible categories that "
    #              # "could fit the observed time series pattern.Evaluate the strengths and weaknesses of these "
    #              # "alternative classifications compared to your initial prediction.]\n"
    #
    #              )

    all_preds = []
    all_labels = []
    neighbor = []
    reasons = []
    for class_dir in class_dirs:
        true_label = int(class_to_label[class_dir])
        class_path = os.path.join(root_dir, class_dir)
        image_files = sorted(
            [f for f in os.listdir(class_path) if f.endswith(".png")],
            key=lambda x: int(x.split(".")[0])
        )

        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)
            idx = int(img_file.split(".")[0])
            base64_image = encode_image(img_path)
            conversation = []
            try:
                # conversation.append({"role": "system", "content": R1_PROMPT})
                # conversation.append({
                #     "role": "user",
                #     "content": [
                #         {"type": "text", "text": "The images to be classified are as follows:\n"},
                #         {
                #             "type": "image_url",
                #             "image_url": {
                #                 "url": f"data:image/png;base64,{base64_image}"
                #             }
                #         },
                #         # {"type": "text", "text": "### Answer Format:\n"
                #         #                          "-- Classification --:(class 0 / class 1)\n"
                #         #                          "-- Alternative Classification --:(could be none)\n"
                #         #                          "-- Explanation --\n"},
                #     ]
                # })
                # response = client.chat.completions.create(
                #     model="gpt-4o",  # 或 "gpt-4o-mini" 更便宜
                #     messages=conversation,
                #     temperature=0.2
                # )
                #
                # pred_text1 = response.choices[0].message.content.strip()
                # conversation.append({"role": "assistant", "content": pred_text1})

                top5_indices, top5_labels = get_neighbor(idx)
                neighbor.append(top5_labels)
                neighbor_imgs = []
                img1_b64 = plot_series_to_base64(top5_indices[0], top5_labels[0])
                neighbor_imgs.append(img1_b64)
                img2_b64 = plot_series_to_base64(top5_indices[1], top5_labels[1])
                neighbor_imgs.append(img2_b64)
                img3_b64 = plot_series_to_base64(top5_indices[2], top5_labels[2])
                neighbor_imgs.append(img3_b64)

                # R1_PROMPT = ("###Task Description\n"
                #              "You are an expert in time series analysis. "
                #              "You are given a time series classification task with the Computers dataset"
                #              f"\n{subset_description}\n"
                #              "your task is to perform the time series classification task on the new data sample."
                #              "You will use your analysis of time series plot patterns, the dataset description, and a textual "
                #              "description of each category.\n"
                #              f"** Class 0 description **: {class_feature[0]}\n"
                #              f"** Class 1 description **: {class_feature[1]}\n"
                #              f"** Class 2 description **: {class_feature[2]}\n"
                #              "### Classification Task\n"
                #              "Please think step by step:\n"
                #              "– Analyze the Time Series Pattern: [Focus on similarity in periodicity, shape, spikes, "
                #              "smoothness, oscillations, amplitude, rate of change, etc.]\n"
                #              "- We will give another 3 images similar to the sample to help you classify. You can "
                #              "compare the time series pattern with these images.\n"
                #              # "– Make a Preliminary Prediction: [Based on your analysis of the time series pattern and the "
                #              # "dataset description, make an initial classification decision.]\n"
                #              "- Compare the image patterns with each type of description and observe whether the patterns are "
                #              "more compatible with a certain class of description.\n"
                #              # "– Review Alternative Classifications: [Consider if there are any other plausible categories that "
                #              # "could fit the observed time series pattern.Evaluate the strengths and weaknesses of these "
                #              # "alternative classifications compared to your initial prediction.]\n"
                #             "- Based on the analysis above, make your final classification decision."
                #              )
                R2_PROMPT = [{"type": "text",
                              "text": "###Task Description\n"
                                      "You are an expert in time series analysis. "
                                      f"You are given a time series classification task with the {dataset_name} dataset"
                                      f"\n{subset_description}\n"
                                      "your task is to perform the time series classification task on the new data "
                                      "sample.\n"
                                      f"** Class 0 description **: {class_feature[0]}\n"
                                      f"** Class 1 description **: {class_feature[1]}\n"
                                      f"** Class 2 description **: {class_feature[2]}\n"
                                      # f"** Class 3 description **: {class_feature[3]}\n"
                                      f"We will also give 3 similar images (the label are respectively class "
                                      f"{top5_labels[0]}, class {top5_labels[1]}, class {top5_labels[2]}) to the sample"
                                      " to help you classify. "
                              # "(But the descriptions is most important)"
                                      "You should use time series plot patterns, the dataset description "
                                      ", the description of each category, the 3 similar images to analyze.\n"
                                      "### Classification Task\n"
                                      "Please think step by step:\n"
                                      "– Analyze the Time Series Pattern: [Focus on similarity in periodicity, shape, "
                                      "spikes, smoothness, oscillations, amplitude, rate of change, etc.]\n"
                                      "- Compare the images to be classified with 3 similar images. Determine whether "
                                      "it is particularly similar to a certain category based on their labels.\n"
                                      "- Compare the image patterns with each type of description and observe whether "
                                      "the patterns are more compatible with a certain class of description.\n"
                                      "- Make a Preliminary Prediction: [Based on your analysis of the time series "
                                      "pattern and the"
                                      "comparison to the description and 3 images, make an initial classification "
                                      "decision.]\n"
                                      "- Reevaluate the rationality of the classification decision:"
                                      "Is the final classification result fully supported by evidence?"
                                      "Are there any other possible classification results?  Why were they excluded?"
                                      "Are there any logical loopholes in the entire reasoning chain? "
                                      "If you find any problems, reanalyze the image.\n"
                                      "- Based on the analysis above, make your final classification decision.\n"},
                             {"type": "text", "text": "The images to be classified are as follows:\n"},
                             {
                                 "type": "image_url",
                                 "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                             },
                             {
                                 "type": "text",
                                 "text": f"3 most similar samples:\n"
                             },
                             {
                                 "type": "image_url",
                                 "image_url": {"url": f"data:image/png;base64,{img1_b64}"}
                             },
                             {
                                 "type": "image_url",
                                 "image_url": {"url": f"data:image/png;base64,{img2_b64}"}
                             },
                             {
                                 "type": "image_url",
                                 "image_url": {"url": f"data:image/png;base64,{img3_b64}"}
                             },
                             {
                                 "type": "text", "text": "### Answer in json Format:\n"
                                                         "{\n"
                                                         "   \"classification\":[one of class],\n"
                                                         "   \"confidence\":[0.0 to 1.0],\n"
                                                         "   \"explanation\":[your step by step reasoning process]\n"
                                                         "}"
                             }]


                conversation.append({"role": "user", "content": R2_PROMPT})
                # print(conversation)
                response = client.chat.completions.create(
                    model=gpt_model,  # gpt-5-mini
                    messages=conversation,
                    temperature=0.2
                )

                pred_text2 = response.choices[0].message.content.strip()
                reasons.append(pred_text2)
                conversation.append({"role": "assistant", "content": pred_text2})

                # 尝试解析为整数
                try:
                    pred = pred_text2.split("classification")[1].split("confidence")[0]
                    numbers = re.findall(r'\d+', pred)
                    pred = int(numbers[-1])
                    all_preds.append(pred)

                except ValueError:
                    print(f"⚠️ Failed to parse prediction: '{pred_text2}', fallback to 0")
                    all_preds.append(0)

                all_labels.append(true_label)
                print(f"[{idx}] True: {true_label}, Pred: {all_preds[-1]}")



            except Exception as e:
                print(f"❌ Error on {img_path}: {e}")
                # 可选择跳过或填默认值
                all_preds.append(0)
                all_labels.append(true_label)

    return all_preds, all_labels, neighbor, reasons


def get_neighbor(current_data_index):
    train_data, train_labels, train_index = load_data(data_name=dataset_name)['train']
    test_data, test_labels, test_index = load_data(data_name=dataset_name)['test']

    pos = np.where(test_index == current_data_index)[0][0]
    current_series = test_data[pos]

    folder_path = Path(f"plots/{dataset_name}_train_nolabel")
    exist_indice = get_select(folder_path)

    # 计算 DTW 距离
    distances = []
    for idx in exist_indice:
        series = train_data[idx]
        dist, _ = fastdtw(current_series.flatten(), series.flatten(), dist=lambda u, v: abs(u - v))
        distances.append((dist, idx))

    # 取最近的3个
    distances.sort(key=lambda x: x[0])
    # 提取前5个的原始局部索引
    top3_indices = [idx for _, idx in distances[:5]]

    top3_labels = []
    for idx in top3_indices:
        pos = train_labels[idx]
        top3_labels.append(pos.item())

    print("Top-3 neighbor indices:", top3_indices)
    print("Top-3 neighbor labels:", top3_labels)

    return top3_indices, top3_labels

    # # 存储每个类别最近的样本距离和索引
    # class_nearest = {}
    #
    # for idx in exist_indice:
    #     series = train_data[idx]
    #     # 计算当前序列与训练样本的距离
    #     dist, _ = fastdtw(current_series.flatten(), series.flatten(), dist=lambda u, v: abs(u - v))
    #     label = train_labels[idx].item()  # 获取当前样本的标签
    #
    #     # 如果是该类别首次出现，或当前距离更近，则更新该类别最近样本
    #     if label not in class_nearest or dist < class_nearest[label][0]:
    #         class_nearest[label] = (dist, idx)
    #
    # # 将每个类别最近的样本转换为列表并按距离排序
    # sorted_class_nearest = sorted(class_nearest.values(), key=lambda x: x[0])
    #
    # # 取距离最近的3个不同类别的样本索引
    # top3_indices = [item[1] for item in sorted_class_nearest[:3]]
    # top3_labels = [train_labels[idx].item() for idx in top3_indices]
    #
    # print("Top-3 unique neighbor indices:", top3_indices)
    # print("Top-3 unique neighbor labels:", top3_labels)
    #
    # return top3_indices, top3_labels


if __name__ == "__main__":
    gpt_model = "qwen3-235b-a22b"
    class_feature = load_prompt(dataset=dataset_name)
    print(class_feature)
    all_preds, all_labels, neighbor, reasons = predict()
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    print(f"\n✅ Total samples: {len(all_labels)}")
    print(f"🎯 GPT-4o Accuracy: {acc:.4f} ({acc * 100:.2f}%)")

    target_names = [f"class{i}" for i in class_labels]
    print("\n📋 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=target_names, digits=4))

    result = {
        "predictions": all_preds.tolist(),
        "labels": all_labels.tolist(),
        "dtw": neighbor,
        "accuracy": acc,
        "report": classification_report(all_labels, all_preds, target_names=target_names, digits=4, output_dict=True),
        "reason": reasons
    }

    with open(f"prediction/{dataset_name}_{gpt_model}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    class_accuracies = {}

    print("📊 Per-class Accuracy:")
    all_correct = 0
    all_total = 0
    for cls in class_labels:
        # 找出该类别的所有样本索引
        mask = (all_labels == cls)
        # 计算该类中预测正确的比例
        correct = np.sum(all_preds[mask] == cls)
        all_correct += correct
        total = np.sum(mask)
        all_total += total
        acc = correct / total
        class_accuracies[cls] = acc
        print(f"  Class {cls}: {acc:.4f} ({correct}/{total})")
    all_acc = all_correct / all_total
    print(f"{all_acc:.4f} ({all_correct}/{all_total})")
