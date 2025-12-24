import os
import base64
import re

import pandas as pd
from openai import OpenAI
from sklearn.metrics import accuracy_score, classification_report
import time
import numpy as np
from fastdtw import fastdtw
from round2_v5 import load_prompt

# -----------------------------
# 配置
# -----------------------------
gpt_model = "gpt-5-nano"
OPENAI_API_KEY = "sk-J3azhkbBoUT2YQC8Sl2KLsqBnKC5LcamDvTxWco3ZglWdcgJ"
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://api.chatanywhere.tech/v1"
)
dataset_name = "Computers"  # BME
root_dir = f"plots/{dataset_name}_test"
class_dirs = sorted([d for d in os.listdir(root_dir) if d.startswith("class")])
class_to_label = {f"class{i}": i for i in range(len(class_dirs))}

num_classes = 2
valid_labels = list(range(num_classes))
valid_label_str = ", ".join([str(l) for l in valid_labels])
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

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def plot_series_to_base64(idx, test_class):
    with open(f"plots/{dataset_name}_train/class{int(test_class)}/{idx}.png", "rb") as f:
        b64 = base64.b64encode(f.read()).decode('utf-8')

    return b64


def predict():
    R1_PROMPT = ("###Task Description\n"
                 "You are an expert in time series analysis. "
                 "You are given a time series classification task with the Computers dataset"
                 f"\n{subset_description}\n"
                 "your task is to perform the time series classification task on the new data sample."
                 "You will use your analysis of time series plot patterns, the dataset description, and a textual "
                 "description of each category.\n"
                 f"** Class 0 description **: {class_feature[0]}\n"
                 f"** Class 1 description **: {class_feature[1]}\n"
                 f"** Class 2 description **: {class_feature[2]}\n"
                 "### Classification Task\n"
                 "Please think step by step:\n"
                 "– Analyze the Time Series Pattern: [Focus on similarity in periodicity, shape, spikes, "
                 "smoothness, oscillations, amplitude, rate of change, etc.]\n"
                 # "– Make a Preliminary Prediction: [Based on your analysis of the time series pattern and the "
                 # "dataset description, make an initial classification decision.]\n"
                 "- Compare the image patterns with each type of description and observe whether the patterns are "
                 "more compatible with a certain class of description.\n"
                 # "– Review Alternative Classifications: [Consider if there are any other plausible categories that "
                 # "could fit the observed time series pattern.Evaluate the strengths and weaknesses of these "
                 # "alternative classifications compared to your initial prediction.]\n"
                 )

    all_preds = []
    all_labels = []

    for class_dir in class_dirs:
        true_label = class_to_label[class_dir]
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
                conversation.append({"role": "system", "content": R1_PROMPT})
                conversation.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "The images to be classified are as follows:\n"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        },
                        # {"type": "text", "text": "### Answer Format:\n"
                        #                          "-- Classification --:(class 0 / class 1)\n"
                        #                          "-- Alternative Classification --:(could be none)\n"
                        #                          "-- Explanation --\n"},
                    ]
                })
                response = client.chat.completions.create(
                    model="gpt-4o",  # 或 "gpt-4o-mini" 更便宜
                    messages=conversation,
                    temperature=0.2
                )

                pred_text1 = response.choices[0].message.content.strip()
                conversation.append({"role": "assistant", "content": pred_text1})

                top5_indices, top5_labels = get_neighbor(idx)
                neighbor_imgs = []
                img1_b64 = plot_series_to_base64(top5_indices[0], top5_labels[0])
                neighbor_imgs.append(img1_b64)
                img2_b64 = plot_series_to_base64(top5_indices[1], top5_labels[1])
                neighbor_imgs.append(img2_b64)
                img3_b64 = plot_series_to_base64(top5_indices[2], top5_labels[2])
                neighbor_imgs.append(img3_b64)
                img4_b64 = plot_series_to_base64(top5_indices[3], top5_labels[3])
                neighbor_imgs.append(img4_b64)
                img5_b64 = plot_series_to_base64(top5_indices[4], top5_labels[4])
                neighbor_imgs.append(img5_b64)

                R2_PROMPT = [{"type": "text",
                              "text": "### Task Description\n"
                                      "You are an expert in time series analysis. "
                                      "You are given a time series classification task with the Computers dataset"
                                      f"\n{subset_description}\n"
                                      "Your task is to perform the time series classification task on the new data "
                                      "sample.\n"
                                      "### Instruction\n"
                                      "We will give another 3 image similar to the sample to help you classify"
                                      "Focus on similarity in shape, spikes, oscillations, and recovery patterns. "
                                      "Combined with tha analysis of dataset background, the da"},

                             {
                                 "type": "text",
                                 "text": f"3 most similar samples (The class is {top5_labels[0]} {top5_labels[1]} {top5_labels[2]}):"
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
                             # {
                             #     "type": "image_url",
                             #     "image_url": {"url": f"data:image/png;base64,{img4_b64}"}
                             # },
                             # {
                             #     "type": "image_url",
                             #     "image_url": {"url": f"data:image/png;base64,{img4_b64}"}
                             # },
                             {
                                 "type": "text", "text": "### Answer in json Format:\n"
                                                         "{\n"
                                                         "   \"classification\":[one of class],\n"
                                                         "   \"confidence\":[0.0 to 1.0],\n"
                                                         "   \"explanation\":\n"
                                                         "}"
                             }]
                conversation.append({"role": "user", "content": R2_PROMPT})
                # print(conversation)
                response = client.chat.completions.create(
                    model="gpt-4o",  # 或 "gpt-4o-mini" 更便宜
                    messages=conversation,
                    temperature=0.2
                )

                pred_text2 = response.choices[0].message.content.strip()
                conversation.append({"role": "assistant", "content": pred_text2})

                # 尝试解析为整数
                try:
                    pred = pred_text2.split("classification")[1].split("confidence")[0]
                    numbers = re.findall(r'\d+', pred)
                    pred = int(numbers[-1])
                    if pred in valid_labels:
                        all_preds.append(pred)
                    else:
                        print(f"⚠️ Invalid prediction: {pred_text2} (out of range), fallback to 0")
                        all_preds.append(0)
                except ValueError:
                    print(f"⚠️ Failed to parse prediction: '{pred_text2}', fallback to 0")
                    all_preds.append(0)

                all_labels.append(true_label)
                print(f"[{idx}] True: {true_label}, Pred: {all_preds[-1]}")

                # 避免 API 限速（可选）
                time.sleep(0.5)

            except Exception as e:
                print(f"❌ Error on {img_path}: {e}")
                # 可选择跳过或填默认值
                all_preds.append(0)
                all_labels.append(true_label)

    return all_preds, all_labels


def get_neighbor(current_data_index):
    # 加载测试集（其实只用到了 current_data_index 对应的序列）
    test_file_path = f'data/{dataset_name}/{dataset_name}_TEST.txt'
    df_test = pd.read_csv(test_file_path, header=None, sep='\s+')
    test_data = df_test.iloc[:, 1:].values
    current_series = test_data[current_data_index]

    # 加载训练集
    train_file_path = f'data/{dataset_name}/{dataset_name}_TRAIN.txt'
    df_train = pd.read_csv(train_file_path, header=None, sep='\s+')
    train_labels = df_train.iloc[:, 0].values  # 原始标签（从1开始）
    train_data = df_train.iloc[:, 1:].values

    # 计算 DTW 距离
    distances = []
    for idx in range(len(train_data)):
        series = train_data[idx]
        dist, _ = fastdtw(current_series.flatten(), series.flatten(), dist=lambda u, v: abs(u - v))
        distances.append((dist, idx))

    # 取最近的3个
    distances.sort(key=lambda x: x[0])
    top5_indices = [idx for _, idx in distances[:5]]

    # 获取对应的原始标签（从1开始）
    top5_labels = train_labels[top5_indices].tolist()  # 转为 Python list

    print("Top-3 neighbor indices:", top5_indices)
    print("Top-3 neighbor labels:", top5_labels)

    return top5_indices, top5_labels


if __name__ == "__main__":
    class_feature = load_prompt(temp=False)
    print(class_feature)
    all_preds, all_labels = predict()
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    print(f"\n✅ Total samples: {len(all_labels)}")
    print(f"🎯 GPT-4o Accuracy: {acc:.4f} ({acc * 100:.2f}%)")

    target_names = [f"class{i}" for i in range(num_classes)]
    print("\n📋 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=target_names, digits=4))

    class_accuracies = {}

    print("📊 Per-class Accuracy:")
    all_correct = 0
    all_total = 0
    for cls in range(num_classes):
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
