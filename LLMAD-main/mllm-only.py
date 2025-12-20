import base64
import os
import time
from matplotlib import pyplot as plt
import json
import re
from dataloader_v2 import load_data
import torch
import random
from openai import OpenAI

OPENAI_API_KEY = "sk-J3azhkbBoUT2YQC8Sl2KLsqBnKC5LcamDvTxWco3ZglWdcgJ"
client = OpenAI(api_key=OPENAI_API_KEY, base_url="https://api.chatanywhere.tech/v1")
dataset_name = "EOGHorizontalSignal"
gpt_model = "gpt-5-mini"
output_file = f"./fewshot-gpt/{dataset_name}.json"

first_round_description = ("You will be provided with two time series samples from each category of this dataset. Your "
                           "first task is to analyze and compare the significant pattern differences across these "
                           "categories.")
first_round_task = ("### Analysis Task\nCompare and summarize the significant differences in the time series patterns "
                    "across categories. Explicitly state if no differences are "
                    "observed. Break the series into meaningful segments (e.g., early, middle, "
                    "late) if applicable.")
third_round_description = ("### Task Description:\nYour second task is to perform "
                           "the time series classification task on the new data sample. You will use your updated "
                           "analysis of time series patterns to make a final "
                           "classification decision.\n"
                           "### Time Series :\n")
third_round_task = ("\n**Answer Format**:\n"
                    "- **True Label**: [Your Final Classification Result]\n- **Confidence**: [Your "
                    "Classification Confidence 0.XX]\n - **Reason**: [reasoning process]")
train_data_tensor, train_labels_tensor, _ = load_data(data_name=dataset_name)['train']
test_data_tensor, test_labels_tensor, _ = load_data(data_name=dataset_name)['test']

MAX_TRAIN_PER_CLASS = 2
MAX_TEST_PER_CLASS = 20




def sample_per_class(data_tensor, labels_tensor, max_per_class, seed=None):
    if seed is not None:
        random.seed(seed)  # 可选：设置随机种子以保证可复现性

    # 第一步：收集每个类别的所有索引
    class_to_indices = {}
    for idx, label in enumerate(labels_tensor):
        label_val = label.item()
        if label_val not in class_to_indices:
            class_to_indices[label_val] = []
        class_to_indices[label_val].append(idx)

    # 第二步：对每个类别随机采样最多 max_per_class 个索引
    sampled_indices = []
    for label_val, indices in class_to_indices.items():
        # 随机打乱或直接采样
        k = min(max_per_class, len(indices))
        sampled = random.sample(indices, k)  # 无放回随机采样
        sampled_indices.extend(sampled)

    # 注意：sampled_indices 是随机顺序的
    # 如果你希望保持原始数据中的大致顺序（可选），可以最后排序：
    # sampled_indices.sort()

    return sampled_indices, class_to_indices


# === 构建子采样后的训练集和测试集索引 ===
filtered_train_indices, _ = sample_per_class(
    train_data_tensor, train_labels_tensor, MAX_TRAIN_PER_CLASS
)
filtered_test_indices, test_class_dict = sample_per_class(
    test_data_tensor, test_labels_tensor, MAX_TEST_PER_CLASS
)

print(f"Train: original={len(train_data_tensor)}, filtered={len(filtered_train_indices)}")
print(f"Test:  original={len(test_data_tensor)}, filtered={len(filtered_test_indices)}")


def gpt_chat(content, conversation):
    max_retries = 3
    retry_count = 0
    messages = [
        {"role": "user", "content": content},  # ❌ 错在这里
    ]
    while retry_count < max_retries:
        try:
            response = client.chat.completions.create(
                model=gpt_model,
                temperature=0.2,
                messages=conversation + messages
            )
            return response.choices[0].message.content

        except Exception as e:
            print(f"API请求失败 (尝试 {retry_count + 1}/{max_retries}): {e}")
            retry_count += 1
            if retry_count < max_retries:
                time.sleep(5)

    print("已达到最大重试次数，请求失败。")
    return None


def get_round1_prompt():
    subset_description = (
        "The dataset are electrooculography signal (EOG). EOG is measurements of the electrical potential between "
        "electrodes placed at points close to the eyes. The EOG recording device is BlueGain, a commercial "
        "biomedical amplifier. The sampling rate was 1.0KHz. This "
        "dataset includes 6 participants eye-writing 7 types of Japanese Katakana strokes. There are 7 classes.")
    task_description = subset_description + " " + first_round_description

    categories = 7
    length = 1250
    dataset_details = "### Dataset Details:\n- **Categories**: " + str(categories) + "\n- **Sequence Length**: " + str(
        length) + " time points"

    time_series_samples = "### Time Series Samples (3 Samples per Category):\n"

    round1_begin = task_description + '\n' + dataset_details + '\n' + time_series_samples

    round1_prompt = [{"type": "text", "text": round1_begin}]

    unique_labels = torch.unique(train_labels_tensor).tolist()
    for label in [0, 2, 3, 5, 6, 7, 8]:
        round1_prompt.append({"type": "text", "text": f"The 3 images of class {label} are as below:\n "})
        for idx in filtered_train_indices:
            if train_labels_tensor[idx] == label:
                round1_prompt.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{plot_series_to_base64(idx, label, train_data_tensor)}"}
                })

    round1_prompt.append({"type": "text", "text": first_round_task})
    return round1_prompt


def plot_series_to_base64(idx, test_class, data):
    # 构造文件路径
    filename = f"mllm/{dataset_name}/class{int(test_class)}/{idx}.png"

    # 确保目录存在（避免保存时出错）
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # 如果文件已存在，直接读取
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            b64 = base64.b64encode(f.read()).decode('utf-8')
        return b64

    # 否则，绘制图像
    data = data[idx]

    plt.figure(figsize=(8, 4))
    if data.ndim == 1:
        plt.plot(data)
    else:
        for c in range(data.shape[0]):
            plt.plot(data[c], label=f"Channel {c}")
        plt.legend()

    plt.xlabel("Time Step")
    plt.ylabel("Value", rotation=0, labelpad=20)
    plt.ylim(-3, 4)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # 保存图像
    plt.savefig(filename, dpi=150)
    plt.close()

    # 读取并返回 base64
    with open(filename, "rb") as f:
        b64 = base64.b64encode(f.read()).decode('utf-8')

    return b64


def round_1(conversation, round1_prompt):
    round1_answer = gpt_chat(round1_prompt, conversation)
    conversation.append({"role": "user", "content": round1_prompt})
    conversation.append({"role": "assistant", "content": round1_answer})

    return conversation


def round_3(conversation, round3_prompt):
    round3_answer = gpt_chat(round3_prompt, conversation)
    conversation.append({"role": "user", "content": round3_prompt})
    conversation.append({"role": "assistant", "content": round3_answer})

    return conversation, round3_answer


# === KNN 分类（1-NN）===
correct = 0
count = 0


def main():
    count = 0
    correct = 0
    round1_prompt = get_round1_prompt()
    root_dir = f"plots/{dataset_name}_test_nolabel"
    class_dirs = sorted([d for d in os.listdir(root_dir) if d.startswith("class")])
    class_labels = [int(cls[5:]) for cls in class_dirs]
    class_to_label = {cls: cls[5:] for cls in class_dirs}
    with open(output_file, 'w') as outf:
        outf.write("[\n")

        # for class_dir in class_dirs:
        #     true_label = int(class_to_label[class_dir])
        #     class_path = os.path.join(root_dir, class_dir)
        #     image_files = sorted(
        #         [f for f in os.listdir(class_path) if f.endswith(".png")],
        #         key=lambda x: int(x.split(".")[0])
        #     )
        #
        #     for img_file in image_files:
        #         conversation = []
        #         conversation = round_1(conversation, round1_prompt)
        #
        #         idx = int(img_file.split(".")[0])
        #
        #         test_data, test_labels, test_index = load_data(data_name=dataset_name)['test']
        #
        #         with open(class_path + "/" + img_file, "rb") as f:
        #             b64 = base64.b64encode(f.read()).decode('utf-8')
        #
        #         round3_prompt = []
        #         round3_prompt.append({"type": "text", "text": third_round_description})
        #         round3_prompt.append({
        #             "type": "image_url",
        #             "image_url": {"url": f"data:image/png;base64,{b64}"}
        #         })
        #         round3_prompt.append({"type": "text", "text": third_round_task})
        #
        #         conversation, answer = round_3(conversation, round3_prompt)
        #         # print(answer)
        #         match = re.search(r'True Label[^0-9]*([0-9]+)', answer)
        #         if match:
        #             label_number = match.group(1)
        #             print(label_number)  # 输出: 1
        #         else:
        #             label_number = -1
        #         output_data = {
        #             "idx": idx,
        #             "label": true_label,
        #             "predicted": label_number,
        #             "conversation": conversation
        #         }
        #         if true_label == label_number:
        #             correct += 1
        #         count += 1
        #
        #         json.dump(output_data, outf, indent=4)
        #         outf.write(",\n")
        #
        #         print(f"sample {idx} processed")
        for test_idx in filtered_test_indices:  # 只遍历采样后的测试样本\
            if test_labels_tensor[test_idx] not in [0, 2, 3, 5, 6, 7, 8]:
                continue
            conversation = []
            conversation = round_1(conversation, round1_prompt)

            round3_prompt = []
            round3_prompt.append({"type": "text", "text": third_round_description})
            round3_prompt.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{plot_series_to_base64(test_idx, test_labels_tensor[test_idx], test_data_tensor)}"}
            })
            round3_prompt.append({"type": "text", "text": third_round_task})
            print(test_idx, test_labels_tensor[test_idx])
            conversation, answer = round_3(conversation, round3_prompt)
            match = re.search(r'True Label[^0-9]*([0-9]+)', answer)
            if match:
                label_number = int(match.group(1))
                print(label_number)  # 输出: 1
            else:
                label_number = -1
            output_data = {
                "idx": test_idx,
                "label": test_labels_tensor[test_idx].item(),
                "predicted": label_number,
                "conversation": answer
            }
            if test_labels_tensor[test_idx] == label_number:
                correct += 1
            count += 1

            json.dump(output_data, outf, indent=4)
            outf.write(",\n")

            print(f"Correct: {correct}")
            print(f"Total: {count}")


if __name__ == "__main__":
    main()
