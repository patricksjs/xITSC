import openai
from openai import OpenAI
import json
import time
import argparse
import os
import base64
import re
import numpy as np
import torch
from sklearn.cluster import KMeans
from fastdtw import fastdtw
from collections import Counter
from typing import Dict, List, Any, Optional
import random

# 导入原来的prompts
from prompt import prompt11, prompt12, prompt21, prompt22, prompt30, prompt31, prompt32, prompt41, prompt42


DATASET_PROMPT_TEMPLATE = '''
- Background: The dataset traces the electrical activity recorded during one heartbeat.
<explanation>
a myocardial infarction event means a heart attack due to prolonged cardiac ischemia
– Categories: [2]
– Sequence Length: [96] time points
– label 0: {myocardial infarction event}
– label 1: {normal heartbeat}
When conducting the analysis, each feature event can be interpreted by combining it with the background information of the dataset.
'''
api_key = "sk-Aqjpx18VnGq62P86z94ZM6EiIWbaiiFWy3o3dyEUXN2LaAJU"
OPENAI_BASE_URL = "https://api.chatanywhere.tech/v1"
client = OpenAI(
    api_key=api_key,
    base_url=OPENAI_BASE_URL
)


def parse_arguments():
    parser = argparse.ArgumentParser(description='xITSC with Images')
    parser.add_argument('--config_file', type=str, required=True, help='Path to config JSON file')
    parser.add_argument('--gpt_model', type=str, required=True, help='GPT model to use')
    parser.add_argument('--output_file', type=str, required=True, help='Path to output JSON file')
    parser.add_argument('--image_root', type=str, default=r'D:\zuhui\xITSC\data\image\train',
                        help='Root directory for reference images')
    parser.add_argument('--test_image_root', type=str, default=r'D:\zuhui\xITSC\data\image\test',
                        help='Root directory for test images')
    parser.add_argument('--model_results_file', type=str, default=r'D:\zuhui\xITSC\ECG200_transformer.json',
                        help='Path to model results JSON file')
    parser.add_argument('--samples_per_class', type=int, default=4, help='Number of reference samples per class')
    return parser.parse_args()


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)


def encode_image_to_base64(image_path):
    """将图片编码为base64字符串"""
    if not os.path.exists(image_path):
        return None
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_reference_images(dataset_name: str, label: int, sample_indices: List[int], image_root: str):
    """获取指定标签和样本索引的参考图片"""
    images = []
    if isinstance(label, float):
        label = int(label)
    label_dir = os.path.join(image_root, f"{dataset_name}_{label}")

    if not os.path.exists(label_dir):
        label_str = str(label).split('.')[0]  # 移除小数部分
        label_dir = os.path.join(image_root, f"{dataset_name}_{label_str}")
        if not os.path.exists(label_dir):
            print(f"Warning: Directory not found: {label_dir}")
        return images

    for sample_idx in sample_indices:
        sample_images = {}
        # 尝试获取三种类型的图片
        image_types = {
            "line_chart": f"plot_sample{sample_idx}.png",
            "spectrogram": f"STFT_sample{sample_idx}.png",
            "heatmap": f"shap_sample{sample_idx}.png"
        }

        for img_type, filename in image_types.items():
            img_path = os.path.join(label_dir, filename)
            b64_image = encode_image_to_base64(img_path)
            if b64_image:
                sample_images[img_type] = b64_image

        if sample_images and len(sample_images) == 3:  # 如果至少有一种图片存在
            images.append({
                "sample_id": sample_idx,
                "images": sample_images
            })

    return images


def get_test_images(dataset_name: str, label: int, sample_id: int, test_image_root: str):
    """获取测试样本的图片"""
    label_dir = os.path.join(test_image_root, f"{dataset_name}_{label}")

    if not os.path.exists(label_dir):
        return None

    test_images = {}
    image_types = {
        "line_chart": f"plot_sample{sample_id}.png",
        "spectrogram": f"STFT_sample{sample_id}.png",
        "heatmap": f"shap_sample{sample_id}.png"
    }

    for img_type, filename in image_types.items():
        img_path = os.path.join(label_dir, filename)
        b64_image = encode_image_to_base64(img_path)
        if b64_image:
            test_images[img_type] = b64_image

    return test_images


def gpt_chat_with_images(client, model, messages, temperature=0.1):
    """支持图片的GPT对话"""
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API request failed: {e}")
        time.sleep(5)
        return None


def build_round1_messages_correct(dataset_name: str, label: int, samples_data: List[Dict], config: Dict):
    """构建正确样本第一轮对话的消息（包含图片）- 使用prompt11"""
    dataset = DATASET_PROMPT_TEMPLATE

    base_prompt = f"""
<dataset details>
{dataset}

<image information>
The label for which you currently need to perform feature extraction and summarization is: label {label}
Next are the plots, time-frequency images, and heatmaps of {len(samples_data)} reference samples for the label {label} to be summarized:"""

    messages = [
        {
            "role": "user",
            "content": []
        }
    ]

    # 添加文本
    messages[0]["content"].append({
        "type": "text",
        "text": base_prompt
    })

    # 添加所有样本的图片
    for i, sample_data in enumerate(samples_data):
        messages[0]["content"].append({
            "type": "text",
            "text": f"sample {i + 1}/{len(samples_data)}: "
        })

        # 添加三张图像
        for img_type in ["line_chart", "spectrogram", "heatmap"]:
            if img_type in sample_data["images"]:
                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{sample_data['images'][img_type]}"}
                })

    # 添加任务 - 使用prompt11
    messages[0]["content"].append({
        "type": "text",
        "text": prompt11
    })

    return messages


def build_round1_messages_wrong(dataset_name: str, label: int, samples_data: List[Dict],
                                model_info_dict: Dict, config: Dict):
    """构建错误样本第一轮对话的消息（包含图片和模型输出）- 使用prompt12"""
    dataset = DATASET_PROMPT_TEMPLATE

    base_prompt = f"""
<dataset details>
{dataset}

<image information>
The label for which you currently need to perform feature extraction and summarization is: label {label}
Next are the plots, time-frequency images, and heatmaps of {len(samples_data)} reference samples for the label {label} to be summarized.
Each sample includes the black-box model's output information:"""

    messages = [
        {
            "role": "user",
            "content": []
        }
    ]

    # 添加文本
    messages[0]["content"].append({
        "type": "text",
        "text": base_prompt
    })

    # 添加所有样本的图片和模型输出信息
    for i, sample_data in enumerate(samples_data):
        sample_id = sample_data["sample_id"]
        # 获取模型输出信息
        model_info = model_info_dict.get(str(sample_id), {})

        messages[0]["content"].append({
            "type": "text",
            "text": f"sample {i + 1}/{len(samples_data)} (ID: {sample_id}): "
        })

        # 添加三张图像
        for img_type in ["line_chart", "spectrogram", "heatmap"]:
            if img_type in sample_data["images"]:
                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{sample_data['images'][img_type]}"}
                })

        # 添加模型输出信息
        if model_info:
            model_output_str = f"""
Black-box model output for this sample:
- Predicted: {model_info.get('predicted', 'N/A')}
- Logits: {model_info.get('logits', 'N/A')}
- True Label: {model_info.get('true_label', 'N/A')}
"""
            messages[0]["content"].append({
                "type": "text",
                "text": model_output_str
            })

    # 添加任务 - 使用prompt12
    messages[0]["content"].append({
        "type": "text",
        "text": prompt12
    })

    return messages


def build_round2_messages_correct(dataset_name: str, label: int, sample_data: Dict,
                                  round1_summary: Dict, config: Dict):
    """构建正确样本第二轮对话的消息 - 使用prompt21"""
    dataset = DATASET_PROMPT_TEMPLATE

    base_prompt = f"""
<dataset details>
{dataset}

### Preliminary Summary for Label {label}:
{json.dumps(round1_summary.get(f'label_{label}', {}), indent=2)}

### Current Sample to Analyze:
Sample ID: {sample_data['sample_id']}
"""

    messages = [
        {
            "role": "user",
            "content": []
        }
    ]

    messages[0]["content"].append({
        "type": "text",
        "text": base_prompt
    })

    # 添加当前样本的图片
    messages[0]["content"].append({
        "type": "text",
        "text": f"Images for sample {sample_data['sample_id']}:"
    })

    for img_type in ["line_chart", "spectrogram", "heatmap"]:
        if img_type in sample_data["images"]:
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{sample_data['images'][img_type]}"}
            })

    messages[0]["content"].append({
        "type": "text",
        "text": prompt21
    })

    return messages


def build_round2_messages_wrong(dataset_name: str, label: int, sample_data: Dict,
                                round1_summary: Dict, model_info: Dict, config: Dict):
    """构建错误样本第二轮对话的消息（包含模型输出）- 使用prompt22"""
    dataset = DATASET_PROMPT_TEMPLATE

    base_prompt = f"""
<dataset details>
{dataset}

### Preliminary Summary for Label {label}:
{json.dumps(round1_summary.get(f'label_{label}', {}), indent=2)}

### Current Sample to Analyze:
Sample ID: {sample_data['sample_id']}
"""

    messages = [
        {
            "role": "user",
            "content": []
        }
    ]

    messages[0]["content"].append({
        "type": "text",
        "text": base_prompt
    })

    # 添加当前样本的图片
    messages[0]["content"].append({
        "type": "text",
        "text": f"Images for sample {sample_data['sample_id']}:"
    })

    for img_type in ["line_chart", "spectrogram", "heatmap"]:
        if img_type in sample_data["images"]:
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{sample_data['images'][img_type]}"}
            })

    # 添加模型输出信息
    if model_info:
        model_output_str = f"""
Black-box model output for this sample:
- Predicted: {model_info.get('predicted', 'N/A')}
- Logits: {model_info.get('logits', 'N/A')}
- True Label: {model_info.get('true_label', 'N/A')}
"""
        messages[0]["content"].append({
            "type": "text",
            "text": model_output_str
        })

    messages[0]["content"].append({
        "type": "text",
        "text": prompt22
    })

    return messages


def build_round3_messages(dataset_name: str, test_sample: Dict,
                          label_summaries_wrong: Dict, config: Dict):
    """构建第三轮对话的消息 - 使用prompt30"""
    dataset = DATASET_PROMPT_TEMPLATE

    base_prompt = f"""
<dataset details>
{dataset}

### Summary of Features and Patterns for Different Labels (The summary under label x indicates that the model may misclassify these samples as x, even though their true labels are not x.):
{json.dumps(label_summaries_wrong, indent=2)}
"""

    messages = [
        {
            "role": "user",
            "content": []
        }
    ]

    messages[0]["content"].append({
        "type": "text",
        "text": base_prompt
    })

    # 添加测试样本的图片
    messages[0]["content"].append({
        "type": "text",
        "text": f"Test sample {test_sample['sample_id']} images:"
    })

    for img_type in ["line_chart", "spectrogram", "heatmap"]:
        if img_type in test_sample["images"]:
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{test_sample['images'][img_type]}"}
            })

    messages[0]["content"].append({
        "type": "text",
        "text": prompt30
    })

    return messages


def build_round4_1_messages(dataset_name: str, test_sample: Dict,
                            similar_samples: List[Dict], dissimilar_samples: List[Dict],
                            label_summaries_correct: Dict, round3_rationale: str, config: Dict):
    """构建4.1轮对话的消息（没有热力图）- 使用prompt31"""
    dataset = DATASET_PROMPT_TEMPLATE

    base_prompt = f"""
<dataset details>
{dataset}

### Summary of Features and Patterns for Different Labels :
{json.dumps(label_summaries_correct, indent=2)}


"""

    messages = [
        {
            "role": "user",
            "content": []
        }
    ]

    messages[0]["content"].append({
        "type": "text",
        "text": base_prompt
    })

    # 添加测试样本的图片（只有折线图和时频图）
    messages[0]["content"].append({
        "type": "text",
        "text": f"Test sample {test_sample['sample_id']}:"
    })

    for img_type in ["line_chart", "spectrogram"]:
        if img_type in test_sample["images"]:
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{test_sample['images'][img_type]}"}
            })

    # 添加参考样本
    if similar_samples:
        messages[0]["content"].append({
            "type": "text",
            "text": f"\n{len(similar_samples)} similar reference samples:"
        })
        for i, sample in enumerate(similar_samples):
            messages[0]["content"].append({
                "type": "text",
                "text": f"Similar sample {i + 1} (Label {sample['label']}):"
            })
            for img_type in ["line_chart", "spectrogram"]:
                if img_type in sample["images"]:
                    messages[0]["content"].append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{sample['images'][img_type]}"}
                    })

    if dissimilar_samples:
        messages[0]["content"].append({
            "type": "text",
            "text": f"\n{len(dissimilar_samples)} dissimilar reference samples:"
        })
        for i, sample in enumerate(dissimilar_samples):
            messages[0]["content"].append({
                "type": "text",
                "text": f"Dissimilar sample {i + 1} (Label {sample['label']}):"
            })
            for img_type in ["line_chart", "spectrogram"]:
                if img_type in sample["images"]:
                    messages[0]["content"].append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{sample['images'][img_type]}"}
                    })

    messages[0]["content"].append({
        "type": "text",
        "text": prompt31
    })

    return messages


def build_round4_2_messages(dataset_name: str, test_sample: Dict,
                            reference_samples: List[Dict],
                            label_summaries_correct: Dict, config: Dict):
    """构建4.2轮对话的消息（有三类图片）- 使用prompt32"""
    dataset = DATASET_PROMPT_TEMPLATE

    base_prompt = f"""
<Dataset Details>
{dataset}

### Summary of Features and Patterns for Different Labels:
{json.dumps(label_summaries_correct, indent=2)}
"""

    messages = [
        {
            "role": "user",
            "content": []
        }
    ]

    messages[0]["content"].append({
        "type": "text",
        "text": base_prompt
    })

    # 添加测试样本的所有图片
    messages[0]["content"].append({
        "type": "text",
        "text": f"Test sample {test_sample['sample_id']}:"
    })

    for img_type in ["line_chart", "spectrogram", "heatmap"]:
        if img_type in test_sample["images"]:
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{test_sample['images'][img_type]}"}
            })

    # 添加参考样本
    if reference_samples:
        messages[0]["content"].append({
            "type": "text",
            "text": f"\n{len(reference_samples)} reference samples:"
        })
        for i, sample in enumerate(reference_samples):
            messages[0]["content"].append({
                "type": "text",
                "text": f"Reference sample {i + 1} (Label {sample['label']}):"
            })
            for img_type in ["line_chart", "spectrogram"]:
                if img_type in sample["images"]:
                    messages[0]["content"].append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{sample['images'][img_type]}"}
                    })

    messages[0]["content"].append({
        "type": "text",
        "text": prompt32
    })

    return messages


def build_round51_messages(dataset_name: str, test_sample: Dict,
                          label_summaries_correct: Dict, model_info: Dict,
                          round4_results: List[str], round3_rationale: str,
                          config: Dict, use_heatmap: bool = False):
    """构建第五轮对话的消息"""
    dataset = DATASET_PROMPT_TEMPLATE

    # 格式化Accuracy信息
    accuracy_info = model_info.get('Accuracy', {})
    if isinstance(accuracy_info, dict):
        accuracy_str = '\n'.join([f'    {label}: {acc:.4f}' for label, acc in accuracy_info.items()])
    else:
        accuracy_str = str(accuracy_info)

    base_prompt = f"""
<Dataset Details>
{dataset}

### Summary of Features and Patterns for Different Labels:
{json.dumps(label_summaries_correct, indent=2)}



### Classification results of the sample to be classified from three other assistants:
"""

    # 添加前三轮的结果
    for i, result in enumerate(round4_results):
        base_prompt += f"\nRound {i + 1} Analysis:\n{result}\n"

    base_prompt += f"""
### Classification result and prediction probability value of the black-box model:
- Predicted: {model_info['predicted']}
- Logits: {model_info['logits']}
- Classification Accuracy for each label:
{accuracy_str}
"""

    messages = [
        {
            "role": "user",
            "content": []
        }
    ]

    messages[0]["content"].append({
        "type": "text",
        "text": base_prompt
    })

    # 添加测试样本的图片
    messages[0]["content"].append({
        "type": "text",
        "text": f"Test sample {test_sample['sample_id']}:"
    })

    img_types = ["line_chart", "spectrogram", "heatmap"] if use_heatmap else ["line_chart", "spectrogram"]
    for img_type in img_types:
        if img_type in test_sample["images"]:
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{test_sample['images'][img_type]}"}
            })

    if use_heatmap:
        messages[0]["content"].append({
            "type": "text",
            "text": prompt42
        })
    else:
        messages[0]["content"].append({
            "type": "text",
            "text": prompt41
        })

    return messages
def build_round52_messages(dataset_name: str, test_sample: Dict,
                          label_summaries_correct: Dict, model_info: Dict,
                          round4_results: List[str],
                          config: Dict, use_heatmap: bool = False):
    """构建第五轮对话的消息"""
    dataset = DATASET_PROMPT_TEMPLATE

    # 格式化Accuracy信息
    accuracy_info = model_info.get('Accuracy', {})
    if isinstance(accuracy_info, dict):
        accuracy_str = '\n'.join([f'    {label}: {acc:.4f}' for label, acc in accuracy_info.items()])
    else:
        accuracy_str = str(accuracy_info)

    base_prompt = f"""
<Dataset Details>
{dataset}

### Summary of Features and Patterns for Different Labels:
{json.dumps(label_summaries_correct, indent=2)}



### Classification results of the sample to be classified from three other assistants:
"""

    # 添加前三轮的结果
    for i, result in enumerate(round4_results):
        base_prompt += f"\nRound {i + 1} Analysis:\n{result}\n"

    base_prompt += f"""
### Classification result and prediction probability value of the black-box model:
- Predicted: {model_info['predicted']}
- Logits: {model_info['logits']}
- Classification Accuracy for each label:
{accuracy_str}
"""

    messages = [
        {
            "role": "user",
            "content": []
        }
    ]

    messages[0]["content"].append({
        "type": "text",
        "text": base_prompt
    })

    # 添加测试样本的图片
    messages[0]["content"].append({
        "type": "text",
        "text": f"Test sample {test_sample['sample_id']}:"
    })

    img_types = ["line_chart", "spectrogram", "heatmap"] if use_heatmap else ["line_chart", "spectrogram"]
    for img_type in img_types:
        if img_type in test_sample["images"]:
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{test_sample['images'][img_type]}"}
            })

    if use_heatmap:
        messages[0]["content"].append({
            "type": "text",
            "text": prompt42
        })
    else:
        messages[0]["content"].append({
            "type": "text",
            "text": prompt41
        })

    return messages

def extract_result_from_response(response_text: str, pattern_type: str):
    """从响应中提取结果"""
    patterns = {
        'round3': re.compile(r'result(.*?)score', re.IGNORECASE),
        'round4_result': re.compile(r'result(.*?)model', re.IGNORECASE),
        'round4_model': re.compile(r'model(.*?)rationale', re.IGNORECASE),
    }

    if pattern_type not in patterns:
        return None

    match = patterns[pattern_type].search(response_text)
    if match:
        # 提取数字
        text = match.group(1)
        numbers = re.findall(r'\d+', text)
        if numbers:
            return int(numbers[0])

    return None


def parse_json_response(response_text: str):
    """解析JSON格式的响应"""
    try:
        # 尝试直接解析
        return json.loads(response_text)
    except:
        # 尝试提取JSON部分
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass
    return None


def get_available_samples(dataset_name: str, label: int, image_root: str):
    """获取指定标签下所有可用的样本ID"""
    label_dir = os.path.join(image_root, f"{dataset_name}_{label}")
    if not os.path.exists(label_dir):
        return []

    available_samples = []
    # 检查所有可能的样本ID
    for i in range(1000):  # 假设最多1000个样本
        # 检查是否有折线图
        plot_path = os.path.join(label_dir, f"plot_sample{i}.png")
        if os.path.exists(plot_path):
            available_samples.append(i)

    return available_samples


def select_reference_samples(sample_dict: Dict, dataset_name: str,
                             image_root: str, samples_per_class: int):
    """选择参考样本，如果数量不足则随机补充"""
    all_samples = {}

    for label_str, sample_ids in sample_dict.items():
        label = int(label_str)
        available_samples = get_available_samples(dataset_name, label, image_root)

        # 先使用指定的样本ID
        selected_samples = []
        for sample_id in sample_ids:
            if sample_id in available_samples:
                selected_samples.append(sample_id)



        # 如果最终选择的样本数超过samples_per_class，只取前samples_per_class个
        all_samples[label_str] = selected_samples[:samples_per_class]

        print(f"Label {label}: Selected {len(all_samples[label_str])} samples from {image_root}")

    return all_samples


def phase1_round1_correct(client, gpt_model, dataset_name: str, config: Dict, args, model_results_dict: Dict):
    """第一阶段第一轮：为每个标签处理正确参考样本 - 使用prompt11"""
    label_summaries = {}

    sample_dict = config.get("correct_IDS", {})
    image_root = "D:/zuhui/xITSC/data/image/test"
    print("Processing correct samples for round 1...")

    selected_samples = select_reference_samples(sample_dict, dataset_name,
                                                image_root, args.samples_per_class)

    for label_str, sample_ids in selected_samples.items():
        label = int(label_str)
        print(f"Processing correct reference samples for label {label}...")

        # 获取样本图片
        samples_data = []
        for sample_id in sample_ids:
            images = get_reference_images(dataset_name, label, [sample_id], image_root)
            if images:
                samples_data.append({
                    "sample_id": sample_id,
                    "label": label,
                    "images": images[0]["images"] if images else {}
                })

        if not samples_data:
            print(f"No images found for label {label}")
            continue

        # 构建消息并发送 - 使用正确样本的提示词
        messages = build_round1_messages_correct(dataset_name, label, samples_data, config)
        response = gpt_chat_with_images(client, gpt_model, messages)

        if response:
            # 解析响应
            parsed = parse_json_response(response)
            if parsed:
                label_summaries[f"label_{label}"] = parsed
            else:
                # 创建默认结构
                label_summaries[f"label_{label}"] = {
                    "feature": [],
                    "pattern": []
                }

        time.sleep(1)  # 避免API限制

    return label_summaries


def phase1_round1_wrong(client, gpt_model, dataset_name: str, config: Dict, args, model_results_dict: Dict):
    """第一阶段第一轮：为每个标签处理错误参考样本 - 使用prompt12"""
    label_summaries = {}

    sample_dict = config.get("wrong_IDS", {})
    image_root = "D:/zuhui/xITSC/data/image/test"
    print("Processing wrong samples for round 1...")

    selected_samples = select_reference_samples(sample_dict, dataset_name,
                                                image_root, args.samples_per_class)

    for label_str, sample_ids in selected_samples.items():
        label = int(label_str)
        print(f"Processing wrong reference samples for label {label}...")

        # 获取样本图片
        samples_data = []
        for sample_id in sample_ids:
            images = get_reference_images(dataset_name, label, [sample_id], image_root)
            if images:
                samples_data.append({
                    "sample_id": sample_id,
                    "label": label,
                    "images": images[0]["images"] if images else {}
                })

        if not samples_data:
            print(f"No images found for label {label}")
            continue

        # 获取样本的模型输出信息
        sample_models_info = {}
        for sample_data in samples_data:
            sample_id = sample_data["sample_id"]
            model_info = model_results_dict.get(str(sample_id))
            if model_info:
                sample_models_info[str(sample_id)] = model_info

        # 构建消息并发送 - 使用错误样本的提示词，包含模型输出
        messages = build_round1_messages_wrong(dataset_name, label, samples_data,
                                               sample_models_info, config)
        response = gpt_chat_with_images(client, gpt_model, messages)

        if response:
            # 解析响应
            parsed = parse_json_response(response)
            if parsed:
                label_summaries[f"label_{label}"] = parsed
            else:
                # 创建默认结构
                label_summaries[f"label_{label}"] = {
                    "feature": [],
                    "pattern": []
                }

        time.sleep(1)  # 避免API限制

    return label_summaries


def phase1_round2_correct(client, gpt_model, dataset_name: str, config: Dict,
                          round1_summaries: Dict, args, model_results_dict: Dict):
    """第一阶段第二轮：处理正确样本并更新总结 - 使用prompt21"""
    updated_summaries = round1_summaries.copy()

    sample_dict = config.get("correct_IDS", {})
    image_root = "D:/zuhui/xITSC/data/image/test"

    for label_str, sample_ids in sample_dict.items():
        label = int(label_str)

        for sample_id in sample_ids:
            print(f"Processing correct sample {sample_id} for label {label}...")

            # 获取样本图片
            images = get_reference_images(dataset_name, label, [sample_id], image_root)
            if not images:
                continue

            sample_data = {
                "sample_id": sample_id,
                "label": label,
                "images": images[0]["images"]
            }

            # 构建消息并发送 - 使用正确样本的提示词
            messages = build_round2_messages_correct(dataset_name, label, sample_data,
                                                     round1_summaries, config)
            response = gpt_chat_with_images(client, gpt_model, messages)

            if response:
                parsed = parse_json_response(response)
                if parsed:
                    # 更新特征和模式
                    label_key = f"label_{label}"
                    if label_key in updated_summaries:
                        # 追加新的特征
                        if "feature" in parsed and parsed["feature"]:
                            updated_summaries[label_key]["feature"].extend(parsed["feature"])
                        # 追加新的模式
                        if "pattern" in parsed and parsed["pattern"]:
                            updated_summaries[label_key]["pattern"].extend(parsed["pattern"])

            time.sleep(1)

    return updated_summaries


def phase1_round2_wrong(client, gpt_model, dataset_name: str, config: Dict,
                        round1_summaries: Dict, args, model_results_dict: Dict):
    """第一阶段第二轮：处理错误样本并更新总结 - 使用prompt22"""
    updated_summaries = round1_summaries.copy()

    sample_dict = config.get("wrong_IDS", {})
    image_root = "D:/zuhui/xITSC/data/image/test"

    for label_str, sample_ids in sample_dict.items():
        label = int(label_str)

        for sample_id in sample_ids:
            print(f"Processing wrong sample {sample_id} for label {label}...")

            # 获取样本图片
            images = get_reference_images(dataset_name, label, [sample_id], image_root)
            if not images:
                continue

            sample_data = {
                "sample_id": sample_id,
                "label": label,
                "images": images[0]["images"]
            }

            # 获取该样本的模型输出信息
            model_info = model_results_dict.get(str(sample_id), {})

            # 构建消息并发送 - 使用错误样本的提示词，包含模型输出
            messages = build_round2_messages_wrong(dataset_name, label, sample_data,
                                                   round1_summaries, model_info, config)
            response = gpt_chat_with_images(client, gpt_model, messages)

            if response:
                parsed = parse_json_response(response)
                if parsed:
                    # 更新特征和模式
                    label_key = f"label_{label}"
                    if label_key in updated_summaries:
                        # 追加新的特征
                        if "feature" in parsed and parsed["feature"]:
                            updated_summaries[label_key]["feature"].extend(parsed["feature"])
                        # 追加新的模式
                        if "pattern" in parsed and parsed["pattern"]:
                            updated_summaries[label_key]["pattern"].extend(parsed["pattern"])

            time.sleep(1)

    return updated_summaries


def get_similar_dissimilar_samples(test_sample_idx: int, train_data, train_labels, n_similar: int = 4):
    """使用DTW找到相似样本"""
    similar_indices = []
    return similar_indices[:n_similar]


def process_test_sample_complete(client, gpt_model, dataset_name: str, config: Dict,
                                 label_summaries_correct: Dict, label_summaries_wrong: Dict,
                                 test_sample: Dict, model_results_dict: Dict,
                                 train_data=None, train_labels=None, args=None):
    """完整处理一个测试样本：连续执行第二轮的所有步骤"""

    sample_id = test_sample["sample_id"]
    print(f"\n=== 开始完整处理测试样本 {sample_id} ===")

    # 获取模型预测结果
    model_info = model_results_dict.get(str(sample_id))
    if not model_info:
        print(f"警告：样本 {sample_id} 的模型结果不存在，跳过")
        return None

    # ===== 步骤1: Round 3 =====
    print(f"步骤1/5: Round 3 - 初始预测")
    messages = build_round3_messages(dataset_name, test_sample, label_summaries_wrong, config)
    round3_response = gpt_chat_with_images(client, gpt_model, messages)
    print(round3_response)

    if not round3_response:
        print(f"Round 3 失败，跳过样本 {sample_id}")
        return None

    round3_predicted = None
    round3_rationale = ""
    round3_score = ""

    # 尝试解析JSON响应
    round3_parsed = parse_json_response(round3_response)

    if round3_parsed:
        # JSON解析成功，尝试从JSON结构中提取
        result_list = round3_parsed.get("result", [])
        rationale_list = round3_parsed.get("rationale", [])
        score_list = round3_parsed.get("score", [])

        if result_list and isinstance(result_list, list):
            first_result = result_list[0]
            if isinstance(first_result, int):
                round3_predicted = first_result
            elif isinstance(first_result, str):
                # 如果是字符串，尝试提取数字
                numbers = re.findall(r'\d+', first_result)
                if numbers:
                    round3_predicted = int(numbers[0])
                else:
                    # 如果是"no"/"yes"等，尝试转换为数字
                    first_result_lower = first_result.lower()
                    if "no" in first_result_lower or "false" in first_result_lower or "negative" in first_result_lower:
                        round3_predicted = 1338
                    elif "yes" in first_result_lower or "true" in first_result_lower or "positive" in first_result_lower:
                        round3_predicted = 1837

        round3_rationale = rationale_list[0] if rationale_list and isinstance(rationale_list, list) else ""
        round3_score = score_list[0] if score_list and isinstance(score_list, list) else ""

    # 如果JSON解析失败或没有提取到数字，使用正则表达式
    if round3_predicted is None:
        print("JSON解析失败或未提取到数字，尝试使用正则表达式...")
        round3_predicted = extract_result_from_response(round3_response, 'round3')

    # 修复matches判断逻辑，避免None值比较报错
    model_predicted = model_info["predicted"]
    matches = (round3_predicted == model_predicted) if round3_predicted is not None else False

    print(f"Round 3 预测: {round3_predicted}, 模型预测: {model_predicted}, 是否匹配: {matches}")

    # 提取rationale和score的简短版本用于显示
    if round3_rationale:
        print(f"Round 3 Rationale: {round3_rationale[:100]}...")  # 显示前100个字符
    else:
        print("Round 3 Rationale: 未提取到")

    round4_results = []
    final_round5_result = None

    if matches:
        # ===== 步骤2-4: 4.1分支 (三轮对话) =====
        print(f"步骤2-4/5: 4.1分支 - 处理匹配情况")

        try:
            # 使用提供的函数获取相似和不相似样本
            from TSC1.dtw import find_similar_and_dissimilar_samples
            result = find_similar_and_dissimilar_samples(
                test_sample_id=sample_id,
                data_name=dataset_name
            )

            similar_samples_info = result['similar_samples']  # [(id, label), ...]
            dissimilar_samples_info = result['dissimilar_samples']  # [(id, label), ...]

            print(f"找到 {len(similar_samples_info)} 个相似样本和 {len(dissimilar_samples_info)} 个不相似样本")

            # 获取相似样本的图片数据
            similar_samples_data = []
            for sample_idx, label in similar_samples_info:
                images = get_reference_images(dataset_name, label, [sample_idx], args.image_root)
                if images:
                    similar_samples_data.append({
                        "sample_id": sample_idx,
                        "label": label,
                        "images": images[0]["images"]
                    })
                else:
                    print(f"警告：相似样本 {sample_idx} (标签 {label}) 的图片未找到")

            # 获取不相似样本的图片数据
            dissimilar_samples_data = []
            for sample_idx, label in dissimilar_samples_info:
                images = get_reference_images(dataset_name, label, [sample_idx], args.image_root)
                if images:
                    dissimilar_samples_data.append({
                        "sample_id": sample_idx,
                        "label": label,
                        "images": images[0]["images"]
                    })
                else:
                    print(f"警告：不相似样本 {sample_idx} (标签 {label}) 的图片未找到")

        except Exception as e:
            print(f"获取相似/不相似样本时出错: {e}")
            # 出错时使用空列表
            similar_samples_data = []
            dissimilar_samples_data = []

        # 执行三轮4.1对话
        for round_num in range(1, 4):
            print(f"  4.1分支 - 第{round_num}轮")
            messages = build_round4_1_messages(
                dataset_name, test_sample, similar_samples_data, dissimilar_samples_data,
                label_summaries_correct, round3_rationale, config
            )
            response = gpt_chat_with_images(client, gpt_model, messages)
            print(response)

            if response:
                round4_results.append(response)
            time.sleep(1)

        # ===== 步骤5: Round 5 =====
        print(f"步骤5/5: Round 5 - 最终决策")
        messages = build_round51_messages(
            dataset_name, test_sample, label_summaries_correct, model_info,
            round4_results, round3_rationale, config, use_heatmap=False
        )
        final_round5_result = gpt_chat_with_images(client, gpt_model, messages)
        print(final_round5_result)

    else:
        # ===== 步骤2-4: 4.2分支 (三轮对话) =====
        print(f"步骤2-4/5: 4.2分支 - 处理不匹配情况")

        try:
            # 使用提供的函数获取每个标签的最相似参考样本
            from data.similarity import find_most_similar_for_each_label

            label_to_most_similar = find_most_similar_for_each_label(
                test_sample_idx=sample_id,
                label=test_sample["label"],
                dataset_name=dataset_name,
                test_shap_path=r"D:\zuhui\xITSC\data\image\test",
                train_shap_path=r"D:\zuhui\xITSC\data\image\train",
                top_k_candidates=5
            )

            print(f"为每个标签找到的最相似训练样本: {label_to_most_similar}")

            # 获取参考样本数据
            reference_samples_data = []
            for label_str, sample_idx in label_to_most_similar.items():
                if sample_idx is None:
                    continue
                label = int(label_str)
                images = get_reference_images(dataset_name, label, [sample_idx], args.image_root)
                if images:
                    reference_samples_data.append({
                        "sample_id": sample_idx,
                        "label": label,
                        "images": images[0]["images"]
                    })
                else:
                    print(f"警告：参考样本 {sample_idx} (标签 {label}) 的图片未找到")

        except Exception as e:
            print(f"获取参考样本时出错: {e}")
            # 出错时使用默认方法（从配置中获取）
            reference_samples_data = []
            all_labels = list(label_summaries_correct.keys())

            for label_key in all_labels:
                label = int(label_key.replace("label_", ""))
                # 从配置中获取该标签的样本
                correct_ids = config.get("correct_IDS", {}).get(str(label), [])
                if correct_ids:
                    # 随机选择1个样本作为参考
                    selected_id = random.choice(correct_ids)
                    images = get_reference_images(dataset_name, label, [selected_id], args.image_root)
                    if images:
                        reference_samples_data.append({
                            "sample_id": selected_id,
                            "label": label,
                            "images": images[0]["images"]
                        })

        # 执行三轮4.2对话
        for round_num in range(1, 4):
            print(f"  4.2分支 - 第{round_num}轮")
            messages = build_round4_2_messages(
                dataset_name, test_sample, reference_samples_data,
                label_summaries_correct, config
            )
            response = gpt_chat_with_images(client, gpt_model, messages)
            print(response)

            if response:
                round4_results.append(response)
            time.sleep(1)

        # ===== 步骤5: Round 5 =====
        print(f"步骤5/5: Round 5 - 最终决策")
        messages = build_round52_messages(
            dataset_name, test_sample, label_summaries_correct, model_info,
            round4_results, config, use_heatmap=True
        )
        final_round5_result = gpt_chat_with_images(client, gpt_model, messages)
        print(final_round5_result)

    # 从最终结果中提取预测
    if final_round5_result:
        round4_r = extract_result_from_response(final_round5_result, 'round4_result')
        round4_m = extract_result_from_response(final_round5_result, 'round4_model')
    else:
        round4_r = None
        round4_m = None

    # 从round4_results中提取中间预测
    round3_1 = extract_result_from_response(round4_results[0], 'round3') if len(round4_results) > 0 else None
    round3_2 = extract_result_from_response(round4_results[1], 'round3') if len(round4_results) > 1 else None
    round3_3 = extract_result_from_response(round4_results[2], 'round3') if len(round4_results) > 2 else None

    result = {
        "sample_id": str(sample_id),
        "true_label": test_sample["label"],
        "round3_rationale": round3_rationale,
        "model_predicted": model_predicted,
        "matches": matches,
        "round3_response": round3_response,
        "round4_responses": round4_results,
        "round5_response": final_round5_result
    }

    print(f"样本 {sample_id} 处理完成")
    return result


def main():
    args = parse_arguments()

    # 加载配置文件
    config = load_json(args.config_file)
    dataset_name = config.get("dataset_name", "computer")

    # 加载模型结果
    model_results = load_json(args.model_results_file)
    # 转换为字典，键为样本ID
    model_results_dict = {str(item["id"]): item for item in model_results}

    # 第一阶段结果文件名
    phase1_output = args.output_file.replace(".json", "_phase1.json")

    # 检查第一阶段结果文件是否存在
    if os.path.exists(phase1_output):
        print(f"发现已有的第一阶段结果文件 {phase1_output}，直接加载...")
        phase1_results = load_json(phase1_output)
        label_summaries_correct = phase1_results.get("label_summaries_correct", {})
        label_summaries_wrong = phase1_results.get("label_summaries_wrong", {})
        print("第一阶段结果加载完成！跳过第一阶段处理。")
    else:
        print("未发现第一阶段结果文件，开始执行第一阶段...")

        # 第一阶段两个独立分支
        print("\n=== Round 1 Correct: Processing correct samples (第一轮) ===")
        round1_summaries_correct = phase1_round1_correct(client, args.gpt_model, dataset_name,
                                                         config, args, model_results_dict)
        print(round1_summaries_correct)
        print("\n=== Round 1 Wrong: Processing wrong samples (第一轮) ===")
        round1_summaries_wrong = phase1_round1_wrong(client, args.gpt_model, dataset_name,
                                                     config, args, model_results_dict)
        print(round1_summaries_wrong)

        # 第一阶段第二轮 - correct samples
        print("\n=== Round 2 Correct: Processing correct samples (第二轮) ===")
        label_summaries_correct = phase1_round2_correct(client, args.gpt_model, dataset_name, config,
                                                        round1_summaries_correct, args, model_results_dict)
        print(label_summaries_correct)

        # 第一阶段第二轮 - wrong samples
        print("\n=== Round 2 Wrong: Processing wrong samples (第二轮) ===")
        label_summaries_wrong = phase1_round2_wrong(client, args.gpt_model, dataset_name, config,
                                                    round1_summaries_wrong, args, model_results_dict)
        print(label_summaries_wrong)

        # 保存第一阶段结果
        phase1_results = {
            "label_summaries_correct": label_summaries_correct,
            "label_summaries_wrong": label_summaries_wrong
        }
        save_json(phase1_results, phase1_output)
        print(f"Phase 1 results saved to {phase1_output}")

    print("\n" + "=" * 50)
    print("Starting Phase 2: Processing test samples...")
    print("=" * 50)

    # 收集测试样本
    test_samples = []
    if os.path.exists(args.test_image_root):
        # 遍历test_image_root下的所有文件夹
        for folder_name in os.listdir(args.test_image_root):
            folder_path = os.path.join(args.test_image_root, folder_name)

            # 检查是否是文件夹且符合命名格式 {dataset_name}_{label}
            if os.path.isdir(folder_path) and folder_name.startswith(f"{dataset_name}_"):
                # 从文件夹名中提取label
                try:
                    label = int(folder_name.split("_")[-1])
                except ValueError:
                    print(f"无法从文件夹名 {folder_name} 中解析标签，跳过")
                    continue


                print(f"扫描label {label}的测试样本...")

                # 扫描该文件夹下的所有样本文件
                # 先收集所有唯一的样本ID
                sample_ids = set()

                # 检查所有图片文件
                for filename in os.listdir(folder_path):
                    if filename.endswith(".png"):
                        # 尝试从文件名中提取样本ID
                        match = re.search(r'sample(\d+)', filename)
                        if match:
                            sample_ids.add(int(match.group(1)))

                print(f"在label {label}文件夹中找到 {len(sample_ids)} 个潜在样本ID")

                # 为每个样本ID检查是否三张图片都存在
                for sample_id in sorted(sample_ids):
                    # 检查三种类型的图片是否存在
                    required_images = [
                        f"plot_sample{sample_id}.png",
                        f"STFT_sample{sample_id}.png",
                        f"shap_sample{sample_id}.png"
                    ]

                    all_exist = all(
                        os.path.exists(os.path.join(folder_path, img_name))
                        for img_name in required_images
                    )

                    if all_exist:
                        # 获取测试样本图片
                        images = get_test_images(dataset_name, label, sample_id, args.test_image_root)
                        if images and len(images) == 3:  # 确保三张图片都存在
                            test_samples.append({
                                "sample_id": sample_id,
                                "label": label,
                                "images": images
                            })
                            print(f"  添加测试样本: ID={sample_id}, label={label}")

    print(f"总共找到 {len(test_samples)} 个完整的测试样本")

    # 第二阶段结果文件名
    phase2_output = args.output_file.replace(".json", "_phase2.json")

    # 加载已有的第二阶段结果（如果存在）
    phase2_results = []
    if os.path.exists(phase2_output):
        print(f"发现已有的第二阶段结果文件 {phase2_output}，加载已处理结果...")
        try:
            existing_results = load_json(phase2_output)
            if isinstance(existing_results, list):
                phase2_results = existing_results
            else:
                phase2_results = existing_results.get("results", [])
        except Exception as e:
            print(f"加载第二阶段结果文件失败: {e}")
            phase2_results = []

        print(f"已加载 {len(phase2_results)} 个已处理的测试样本结果")

        # 创建已处理样本ID集合
        processed_ids = {r["sample_id"] for r in phase2_results}

        # 过滤掉已处理的测试样本
        original_count = len(test_samples)
        test_samples = [s for s in test_samples if str(s["sample_id"]) not in processed_ids]
        print(f"过滤掉 {original_count - len(test_samples)} 个已处理样本，剩余 {len(test_samples)} 个待处理")

    # 第二阶段：连续处理每个测试样本
    for i, test_sample in enumerate(test_samples):
        print(f"\n处理测试样本 {i + 1}/{len(test_samples)} (ID: {test_sample['sample_id']})")

        result = process_test_sample_complete(
            client, args.gpt_model, dataset_name, config,
            label_summaries_correct, label_summaries_wrong,
            test_sample, model_results_dict,
            train_data=None, train_labels=None, args=args
        )

        if result:
            phase2_results.append(result)

            # 立即保存第二阶段结果到文件
            save_json(phase2_results, phase2_output)
            print(f"已保存第 {i + 1} 个测试样本的结果到 {phase2_output}")

        # 避免API限制
        time.sleep(2)

    # 保存最终完整结果（合并第一阶段和第二阶段）
    final_output = {
        "phase1_results": {
            "label_summaries_correct": label_summaries_correct,
            "label_summaries_wrong": label_summaries_wrong
        },
        "phase2_results": phase2_results
    }

    save_json(final_output, args.output_file)
    print(f"\n所有处理完成！最终结果保存到 {args.output_file}")

    # 统计结果
    if phase2_results:
        total_samples = len(phase2_results)
        matches_count = sum(1 for r in phase2_results if r.get("matches", False))
        print(f"\n统计信息:")
        print(f"总测试样本数: {total_samples}")
        print(f"Round 3与模型预测匹配数: {matches_count} ({matches_count / total_samples * 100:.1f}%)")


if __name__ == "__main__":
    # 替换为你的默认参数
    import sys

    sys.argv = [
        "xitsc_with_images.py",
        "--config_file", r"D:\zuhui\xITSC\TSC\config.json",
        "--gpt_model", "gpt-5-mini",
        "--output_file", "D:/zuhui/xITSC/result.json"
    ]
    main()