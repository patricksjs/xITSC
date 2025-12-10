# stage1_processor.py (第一阶段处理器)
import os
import json
import copy
from time import sleep

import config
import prompt
from utils import (
    local_image_to_base64,
    gpt_chat,
    auto_load_reference_images
)


def get_round1_prompt_single_label(current_label_idx, class_samples_data):
    """生成第1轮提示词消息 - 只处理同一个label的所有样本"""
    base_prompt = f"""
{prompt.prompt1}

The label for which you currently need to perform feature extraction and summarization is: label {current_label_idx}

Next are the plots, time-frequency images, and heatmaps of {config.SAMPLES_PER_CLASS} reference samples for the label {current_label_idx} to be summarized:"""

    # 开始构建消息内容
    messages = [
        {
            "type": "text",
            "text": base_prompt
        }
    ]

    # 添加当前label的所有样本
    for i, sample_data in enumerate(class_samples_data):
        # 添加样本标识文本
        messages.append({
            "type": "text",
            "text": f"sample {i + 1}/{config.SAMPLES_PER_CLASS}: "
        })

        # 添加三张图像：折线图、时频图、热力图
        if sample_data["line_chart"]:
            messages.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{sample_data['line_chart']}"}
            })

        if sample_data["spectrogram"]:
            messages.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{sample_data['spectrogram']}"}
            })

        if sample_data["heatmap"]:
            messages.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{sample_data['heatmap']}"}
            })



    return messages


def get_round2_prompt_single_sample(sample_idx, label_summary, sample_data):
    """生成第2轮提示词消息 - 处理单个参考样本"""
    text_prompt = f"""
{prompt.prompt2}

Current processing sample: {sample_idx + 1}

Preliminary feature summary of the label corresponding to this sample:
{json.dumps(label_summary, ensure_ascii=False)}

Next are the plot, time-frequency image, and heatmap of the sample you need to analyze.:"""

    # 构建消息内容
    messages = [
        {
            "type": "text",
            "text": text_prompt
        }
    ]

    # 添加当前样本的三张图像
    if sample_data["line_chart"]:
        messages.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{sample_data['line_chart']}"}
        })

    if sample_data["spectrogram"]:
        messages.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{sample_data['spectrogram']}"}
        })

    if sample_data["heatmap"]:
        messages.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{sample_data['heatmap']}"}
        })



    return messages


def process_single_label_reference_samples():
    """第一阶段：处理参考样本，分为两个轮次"""
    print("开始第一阶段：处理参考样本")

    # 加载参考样本
    reference_images, selected_ids = auto_load_reference_images(config.ROOT_SPECTROGRAM_FOLDER)

    if not reference_images:
        print("无参考样本数据，程序退出")
        return None, None

    # 预先转换所有参考样本的图像数据
    all_reference_images_data = {}
    for class_name, class_data in reference_images.items():
        class_idx = int(class_name.split('_')[-1])
        line_charts = class_data["line_charts"]
        time_frequency = class_data["time_frequency"]
        heatmaps = class_data.get("heatmap", [])

        min_count = min(len(line_charts), len(time_frequency))
        class_images_data = []

        for i in range(min_count):
            line_data = local_image_to_base64(line_charts[i])
            spec_data = local_image_to_base64(time_frequency[i])
            heatmap_data = local_image_to_base64(heatmaps[i]) if i < len(heatmaps) and heatmaps[i] else None

            image_pair = {
                "line_chart": line_data,
                "spectrogram": spec_data,
                "heatmap": heatmap_data
            }
            class_images_data.append(image_pair)

        all_reference_images_data[class_idx] = class_images_data

    # 第一轮：处理每个类别的参考样本（只处理同一个label的所有样本）
    print("开始第一轮：处理每个类别的参考样本")
    label_summaries_round1 = {}

    for class_idx in range(config.CLASS_COUNT):
        class_name = f"label_{class_idx}"
        print(f"\n处理 {class_name} 的参考样本（第一轮）...")

        if class_idx not in all_reference_images_data:
            print(f"{class_name} 无有效参考样本，跳过")
            continue

        # 获取当前label的所有样本
        current_label_samples = all_reference_images_data[class_idx]

        # 构建提示词消息
        messages = get_round1_prompt_single_label(class_idx, current_label_samples)

        # Round 1: 分析参考样本特征
        print(f"分析 {class_name} 的特征...")
        label_summary = gpt_chat(messages, [])

        if label_summary:
            # 确保输出格式为 { "feature": [], "pattern": [] }
            if isinstance(label_summary, str):
                try:
                    label_summary = json.loads(label_summary)
                except:
                    # 如果解析失败，创建默认结构
                    label_summary = {"feature": [label_summary], "pattern": [label_summary]}

            # 确保有feature和pattern字段
            if "feature" not in label_summary:
                label_summary["feature"] = []
            if "pattern" not in label_summary:
                label_summary["pattern"] = []

            label_summaries_round1[class_name] = label_summary
            print(f"{class_name} 特征总结生成成功")
        else:
            # 创建空的总结结构
            label_summaries_round1[class_name] = {"feature": [], "pattern": []}
            print(f"{class_name} 特征总结生成失败，使用空结构")

        # 标签间延迟
        sleep(1)

    # 第二轮：循环遍历每一个参考样本
    print("\n开始第二轮：逐个处理参考样本")
    label_summaries_final = copy.deepcopy(label_summaries_round1)

    for class_idx in range(config.CLASS_COUNT):
        class_name = f"label_{class_idx}"

        if class_idx not in all_reference_images_data:
            continue

        current_label_samples = all_reference_images_data[class_idx]
        current_label_summary = label_summaries_round1.get(class_name, {"feature": [], "pattern": []})

        print(f"\n处理 {class_name} 的 {len(current_label_samples)} 个参考样本（第二轮）...")

        for sample_idx, sample_data in enumerate(current_label_samples):
            print(f"  处理样本 {sample_idx + 1}/{len(current_label_samples)}...")

            # 构建第二轮提示词消息
            messages = get_round2_prompt_single_sample(sample_idx, current_label_summary, sample_data)

            # 处理单个样本
            sample_summary = gpt_chat(messages, [])

            if sample_summary:
                # 解析样本总结
                if isinstance(sample_summary, str):
                    try:
                        sample_summary = json.loads(sample_summary)
                    except:
                        sample_summary = {"feature": [sample_summary], "pattern": [sample_summary]}

                # 确保有feature和pattern字段
                if "feature" not in sample_summary:
                    sample_summary["feature"] = []
                if "pattern" not in sample_summary:
                    sample_summary["pattern"] = []

                # 将第二轮结果添加到第一轮对应字段
                label_summaries_final[class_name]["feature"].extend(sample_summary["feature"])
                label_summaries_final[class_name]["pattern"].extend(sample_summary["pattern"])

                print(
                    f"    样本 {sample_idx + 1} 处理成功，添加了 {len(sample_summary['feature'])} 个特征和 {len(sample_summary['pattern'])} 个模式")
            else:
                print(f"    样本 {sample_idx + 1} 处理失败")

            # 样本间延迟
            sleep(0.5)

    # 保存第一阶段结果
    stage1_output = {
        "label_summaries_round1": label_summaries_round1,  # 第一轮结果
        "label_summaries_final": label_summaries_final,  # 第二轮合并后的结果
        "selected_reference_ids": selected_ids
    }

    os.makedirs(os.path.dirname(config.STAGE1_OUTPUT_FILE), exist_ok=True)
    with open(config.STAGE1_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(stage1_output, f, indent=4, ensure_ascii=False)

    print(f"\n第一阶段完成！结果已保存至: {config.STAGE1_OUTPUT_FILE}")
    return label_summaries_final, selected_ids


if __name__ == "__main__":
    process_single_label_reference_samples()