# main.py (修改后的主程序)
import os
import json
import copy
from time import sleep
import random

import config
import prompt
from utils import (
    local_image_to_base64,
    gpt_chat,
    batch_read_test_samples,
    build_multimodal_content_with_interleaved_text_images,
    auto_load_reference_images,
    extract_classification_results,
    save_classification_results
)


def get_round1_prompt_with_comparison(current_label_idx, other_label_samples):
    """生成第1轮提示词 - 包含当前label的所有样本和其他label的对比样本"""

    base_prompt = f"""
{prompt.prompt1}

The label for which you currently need to perform feature extraction and summarization is: label {current_label_idx}

Next are the plots, time-frequency images, and heatmaps of {config.SAMPLES_PER_CLASS} reference samples for the label {current_label_idx} to be summarized:"""

    text_blocks = [base_prompt]
    image_blocks = [[]]

    # 添加当前label的所有样本
    for i in range(config.SAMPLES_PER_CLASS):
        text_blocks.append(f"sample {i + 1}/{config.SAMPLES_PER_CLASS}: ")
        image_blocks.append([])

    # 添加其他label的对比样本
    text_blocks.append("Next are the images of the reference samples from other labels for comparative analysis: ")
    image_blocks.append([])

    for label_idx, sample_data in other_label_samples.items():
        text_blocks.append(f"The images of the sample for label {label_idx} are: ")
        image_blocks.append([])

    return text_blocks, image_blocks


def get_round2_prompt(all_label_summaries):
    """生成第2轮提示词 - 处理所有label的特征总结"""
    return f"""

All Label Feature Summaries:
{json.dumps(all_label_summaries, ensure_ascii=False)}

{prompt.prompt2}"""


def get_round3_prompt(all_label_summaries, label_summaries_processed):
    """生成第3轮提示词 - 使用处理后的label特征进行分类"""
    return f"""
# Feature Analysis Report
## 1. Detailed Analysis for Each Label
This section provides an in-depth analysis of the features within each individual label:
{json.dumps(all_label_summaries, ensure_ascii=False)}

## 2. Cross-Label Comparison Analysis
This section compares features across different labels to highlight similarities and differences:
{json.dumps(label_summaries_processed, ensure_ascii=False)}

{prompt.prompt3}"""


def get_round4_prompt(all_label_summaries, thinking_chains, label_summaries_processed, domain_model_result):
    """生成第4轮提示词 - 整合三条思维链得到最终结果，包含领域模型预测结果"""
    return f"""

Three classification thought chains:
{json.dumps(thinking_chains, ensure_ascii=False)}

# Feature Analysis Report
## 1. Detailed Analysis for Each Label
This section provides an in-depth analysis of the features within each individual label:
{json.dumps(all_label_summaries, ensure_ascii=False)}

## 2. Cross-Label Comparison Analysis
This section compares features across different labels to highlight similarities and differences:
{json.dumps(label_summaries_processed, ensure_ascii=False)}

Domain Model Classification Results:
{json.dumps(domain_model_result, ensure_ascii=False)}

{prompt.prompt4}"""


def get_closest_reference_samples_for_test(test_sample_id, dataset_name="computer"):
    """
    为待分类样本找到每个label中最接近的两个参考样本
    """
    try:
        from data.data_loader import load_data
    except ImportError:
        print("错误：无法导入 load_data 函数")
        return None

    # 加载数据
    data, labels = load_data(dataset_name)

    # 定义参考样本ID
    manual_ids = config.MANUAL_IDS
    from select1 import process_test_sample_with_dtw

    # 处理测试样本
    result = process_test_sample_with_dtw(
        test_sample_id, data, labels, manual_ids,
        dataset_name, config.ROOT_SPECTROGRAM_FOLDER
    )

    return result


def build_multimodal_content_with_closest_samples(test_images_data, closest_samples_result):
    """
    构建包含待分类样本和最近参考样本的多模态内容
    """
    from utils import local_image_to_base64

    text_blocks = []
    image_blocks = []

    # 添加待分类样本
    text_blocks.append("Next are the three types of images for the test sample to be classified:")
    image_blocks.append(test_images_data)

    # 添加每个label的参考样本
    text_blocks.append(
        "\nNext are the reference sample images (two samples per label). You can further analyze which label the test sample is most likely to be classified as based on these reference sample images to prevent understanding bias from text feature summaries:")
    image_blocks.append([])  # 空图像块，对应上面的文本

    closest_samples_images = closest_samples_result["closest_samples_images"]

    for label, samples_dict in closest_samples_images.items():
        text_blocks.append(f"\nReference samples for label {label}:")
        image_blocks.append([])  # 空图像块，对应label描述

        for sample_id, image_paths in samples_dict.items():
            # 转换图片为base64
            line_data = local_image_to_base64(image_paths["line_chart"]) if image_paths["line_chart"] else None
            spec_data = local_image_to_base64(image_paths["spectrogram"]) if image_paths["spectrogram"] else None
            heatmap_data = local_image_to_base64(image_paths["heatmap"]) if image_paths["heatmap"] else None

            sample_images = []
            if line_data:
                sample_images.append(line_data)
            if spec_data:
                sample_images.append(spec_data)
            if heatmap_data:
                sample_images.append(heatmap_data)

            text_blocks.append(f"Sample {sample_id} for label {label}:")
            image_blocks.append(sample_images)

    return build_multimodal_content_with_interleaved_text_images(text_blocks, image_blocks)


def load_domain_model_results():
    """从配置文件路径加载领域模型预测结果"""
    if not os.path.exists(config.model_file_path):
        print(f"领域模型结果文件不存在: {config.model_file_path}")
        return None

    try:
        with open(config.model_file_path, 'r', encoding='utf-8') as f:
            model_results = json.load(f)

        # 提取需要的字段：predicted, result, logits
        processed_results = []
        for result in model_results:
            processed_result = {
                "id": result.get("id"),
                "predicted": result.get("predicted"),
                "result": result.get("result", []),
                "logits": result.get("logits", [])
            }
            processed_results.append(processed_result)

        return processed_results
    except Exception as e:
        print(f"加载领域模型结果失败: {e}")
        return None


def get_domain_model_result_for_sample(domain_model_results, sample_id):
    """根据样本ID获取对应的领域模型预测结果"""
    if not domain_model_results:
        return None

    # 尝试将sample_id转换为整数进行匹配
    try:
        sample_id_int = int(sample_id)
        for result in domain_model_results:
            if result.get("id") == sample_id_int:
                return result
    except ValueError:
        # 如果sample_id不是整数，尝试其他匹配方式
        for result in domain_model_results:
            if str(result.get("id")) == str(sample_id):
                return result

    return None


def process_reference_samples():
    """第一阶段：处理参考样本，包含当前label的所有样本和其他label的对比样本"""
    print("开始第一阶段：处理参考样本")

    # 加载参考样本
    reference_images, selected_ids = auto_load_reference_images(config.ROOT_SPECTROGRAM_FOLDER)

    if not reference_images:
        print("无参考样本数据，程序退出")
        return None, None, None

    label_summaries = {}
    selected_reference_ids = {}
    reference_images_data = {}  # 存储所有参考样本的图像数据

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

        # 记录选中的样本ID
        sample_ids = []
        for line_path in line_charts[:min_count]:
            filename = os.path.basename(line_path)
            sample_id = filename.replace("plot_sample", "").replace(".png", "")
            sample_ids.append(sample_id)

        selected_reference_ids[f"label_{class_idx}"] = sample_ids
        reference_images_data[f"label_{class_idx}"] = class_images_data

    # 处理每个类别的参考样本
    for class_idx in range(config.CLASS_COUNT):
        class_name = f"label_{class_idx}"
        print(f"\n处理 {class_name} 的参考样本...")

        if class_idx not in all_reference_images_data:
            print(f"{class_name} 无有效参考样本，跳过")
            continue

        # 获取当前label的所有样本
        current_label_samples = all_reference_images_data[class_idx]

        # 随机选择其他label各一个样本用于对比
        other_label_samples = {}
        for other_idx in range(config.CLASS_COUNT):
            if other_idx != class_idx and other_idx in all_reference_images_data:
                other_samples = all_reference_images_data[other_idx]
                if other_samples:
                    random_sample = random.choice(other_samples)
                    other_label_samples[other_idx] = random_sample

        # 构建提示词和图片块
        text_blocks, image_blocks = get_round1_prompt_with_comparison(class_idx, other_label_samples)

        # 填充当前label的所有样本图片
        for i, sample_data in enumerate(current_label_samples):
            if i + 1 < len(image_blocks):  # i+1 因为索引0是基础提示
                image_blocks[i + 1] = [sample_data["line_chart"], sample_data["spectrogram"], sample_data["heatmap"]]

        # 填充其他label的对比样本图片
        current_index = len(current_label_samples) + 2
        for label_idx, sample_data in other_label_samples.items():
            if current_index < len(image_blocks):
                image_blocks[current_index] = [sample_data["line_chart"], sample_data["spectrogram"],
                                               sample_data["heatmap"]]
                current_index += 1

        # 构建多模态内容
        multimodal_content = build_multimodal_content_with_interleaved_text_images(text_blocks, image_blocks)

        # Round 1: 分析参考样本特征
        print(f"分析 {class_name} 的特征...")
        label_summary = gpt_chat(multimodal_content, [])

        if label_summary:
            label_summaries[class_name] = label_summary
            print(f"{class_name} 特征总结生成成功")
        else:
            print(f"{class_name} 特征总结生成失败")

        # 标签间延迟
        sleep(1)

    # Round 2: 处理所有label的特征总结
    print("\n执行第2轮：处理所有label的特征总结...")
    label_summaries_processed = gpt_chat(get_round2_prompt(label_summaries), [])

    # 保存第一阶段和第二阶段的结果
    stage1_output = {
        "label_summaries": label_summaries,  # 第一轮结果
        "label_summaries_processed": label_summaries_processed,  # 第二轮结果
        "selected_reference_ids": selected_reference_ids
    }

    os.makedirs(os.path.dirname(config.STAGE1_OUTPUT_FILE), exist_ok=True)
    with open(config.STAGE1_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(stage1_output, f, indent=4, ensure_ascii=False)

    print(f"\n第一阶段完成！结果已保存至: {config.STAGE1_OUTPUT_FILE}")
    return label_summaries, label_summaries_processed, selected_reference_ids


def generate_thinking_chains(all_label_summaries, label_summaries_processed, test_images_data):
    """Round3: 生成三条分类思维链"""
    print("执行第3轮：生成三条分类思维链...")

    thinking_chains = {}

    for i in range(3):
        print(f"  生成第 {i + 1} 条思维链...")

        chain_result = gpt_chat(
            build_multimodal_content_with_interleaved_text_images(
                [get_round3_prompt(all_label_summaries, label_summaries_processed)],
                [test_images_data]  # 传入所有测试样本图像
            ),
            []
        )

        if chain_result:
            thinking_chains[f"chain_{i + 1}"] = chain_result
            print(f"    第 {i + 1} 条思维链生成成功")
        else:
            thinking_chains[f"chain_{i + 1}"] = {"error": "生成失败"}
            print(f"    第 {i + 1} 条思维链生成失败")

    return thinking_chains


def process_final_classification(all_label_summaries, thinking_chains, label_summaries_processed, domain_model_result,
                                 test_images_data):
    """Round4: 最终分类，只传入测试样本图像"""
    print("执行第4轮：最终分类（只传入测试样本图像）...")

    final_result = gpt_chat(
        build_multimodal_content_with_interleaved_text_images(
            [get_round4_prompt(all_label_summaries, thinking_chains, label_summaries_processed, domain_model_result)],
            [test_images_data]  # 传入测试样本图像
        ),
        []
    )

    return final_result


def process_test_samples(label_summaries, label_summaries_processed):
    """第二阶段：测试样本分析"""
    print("\n开始第二阶段：测试样本分析")

    if not label_summaries:
        print("无标签特征总结，程序退出")
        return

    # 加载领域模型结果
    domain_model_results = load_domain_model_results()
    if not domain_model_results:
        print("警告：无法加载领域模型结果，将继续处理但不包含领域模型信息")

    # 读取测试样本（现在包含真实标签）
    test_samples = batch_read_test_samples(config.TEST_IMAGE_FOLDER)
    if not test_samples:
        print("无测试样本，程序退出")
        return

    # 确保输出目录存在
    os.makedirs(os.path.dirname(config.STAGE2_OUTPUT_FILE), exist_ok=True)

    # 初始化输出数据
    stage2_output = {
        "test_samples": [],
        "stage1_reference": label_summaries,
        "stage2_processed": label_summaries_processed
    }
    if os.path.exists(config.STAGE2_OUTPUT_FILE):
        with open(config.STAGE2_OUTPUT_FILE, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
            stage2_output["test_samples"] = existing_data.get("test_samples", [])

    for idx, sample_info in enumerate(test_samples):
        sample_id = sample_info["sample_id"]
        sample_paths = sample_info["paths"]
        true_label = sample_info["true_label"]

        # 跳过已处理的样本
        processed_sample_ids = [s["sample_id"] for s in stage2_output["test_samples"]]
        if sample_id in processed_sample_ids:
            print(f"\n测试样本 {idx + 1}/{len(test_samples)}，ID: {sample_id} 已处理，跳过")
            continue

        print(f"\n处理测试样本 {idx + 1}/{len(test_samples)}，ID: {sample_id}，真实标签: {true_label}")

        # 转换测试图片为Base64
        test_line_data = local_image_to_base64(sample_paths[0])
        test_spec_data = local_image_to_base64(sample_paths[1])
        test_heatmap_data = local_image_to_base64(sample_paths[2]) if len(sample_paths) >= 3 and sample_paths[
            2] else None

        test_images_data = [test_line_data, test_spec_data]
        if test_heatmap_data:
            test_images_data.append(test_heatmap_data)

        if not all([test_line_data, test_spec_data]):
            print("测试图片转换失败，跳过")
            continue

        # 获取领域模型预测结果
        domain_model_result = get_domain_model_result_for_sample(domain_model_results, sample_id)

        # 找到最接近的参考样本
        print("计算最接近的参考样本...")
        closest_samples_result = get_closest_reference_samples_for_test(int(sample_id), config.dataset)

        sample_results = {
            "sample_id": sample_id,
            "true_label": true_label,
            "domain_model_result": domain_model_result,
            "thinking_chains": {},
            "final_result": {}
        }

        # Round 3: 生成三条分类思维链（包含最近参考样本）
        print("执行第3轮：生成三条分类思维链...")
        round3_prompt_text = get_round3_prompt(label_summaries, label_summaries_processed)
        closest_samples_content = build_multimodal_content_with_closest_samples(
            test_images_data, closest_samples_result
        )

        # 直接合并内容：第三轮提示词 + 样本内容
        # 由于build_multimodal_content_with_interleaved_text_images返回的是内容列表，
        # 我们需要将第三轮提示词作为一个单独的文本块添加到样本内容的前面
        round3_prompt_content = build_multimodal_content_with_interleaved_text_images(
            [round3_prompt_text], [[]]  # 第三轮提示词文本，没有对应的图像
        )

        # 合并内容列表
        combined_content_round3 = round3_prompt_content + closest_samples_content

        thinking_chains = {}
        for i in range(3):
            print(f"  生成第 {i + 1} 条思维链...")
            chain_result = gpt_chat(combined_content_round3, [])
            if chain_result:
                thinking_chains[f"chain_{i + 1}"] = chain_result
                print(f"    第 {i + 1} 条思维链生成成功")
            else:
                thinking_chains[f"chain_{i + 1}"] = {"error": "生成失败"}
                print(f"    第 {i + 1} 条思维链生成失败")

        sample_results["thinking_chains"] = thinking_chains

        # Round 4: 最终分类（同样包含最近参考样本）
        print("执行第4轮：最终分类...")

        # 构建第四轮的多模态内容
        round4_prompt_text = get_round4_prompt(
            label_summaries, thinking_chains, label_summaries_processed, domain_model_result
        )

        # 使用相同的样本内容，只更换提示词
        round4_prompt_content = build_multimodal_content_with_interleaved_text_images(
            [round4_prompt_text], [[]]  # 第四轮提示词文本，没有对应的图像
        )

        # 合并内容列表
        combined_content_round4 = round4_prompt_content + closest_samples_content

        final_result = gpt_chat(combined_content_round4, [])
        sample_results["final_result"] = final_result

        # 提取分类结果并保存
        classification_results = extract_classification_results(thinking_chains, final_result, true_label)
        save_classification_results(sample_id, classification_results, config.CLASSIFICATION_RESULTS_FILE)

        # 处理完成后，将当前样本结果添加到输出数据中
        stage2_output["test_samples"].append(sample_results)
        print(f"测试样本 {sample_id} 处理完成，写入文件...")

        # 实时写入文件
        with open(config.STAGE2_OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(stage2_output, f, indent=4, ensure_ascii=False)

        # 样本间延迟
        if idx < len(test_samples) - 1:
            sleep(1)

    print(f"\n第二阶段完成！所有样本结果已保存至: {config.STAGE2_OUTPUT_FILE}")


def main():
    """主函数：集成两个阶段的处理流程"""

    # 检查是否已有第一阶段结果
    if os.path.exists(config.STAGE1_OUTPUT_FILE):
        print("检测到已有第一阶段结果，直接加载...")
        with open(config.STAGE1_OUTPUT_FILE, 'r', encoding='utf-8') as f:
            stage1_data = json.load(f)
        label_summaries = stage1_data.get("label_summaries", {})
        label_summaries_processed = stage1_data.get("label_summaries_processed", {})
    else:
        # 执行第一阶段
        label_summaries, label_summaries_processed, selected_reference_ids = process_reference_samples()
        if not label_summaries:
            print("第一阶段处理失败，程序退出")
            return

    # 执行第二阶段
    process_test_samples(label_summaries, label_summaries_processed)


if __name__ == "__main__":
    main()