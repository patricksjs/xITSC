# main.py (修改后的主程序)
import os
import json
import re
from time import sleep
import prompt
import config
from stage1 import process_single_label_reference_samples
from utils import (
    local_image_to_base64,
    gpt_chat,
    batch_read_test_samples,
    extract_classification_results,
    save_classification_results
)
import numpy as np
import torch
import pandas as pd
# TODO :继续添加模型权重文件路径
def get_model_path_for_dataset(dataset_name):
    """根据数据集名称获取模型路径"""
    model_paths = {
        "computer": r"C:\Users\34517\Desktop\zuhui\xITSC\classification_models\computer\transformer\transformer.pt",
        "cincecgtorso": r"C:\Users\34517\Desktop\zuhui\xITSC\classification_models\cincecgtorso\transformer\transformer.pt",
        # 添加其他数据集的模型路径
    }
    return model_paths.get(dataset_name)


def get_model_args_for_dataset(dataset_name):
    """根据数据集名称获取模型参数"""

    class ModelArgs:
        def __init__(self, timesteps, num_classes):
            self.d_model = 64
            self.nhead = 8
            self.num_layers = 2
            self.dim_feedforward = 256
            self.dropout = 0.2
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.timesteps = timesteps
            self.num_classes = num_classes

    if dataset_name == "computer":
        return ModelArgs(720, 2)
    elif dataset_name == "cincecgtorso":
        return ModelArgs(1639, 2)
    else:
        # 默认参数
        return ModelArgs(720, 2)


def load_stage1_output2_summaries():
    """加载模型分错的特征总结"""
    if not os.path.exists(config.STAGE1_OUTPUT_FILE2):
        print(f"STAGE1_OUTPUT_FILE2不存在: {config.STAGE1_OUTPUT_FILE2}")
        return None

    try:
        with open(config.STAGE1_OUTPUT_FILE2, 'r', encoding='utf-8') as f:
            stage1_data = json.load(f)

        label_summaries = stage1_data.get("label_summaries_final", {})
        return label_summaries
    except Exception as e:
        print(f"加载STAGE1_OUTPUT_FILE2失败: {e}")
        return None


def extract_label_from_response(response_text):
    """使用正则表达式提取分类结果中的label编号"""
    pattern = re.compile(r'result(.*?)rationale', re.IGNORECASE | re.DOTALL)
    match = pattern.search(response_text)

    if match:
        result_text = match.group(1).strip()
        # 提取数字
        numbers = re.findall(r'\d+', result_text)
        if numbers:
            return int(numbers[0])

    return None



# TODO :提示词要修改
def get_round2_5_prompt(label_summaries, test_images_data):
    """生成第2.5轮提示词 - 使用STAGE1_OUTPUT_FILE2的label特征进行分类"""

    content = [
        {
            "type": "text",
            "text": f"""
# Feature Analysis Report (from STAGE1_OUTPUT_FILE2):
This section provides an in-depth analysis of the features and patterns within each individual label:
{json.dumps(label_summaries, ensure_ascii=False)}

Please classify the test image based on the above feature analysis.
Provide your answer in the format:
result: [label number]
rationale: [your reasoning]
"""
        }
    ]

    # 添加测试样本图片
    if test_images_data and len(test_images_data) > 0:
        content.append({
            "type": "text",
            "text": "The test image to be classified is as follows:"
        })

        for i, image_data in enumerate(test_images_data):
            if image_data:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_data}"}
                })

    return [{"role": "user", "content": content}]


def load_model_and_get_prediction(sample_id, dataset_name="computer"):
    """
    加载模型并获取
    测试样本的分类结果和特征
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 根据数据集获取模型参数
        args = get_model_args_for_dataset(dataset_name)

        # TODO: 检查这里的模型导入路径
        from models.feat import TransformerModel
        model = TransformerModel(args=args, num_classes=args.num_classes).to(device)

        # 加载预训练权重
        model_path = get_model_path_for_dataset(dataset_name)
        if not model_path or not os.path.exists(model_path):
            print(f"模型路径不存在: {model_path}")
            return None

        checkpoint = torch.load(model_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.eval()

        # 加载测试样本数据
        test_data = load_sample_data(sample_id, dataset_name, is_test=True)
        if test_data is None:
            return None

        # 转换为tensor并调整形状
        if test_data.dim() == 1:
            test_data = test_data.unsqueeze(0).unsqueeze(-1)  # [1, seq_len, 1]

        # 获取模型预测
        with torch.no_grad():
            logits, _ = model(test_data.to(device))
            predicted = torch.argmax(logits, dim=1).item()

        return predicted

    except Exception as e:
        print(f"加载模型获取预测失败: {e}")
        return None


def load_sample_data(sample_id, dataset_name, is_test=True):
    """
    加载样本数据
    """
    try:
        # 根据您提供的load_data函数加载数据
        if dataset_name == "computer":
            if is_test:
                file_path = r"C:\Users\34517\Desktop\zuhui\论文\Computers\Computers_TEST.txt"
            else:
                file_path = r"C:\Users\34517\Desktop\zuhui\论文\Computers\Computers_TRAIN.txt"

            df = pd.read_csv(file_path, header=None, sep='\s+')

            if sample_id < len(df):
                row = df.iloc[sample_id]
                data = row[1:].values.astype(np.float32)

                return torch.tensor(data, dtype=torch.float32)

        elif dataset_name == "cincecgtorso":
            if is_test:
                file_path = r"C:\Users\34517\Desktop\zuhui\论文\CinCECGTorso\CinCECGTorso_TEST.txt"
            else:
                file_path = r"C:\Users\34517\Desktop\zuhui\论文\CinCECGTorso\CinCECGTorso_TRAIN.txt"

            df = pd.read_csv(file_path, header=None, delim_whitespace=True)

            if sample_id < len(df):
                row = df.iloc[sample_id]
                data = row[1:].values.astype(np.float32)

                return torch.tensor(data, dtype=torch.float32)
        # TODO: 继续加数据集，不需要加载train
    except Exception as e:
        print(f"加载样本 {sample_id} 数据失败: {e}")

    return None


def get_sample_label(sample_id, dataset_name, is_test=True):
    """
    获取样本的标签
    """
    try:
        if dataset_name == "computer":
            if is_test:
                file_path = r"C:\Users\34517\Desktop\zuhui\论文\Computers\Computers_TEST.txt"
            else:
                file_path = r"C:\Users\34517\Desktop\zuhui\论文\Computers\Computers_TRAIN.txt"

            df = pd.read_csv(file_path, header=None, sep='\s+')

            if sample_id < len(df):
                label = int(df.iloc[sample_id, 0]) - 1  # 根据您的代码，标签减1
                return label

        elif dataset_name == "cincecgtorso":
            if is_test:
                file_path = r"C:\Users\34517\Desktop\zuhui\论文\CinCECGTorso\CinCECGTorso_TEST.txt"
            else:
                file_path = r"C:\Users\34517\Desktop\zuhui\论文\CinCECGTorso\CinCECGTorso_TRAIN.txt"

            df = pd.read_csv(file_path, header=None, delim_whitespace=True)

            if sample_id < len(df):
                label = int(df.iloc[sample_id, 0]) - 1  # 根据您的代码，标签减1
                return label
        # TODO: 继续加数据集，不需要加载train

    except Exception as e:
        print(f"获取样本 {sample_id} 标签失败: {e}")

    return None


def load_similar_sample_images(similar_sample_ids, dataset_name):
    """加载相似样本的图像（plot和时频图）"""
    similar_images_data = []

    for sample_id in similar_sample_ids:
        # 获取样本标签
        sample_label = get_sample_label(sample_id, dataset_name, is_test=False)
        if sample_label is None:
            continue

        # plot图像
        plot_path = f"D:/zuhui/xITSC/data/image1/{dataset_name}_{sample_label}/plot_sample{sample_id}.png"
        if os.path.exists(plot_path):
            plot_data = local_image_to_base64(plot_path)
            if plot_data:
                similar_images_data.append({
                    "type": "plot",
                    "sample_id": sample_id,
                    "label": sample_label,
                    "data": plot_data
                })

        # 时频图
        stft_path = f"D:/zuhui/xITSC/data/image1/{dataset_name}_{sample_label}/STFT_sample{sample_id}.png"
        if os.path.exists(stft_path):
            stft_data = local_image_to_base64(stft_path)
            if stft_data:
                similar_images_data.append({
                    "type": "stft",
                    "sample_id": sample_id,
                    "label": sample_label,
                    "data": stft_data
                })

    return similar_images_data


def get_round3_prompt_with_similar_samples(all_label_summaries, test_images_data, similar_images_data):
    """生成第3轮提示词 - 包含相似样本图像"""

    # 构建相似样本描述
    similar_samples_text = "Similar Reference Samples:\n"
    for img_info in similar_images_data:
        similar_samples_text += f"Sample {img_info['sample_id']} (Label: {img_info['label']}) - {img_info['type']}:\n"

    content = [
        {
            "type": "text",
            "text": f"""
# Feature Analysis Report:
This section provides an in-depth analysis of the features and patterns within each individual label:
{json.dumps(all_label_summaries, ensure_ascii=False)}

{similar_samples_text}

{prompt.prompt3}

Note: The following images include both the test sample and similar reference samples for comparison.
"""
        }
    ]

    # 添加测试样本图片
    if test_images_data and len(test_images_data) > 0:
        content.append({
            "type": "text",
            "text": "The test image to be classified is as follows:"
        })

        for image_data in test_images_data:
            if image_data:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_data}"}
                })

    # 添加相似样本图片
    if similar_images_data:
        content.append({
            "type": "text",
            "text": "Similar reference samples for comparison:"
        })

        for img_info in similar_images_data:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_info['data']}"}
            })

    return [{"role": "user", "content": content}]


def get_round4_prompt_with_similar_samples(all_label_summaries, thinking_chains, domain_model_result, test_images_data,
                                           similar_images_data):
    """生成第4轮提示词 - 包含相似样本图像"""

    # 构建相似样本描述
    similar_samples_text = "Similar Reference Samples:\n"
    for img_info in similar_images_data:
        similar_samples_text += f"Sample {img_info['sample_id']} (Label: {img_info['label']}) - {img_info['type']}:\n"

    content = [
        {
            "type": "text",
            "text": f"""
Three classification thought chains:
{json.dumps(thinking_chains, ensure_ascii=False)}

# Feature Analysis Report:
This section provides an in-depth analysis of the features and patterns within each individual label:
{json.dumps(all_label_summaries, ensure_ascii=False)}

{similar_samples_text}

Domain Model Classification Results:
{json.dumps(domain_model_result, ensure_ascii=False)}

{prompt.prompt4}

Note: The following images include both the test sample and similar reference samples for comparison.
"""
        }
    ]

    # 添加测试样本图片
    if test_images_data and len(test_images_data) > 0:
        content.append({
            "type": "text",
            "text": "The test image to be classified is as follows:"
        })

        for image_data in test_images_data:
            if image_data:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_data}"}
                })

    # 添加相似样本图片
    if similar_images_data:
        content.append({
            "type": "text",
            "text": "Similar reference samples for comparison:"
        })

        for img_info in similar_images_data:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_info['data']}"}
            })

    return [{"role": "user", "content": content}]


def load_domain_model_results():
    """从配置文件路径加载领域模型预测结果"""
    if not os.path.exists(config.model_file_path):
        print(f"领域模型结果文件不存在: {config.model_file_path}")
        return None

    try:
        with open(config.model_file_path, 'r', encoding='utf-8') as f:
            model_results = json.load(f)

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

    try:
        sample_id_int = int(sample_id)
        for result in domain_model_results:
            if result.get("id") == sample_id_int:
                return result
    except ValueError:
        for result in domain_model_results:
            if str(result.get("id")) == str(sample_id):
                return result

    return None


def process_round2_5(label_summaries, test_images_data):
    """处理第2.5轮对话"""
    print("执行第2.5轮：基于STAGE1_OUTPUT_FILE2特征进行分类...")

    # 生成提示词
    messages = get_round2_5_prompt(label_summaries, test_images_data)
    response = gpt_chat(messages, [])

    if response:
        # 提取分类结果
        label_summary_result = extract_label_from_response(response)
        print(f"第2.5轮分类结果: {label_summary_result}")
        return label_summary_result, response
    else:
        print("第2.5轮对话失败")
        return None, None


def get_similar_samples_by_feature_module(test_sample_id, dataset_name, model_path, model_args, k=5):
    """
    使用feature模块获取相似样本
    参数:
    - test_sample_id: 测试样本ID
    - dataset_name: 数据集名称
    - model_path: 模型路径
    - model_args: 模型参数
    - k: 返回的最相似样本数量
    """
    try:
        # 导入feature模块
        import feature

        # 使用缓存查找（更快）
        cache_file = f"feature_cache/{dataset_name}_features.pkl"
        if os.path.exists(cache_file):
            print("使用缓存查找相似样本...")
            top_k_indices, top_k_similarities = feature.find_top_k_with_cache(
                test_sample_id, k, dataset_name, cache_file, model_path, model_args
            )
        else:
            print("无缓存，实时计算相似样本...")
            top_k_indices, top_k_similarities = feature.find_top_k_similar_samples(
                test_sample_id, k, dataset_name, model_path, model_args
            )

        print(f"找到 {len(top_k_indices)} 个相似样本")
        return top_k_indices

    except Exception as e:
        # 尝试使用简化版本
        try:
            top_k_indices, _ = feature.main_simple(test_sample_id, k, dataset_name)
            return top_k_indices
        except Exception as e2:
            print(f"使用简化版本也失败: {e2}")
            return []


def generate_thinking_chains_with_similar_samples(all_label_summaries, test_images_data, similar_images_data):
    """Round3: 生成三条分类思维链（包含相似样本）"""
    print("执行第3轮：生成三条分类思维链（包含相似样本）...")

    thinking_chains = {}

    for i in range(3):
        print(f"  生成第 {i + 1} 条思维链...")

        # 构建消息（包含相似样本）
        messages = get_round3_prompt_with_similar_samples(
            all_label_summaries,
            test_images_data,
            similar_images_data
        )
        chain_result = gpt_chat(messages, [])

        if chain_result:
            thinking_chains[f"chain_{i + 1}"] = chain_result
            print(f"    第 {i + 1} 条思维链生成成功")
        else:
            thinking_chains[f"chain_{i + 1}"] = {"error": "生成失败"}
            print(f"    第 {i + 1} 条思维链生成失败")

    return thinking_chains


def process_final_classification_with_similar_samples(all_label_summaries, thinking_chains, domain_model_result,
                                                      test_images_data, similar_images_data):
    """Round4: 最终分类（包含相似样本）"""
    print("执行第4轮：最终分类（包含相似样本）...")

    # 构建消息（包含相似样本）
    messages = get_round4_prompt_with_similar_samples(
        all_label_summaries,
        thinking_chains,
        domain_model_result,
        test_images_data,
        similar_images_data
    )
    final_result = gpt_chat(messages, [])

    return final_result


def process_test_samples_with_new_features(all_label_summaries, dataset_name="computer"):
    """第二阶段：测试样本分析（添加新功能）"""
    print(f"\n开始第二阶段：测试样本分析，数据集: {dataset_name}")

    if not all_label_summaries:
        print("无标签特征总结，程序退出")
        return

    # 加载STAGE1_OUTPUT_FILE2的特征总结
    stage1_output2_summaries = load_stage1_output2_summaries()
    if not stage1_output2_summaries:
        print("警告：无法加载STAGE1_OUTPUT_FILE2特征总结")

    # 加载领域模型结果
    domain_model_results = load_domain_model_results()
    if not domain_model_results:
        print("警告：无法加载领域模型结果，将继续处理但不包含领域模型信息")

    # 读取测试样本
    test_samples = batch_read_test_samples(config.TEST_IMAGE_FOLDER)
    if not test_samples:
        print("无测试样本，程序退出")
        return

    # 获取模型参数和路径
    model_args = get_model_args_for_dataset(dataset_name)
    model_path = get_model_path_for_dataset(dataset_name)

    if not model_path:
        print(f"错误：未找到数据集 {dataset_name} 的模型路径")
        return

    # 确保输出目录存在
    os.makedirs(os.path.dirname(config.STAGE2_OUTPUT_FILE), exist_ok=True)

    # 初始化输出数据
    stage2_output = {
        "test_samples": [],
        "stage1_reference": all_label_summaries,
        "stage1_output2_reference": stage1_output2_summaries
    }
    if os.path.exists(config.STAGE2_OUTPUT_FILE):
        with open(config.STAGE2_OUTPUT_FILE, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
            stage2_output["test_samples"] = existing_data.get("test_samples", [])

    for idx, sample_info in enumerate(test_samples):
        sample_id = sample_info["sample_id"]
        sample_paths = sample_info["paths"]
        true_label = sample_info["true_label"]

        if len(sample_paths) >= 2:
            line_path, spec_path = sample_paths[0], sample_paths[1]
            heatmap_path = sample_paths[2] if len(sample_paths) >= 3 else None
        else:
            print(f"样本 {idx} 路径不足，跳过")
            continue

        # 跳过已处理的样本
        processed_sample_ids = [s["sample_id"] for s in stage2_output["test_samples"]]
        if sample_id in processed_sample_ids:
            print(f"\n测试样本 {idx + 1}/{len(test_samples)}，ID: {sample_id} 已处理，跳过")
            continue

        print(f"\n处理测试样本 {idx + 1}/{len(test_samples)}，ID: {sample_id}，真实标签: {true_label}")

        # 转换测试图片为Base64
        test_line_data = local_image_to_base64(line_path)
        test_spec_data = local_image_to_base64(spec_path)
        test_heatmap_data = local_image_to_base64(heatmap_path) if heatmap_path else None

        test_images_data = [test_line_data, test_spec_data]
        if test_heatmap_data:
            test_images_data.append(test_heatmap_data)

        if not all([test_line_data, test_spec_data]):
            print("测试图片转换失败，跳过")
            continue

        # 获取领域模型预测结果
        domain_model_result = get_domain_model_result_for_sample(domain_model_results, sample_id)

        # 第2.5轮：基于STAGE1_OUTPUT_FILE2特征进行分类
        label_summary_result, round2_5_response = process_round2_5(
            stage1_output2_summaries,
            test_images_data
        )

        # 获取模型分类结果
        model_result = load_model_and_get_prediction(sample_id, dataset_name)

        sample_results = {
            "sample_id": sample_id,
            "true_label": true_label,
            "round2_5_result": label_summary_result,
            "round2_5_response": round2_5_response,
            "model_result": model_result,
            "domain_model_result": domain_model_result,
            "thinking_chains": {},
            "final_result": {}
        }

        # 判断分类结果是否相等
        classification_equal = (label_summary_result == model_result)
        sample_results["classification_equal"] = classification_equal

        # 获取相似样本（使用feature模块）
        similar_sample_ids = get_similar_samples_by_feature_module(
            sample_id,
            dataset_name,
            model_path,
            model_args,
            k=config.SIMILAR_SAMPLE_NUM
        )

        # 加载相似样本图像
        similar_images_data = []
        if similar_sample_ids:
            similar_images_data = load_similar_sample_images(similar_sample_ids, dataset_name)
            sample_results["similar_sample_ids"] = similar_sample_ids
            sample_results["similar_images_count"] = len(similar_images_data)

        # Round 3: 生成三条分类思维链（包含相似样本）
        thinking_chains = generate_thinking_chains_with_similar_samples(
            all_label_summaries,
            test_images_data,
            similar_images_data
        )
        sample_results["thinking_chains"] = thinking_chains

        # Round 4: 最终分类（包含相似样本）
        final_result = process_final_classification_with_similar_samples(
            all_label_summaries,
            thinking_chains,
            domain_model_result,
            test_images_data,
            similar_images_data
        )

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
    """主函数：集成两个阶段的处理流程（添加新功能）"""

    # 检查是否已有第一阶段结果
    if os.path.exists(config.STAGE1_OUTPUT_FILE):
        print("检测到已有第一阶段结果，直接加载...")
        with open(config.STAGE1_OUTPUT_FILE, 'r', encoding='utf-8') as f:
            stage1_data = json.load(f)
        all_label_summaries = stage1_data.get("label_summaries_final", {})
    else:
        # 执行第一阶段
        all_label_summaries, selected_reference_ids = process_single_label_reference_samples()
        if not all_label_summaries:
            print("第一阶段处理失败，程序退出")
            return

    # 从配置或环境变量获取数据集名称
    dataset_name = getattr(config, 'DATASET_NAME', 'computer')

    # 执行第二阶段（使用新功能）
    process_test_samples_with_new_features(all_label_summaries, dataset_name)


if __name__ == "__main__":
    main()