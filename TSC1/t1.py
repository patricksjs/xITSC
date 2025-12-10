# main.py (修改后的主程序)
import os
import json
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


def get_round3_prompt(all_label_summaries, test_images_data):
    """生成第3轮提示词 - 使用处理后的label特征进行分类"""

    # 构建多模态消息内容
    content = [
        {
            "type": "text",
            "text": f"""
# Feature Analysis Report:
This section provides an in-depth analysis of the features and patterns within each individual label:
{json.dumps(all_label_summaries, ensure_ascii=False)}

{prompt.prompt3}"""
        }
    ]

    # 添加测试样本图片
    if test_images_data and len(test_images_data) > 0:
        content.append({
            "type": "text",
            "text": "The test image to be classified is as follows:"
        })

        for i, image_data in enumerate(test_images_data):
            if image_data:  # 确保图片数据不为空
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_data}"}
                })

    return [{"role": "user", "content": content}]


def get_round4_prompt(all_label_summaries, thinking_chains, domain_model_result, test_images_data):
    """生成第4轮提示词 - 整合思维链得到最终结果"""

    # 构建多模态消息内容
    content = [
        {
            "type": "text",
            "text": f"""

Three classification thought chains:
{json.dumps(thinking_chains, ensure_ascii=False)}

# Feature Analysis Report:
This section provides an in-depth analysis of the features and patterns within each individual label:
{json.dumps(all_label_summaries, ensure_ascii=False)}

Domain Model Classification Results:
{json.dumps(domain_model_result, ensure_ascii=False)}

{prompt.prompt4}"""
        }
    ]

    # 添加测试样本图片
    if test_images_data and len(test_images_data) > 0:
        content.append({
            "type": "text",
            "text": "The test image to be classified is as follows:"
        })

        for i, image_data in enumerate(test_images_data):
            if image_data:  # 确保图片数据不为空
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_data}"}
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


def generate_thinking_chains(all_label_summaries, test_images_data):
    """Round3: 生成三条分类思维链"""
    print("执行第3轮：生成三条分类思维链...")

    thinking_chains = {}

    for i in range(3):
        print(f"  生成第 {i + 1} 条思维链...")

        # 直接构建消息而不调用辅助函数
        messages = get_round3_prompt(all_label_summaries, test_images_data)
        chain_result = gpt_chat(messages, [])

        if chain_result:
            thinking_chains[f"chain_{i + 1}"] = chain_result
            print(f"    第 {i + 1} 条思维链生成成功")
        else:
            thinking_chains[f"chain_{i + 1}"] = {"error": "生成失败"}
            print(f"    第 {i + 1} 条思维链生成失败")

    return thinking_chains


def process_final_classification(all_label_summaries, thinking_chains, domain_model_result, test_images_data):
    """Round4: 最终分类"""
    print("执行第4轮：最终分类...")

    # 直接构建消息而不调用辅助函数
    messages = get_round4_prompt(all_label_summaries, thinking_chains, domain_model_result, test_images_data)
    final_result = gpt_chat(messages, [])

    return final_result


def process_test_samples(all_label_summaries):
    """第二阶段：测试样本分析"""
    print("\n开始第二阶段：测试样本分析")

    if not all_label_summaries:
        print("无标签特征总结，程序退出")
        return

    # 加载领域模型结果
    domain_model_results = load_domain_model_results()
    if not domain_model_results:
        print("警告：无法加载领域模型结果，将继续处理但不包含领域模型信息")

    # 读取测试样本
    test_samples = batch_read_test_samples(config.TEST_IMAGE_FOLDER)
    if not test_samples:
        print("无测试样本，程序退出")
        return

    # 确保输出目录存在
    os.makedirs(os.path.dirname(config.STAGE2_OUTPUT_FILE), exist_ok=True)

    # 初始化输出数据
    stage2_output = {
        "test_samples": [],
        "stage1_reference": all_label_summaries
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

        sample_results = {
            "sample_id": sample_id,
            "true_label": true_label,
            "domain_model_result": domain_model_result,
            "thinking_chains": {},
            "final_result": {}
        }

        # Round 3: 生成三条分类思维链
        thinking_chains = generate_thinking_chains(all_label_summaries, test_images_data)
        sample_results["thinking_chains"] = thinking_chains

        # Round 4: 最终分类
        final_result = process_final_classification(
            all_label_summaries,
            thinking_chains,
            domain_model_result,
            test_images_data
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
    """主函数：集成两个阶段的处理流程"""

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

    # 执行第二阶段
    process_test_samples(all_label_summaries)


if __name__ == "__main__":
    main()