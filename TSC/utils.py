# utils.py (简化提取逻辑)
import base64
import json
import os
import time
import re
from PIL import Image
from io import BytesIO
from openai import OpenAI
import config

# 初始化OpenAI客户端
client = OpenAI(
    api_key=config.OPENAI_API_KEY,
    base_url=config.OPENAI_BASE_URL
)


def local_image_to_base64(image_path, image_format="PNG"):
    """将本地图片转换为Base64编码"""
    try:
        with Image.open(image_path) as img:
            if image_format == "JPEG" and img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            buffer = BytesIO()
            img.save(buffer, format=image_format, quality=90)
            buffer.seek(0)

            base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return f"data:image/{image_format.lower()};base64,{base64_str}"

    except Exception as e:
        print(f"本地图片处理失败：{str(e)}")
        return None


def build_multimodal_content_with_interleaved_text_images(text_blocks, image_blocks):
    """构建文本和图片穿插的多模态内容"""
    content = []

    # 确保文本块和图片块数量匹配
    if len(text_blocks) != len(image_blocks):
        raise ValueError(
            f"round1,文本块数量（{len(text_blocks)}）与图片块数量（{len(image_blocks)}）不相等"
        )
    for i in range(len(text_blocks)):
        # 添加文本
        content.append({"type": "text", "text": text_blocks[i]})
        # 添加对应的图片
        for img_data in image_blocks[i]:
            if img_data:  # 确保图片数据不为空
                content.append({"type": "image_url", "image_url": {"url": img_data}})

    return content

def gpt_chat(content, conversation, max_retries=3):
    """发送聊天请求（支持图片，强制JSON输出），JSON格式错误时自动重试"""
    if isinstance(content, list):
        for item in content:
            if item["type"] == "text":
                item[
                    "text"] += "\n\nStrictly output the content in the JSON format required by the task. Return only the JSON content without any additional text, explanations, or code block markers."
                break
        user_message = {"role": "user", "content": content}
    else:
        content += "\n\nStrictly output the content in the JSON format required by the task. Return only the JSON content without any additional text, explanations, or code block markers."
        user_message = {"role": "user", "content": content}

    retry_count = 0
    while retry_count < max_retries:
        try:
            print(f"发送请求到API (尝试 {retry_count + 1}/{max_retries})...")
            response = client.chat.completions.create(
                model=config.GPT_MODEL,
                temperature=0.1,
                messages=conversation + [user_message],
                stream=False
            )
            raw_output = response.choices[0].message.content.strip()
            print(f"API原始输出内容：\n{raw_output}\n")

            if not raw_output:
                print("API返回空输出，重试...")
                retry_count += 1
                time.sleep(2)
                continue

            # 处理可能的代码块标记
            cleaned_output = raw_output
            if raw_output.startswith("```json"):
                cleaned_output = raw_output[7:]
                if cleaned_output.endswith("```"):
                    cleaned_output = cleaned_output[:-3]
                cleaned_output = cleaned_output.strip()
            elif raw_output.startswith("```"):
                cleaned_output = raw_output[3:]
                if cleaned_output.endswith("```"):
                    cleaned_output = cleaned_output[:-3]
                cleaned_output = cleaned_output.strip()

            # 尝试解析JSON
            try:
                json_result = json.loads(cleaned_output)
                print("JSON解析成功！")
                return json_result

            except json.JSONDecodeError as e:
                print(f"JSON解析失败 (尝试 {retry_count + 1}/{max_retries}): {str(e)[:200]}")
                print(f"清理后的输出：{cleaned_output}")
                retry_count += 1
                if retry_count < max_retries:
                    print(f"等待2秒后重试...")
                    time.sleep(5)
                continue

        except Exception as e:
            print(f"API请求失败 (尝试 {retry_count + 1}/{max_retries}): {str(e)[:250]}")
            retry_count += 1
            if retry_count < max_retries:
                time.sleep(2)

    print("达到最大重试次数，请求失败")
    return {"error": "max_retries_exceeded", "message": "无法获取有效的JSON响应"}
'''
def gpt_chat(content, conversation, max_retries=3):
    """发送聊天请求（支持图片，强制JSON输出），JSON格式错误时自动重试"""
    if isinstance(content, list):
        for item in content:
            if item["type"] == "text":
                item[
                    "text"] += "\n\nStrictly output the content in the JSON format required by the task. Return only the JSON content without any additional text, explanations, or code block markers."
                break
        user_message = {"role": "user", "content": content}
    else:
        content += "\n\nStrictly output the content in the JSON format required by the task. Return only the JSON content without any additional text, explanations, or code block markers."
        user_message = {"role": "user", "content": content}

    retry_count = 0
    while retry_count < max_retries:
        try:
            print(f"发送请求到API (尝试 {retry_count + 1}/{max_retries})...")
            response = client.chat.completions.create(
                model=config.GPT_MODEL,
                temperature=0.1,
                messages=conversation + [user_message],
                stream=False
            )
            raw_output = response.choices[0].message.content.strip()
            print(f"API原始输出内容：\n{raw_output}\n")

            if not raw_output:
                print("API返回空输出，重试...")
                retry_count += 1
                time.sleep(2)
                continue

            # 处理可能的代码块标记
            cleaned_output = raw_output
            if raw_output.startswith("```json"):
                cleaned_output = raw_output[7:]
                if cleaned_output.endswith("```"):
                    cleaned_output = cleaned_output[:-3]
                cleaned_output = cleaned_output.strip()
            elif raw_output.startswith("```"):
                cleaned_output = raw_output[3:]
                if cleaned_output.endswith("```"):
                    cleaned_output = cleaned_output[:-3]
                cleaned_output = cleaned_output.strip()

            # 尝试解析JSON
            try:
                json_result = json.loads(cleaned_output)
                print("JSON解析成功！")
                return json_result

            except json.JSONDecodeError as e:
                print(f"JSON解析失败 (尝试 {retry_count + 1}/{max_retries}): {str(e)[:200]}")
                print(f"清理后的输出：{cleaned_output}")
                retry_count += 1
                if retry_count < max_retries:
                    print(f"等待2秒后重试...")
                    time.sleep(5)
                continue

        except Exception as e:
            print(f"API请求失败 (尝试 {retry_count + 1}/{max_retries}): {str(e)[:250]}")
            retry_count += 1
            if retry_count < max_retries:
                time.sleep(2)

    print("达到最大重试次数，请求失败")
    return {"error": "max_retries_exceeded", "message": "无法获取有效的JSON响应"}
'''

def extract_classification_results(thinking_chains, final_result, true_label):
    """从第三轮和第四轮结果中提取分类结果"""
    results = {
        "true_label":true_label,
        "round3_1": None,
        "round3_2": None,
        "round3_3": None,
        "round4_r": None,
        "round4_m": None
    }

    # 提取第三轮的三条思维链结果
    for i in range(1, 4):
        chain_key = f"chain_{i}"
        if chain_key in thinking_chains:
            chain_data = thinking_chains[chain_key]
            # 提取result字段
            results[f"round3_{i}"] = extract_field_value(chain_data, 'result')

    # 提取第四轮结果
    if final_result and isinstance(final_result, dict):
        # 提取result字段
        results["round4_r"] = extract_field_value(final_result, 'result')
        # 提取model字段
        results["round4_m"] = extract_field_value(final_result, 'model')

    return results


def extract_field_value(data, field_name):
    """提取指定字段的值中的数字"""
    if not data or not isinstance(data, dict):
        return None

    # 首先尝试直接访问字段
    if field_name in data:
        field_value = data[field_name]
        return extract_number_from_any_format(field_value)

    # 如果直接访问失败，使用正则表达式在整个JSON字符串中搜索
    try:
        json_str = json.dumps(data)

        # 根据字段名选择对应的正则表达式
        if field_name == 'result':
            pattern = config.RESULT_PATTERNS['round3']
        elif field_name == 'model':
            pattern = config.RESULT_PATTERNS['round4_model']
        else:
            return None

        match = pattern.search(json_str)
        if match:
            # 提取匹配到的整个值部分
            value_str = match.group(1)
            return extract_number_from_any_format(value_str)
    except Exception as e:
        print(f"使用正则表达式提取字段 {field_name} 失败: {e}")

    return None


def extract_number_from_any_format(value):
    """从任何格式的值中提取数字"""
    if value is None:
        return None

    # 如果是列表，取第一个元素
    if isinstance(value, list) and len(value) > 0:
        return extract_number_from_string(str(value[0]))
    # 如果是数字，直接返回
    elif isinstance(value, (int, float)):
        return int(value)
    # 如果是字符串，提取数字
    elif isinstance(value, str):
        return extract_number_from_string(value)

    return None


def extract_number_from_string(text):
    """从字符串中提取第一个数字"""
    if not text:
        return None

    # 使用正则表达式匹配第一个数字
    match = re.search(r'(\d+)', str(text))
    if match:
        return int(match.group(1))
    return None


def save_classification_results(sample_id, classification_results, file_path):
    """保存分类结果到文件"""
    # 如果文件已存在，加载现有数据
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        except:
            existing_data = {"results": []}
    else:
        existing_data = {"results": []}

    # 检查是否已存在该样本的结果
    sample_exists = False
    for item in existing_data["results"]:
        if item.get("sample_id") == sample_id:
            # 更新现有结果
            item.update(classification_results)
            sample_exists = True
            break

    # 如果样本不存在，添加新结果
    if not sample_exists:
        result_entry = {"sample_id": sample_id}
        result_entry.update(classification_results)
        existing_data["results"].append(result_entry)

    # 保存文件
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, indent=4, ensure_ascii=False)

    print(f"分类结果已保存到: {file_path}")


# 其他函数保持不变...
def auto_load_reference_images(root_folder):
    """自动加载参考样本图片路径"""
    reference_images = {}
    selected_ids = []
    dataset = config.dataset

    for class_idx in range(config.CLASS_COUNT):
        class_folder = os.path.join(root_folder, f"{dataset}_{class_idx}")
        if not os.path.exists(class_folder):
            print(f"警告：类别文件夹 {class_folder} 不存在，跳过")
            continue

        # 提取所有样本ID，并筛选出三张图片齐全的有效样本ID
        all_valid_sample_ids = []
        for filename in os.listdir(class_folder):
            if filename.endswith(".png") and "sample" in filename:
                # 从文件名中提取样本ID
                if filename.startswith("plot_sample"):
                    sample_id = filename.replace("plot_sample", "").replace(".png", "")
                elif filename.startswith("shap_sample"):
                    sample_id = filename.replace("shap_sample", "").replace(".png", "")
                elif filename.startswith("STFT_sample"):
                    sample_id = filename.replace("STFT_sample", "").replace(".png", "")
                else:
                    continue

                # 校验该样本的三张图是否齐全
                plot_path = os.path.join(class_folder, f"plot_sample{sample_id}.png")
                stft_path = os.path.join(class_folder, f"STFT_sample{sample_id}.png")
                shap_path = os.path.join(class_folder, f"shap_sample{sample_id}.png")

                if os.path.exists(plot_path) and os.path.exists(stft_path) and os.path.exists(shap_path):
                    all_valid_sample_ids.append(sample_id)

        all_valid_sample_ids = list(set(all_valid_sample_ids))  # 去重
        if not all_valid_sample_ids:
            raise FileNotFoundError(f"错误：类别 {dataset}_{class_idx} 下无任何图片齐全的有效样本")

        print(f"类别 {dataset}_{class_idx} 下所有图片齐全的样本ID：{all_valid_sample_ids}")

        # 选择样本
        if config.MANUAL_IDS and class_idx in config.MANUAL_IDS:
            manual_selected = [str(id) for id in config.MANUAL_IDS[class_idx]]
            init_selected = [id for id in manual_selected if id in all_valid_sample_ids]
            invalid_manual = [id for id in manual_selected if id not in all_valid_sample_ids]
            if invalid_manual:
                print(f"警告：类别 dataset_{class_idx} 手动指定的ID {invalid_manual} 图片不齐全，将自动补选")
        else:
            init_selected = all_valid_sample_ids[:config.SAMPLES_PER_CLASS]

        # 计算需要补选的样本数量
        need_supplement = max(0, config.SAMPLES_PER_CLASS - len(init_selected))
        supplement_ids = []
        if need_supplement > 0:
            candidate_ids = [id for id in all_valid_sample_ids if id not in init_selected]
            supplement_ids = candidate_ids[:need_supplement]
            print(f"类别 dataset_{class_idx} 需要补选 {need_supplement} 个样本，补选ID：{supplement_ids}")

        # 最终选中的有效样本ID
        final_selected = init_selected + supplement_ids
        final_selected = final_selected[:config.SAMPLES_PER_CLASS]

        if not final_selected:
            print(f"警告：类别 dataset_{class_idx} 无有效选中样本，跳过")
            continue

        selected_ids.extend([int(id_str) for id_str in final_selected])
        print(f"类别 dataset_{class_idx} 最终选中样本ID：{final_selected}")

        # 收集最终选中样本的图片路径
        line_charts = []
        time_frequency = []
        heatmap = []
        for sample_id in final_selected:
            plot_path = os.path.join(class_folder, f"plot_sample{sample_id}.png")
            stft_path = os.path.join(class_folder, f"STFT_sample{sample_id}.png")
            shap_path = os.path.join(class_folder, f"shap_sample{sample_id}.png")

            line_charts.append(plot_path)
            time_frequency.append(stft_path)
            heatmap.append(shap_path)

        reference_images[f"dataset_{class_idx}"] = {
            "line_charts": line_charts,
            "time_frequency": time_frequency,
            "heatmap": heatmap
        }

    return reference_images, selected_ids


def batch_read_test_samples(folder_path):
    """批量读取测试样本的线图、时频图和热力图路径，并提取真实标签"""
    sample_pairs = []
    processed_ids = set()

    # 假设测试样本也按照类别文件夹组织，遍历所有类别文件夹
    for class_folder_name in os.listdir(folder_path):

        class_folder = os.path.join(folder_path, class_folder_name)
        if not os.path.isdir(class_folder):
            continue

        print(f"正在读取测试样本类别文件夹: {class_folder_name}")

        # 从文件夹名称中提取真实标签（假设文件夹名为 dataset_classidx 格式）
        try:
            true_label = int(class_folder_name.split('_')[-1])
        except:
            true_label = None
            print(f"警告：无法从文件夹名 {class_folder_name} 中提取真实标签")

        for filename in os.listdir(class_folder):
            if not filename.endswith(".png") or "sample" not in filename:
                continue

            # 从文件名中提取样本ID
            if filename.startswith("plot_sample"):
                sample_id = filename.replace("plot_sample", "").replace(".png", "")
            else:
                continue

            if sample_id in processed_ids:
                continue

            # 构建三张图片的路径
            plot_path = os.path.join(class_folder, f"plot_sample{sample_id}.png")
            stft_path = os.path.join(class_folder, f"STFT_sample{sample_id}.png")
            shap_path = os.path.join(class_folder, f"shap_sample{sample_id}.png")

            if os.path.exists(plot_path) and os.path.exists(stft_path) and os.path.exists(shap_path):
                # 返回样本路径和对应的真实标签
                sample_pairs.append({
                    "sample_id": sample_id,
                    "paths": [plot_path, stft_path, shap_path],
                    "true_label": true_label
                })
                processed_ids.add(sample_id)
                print(f"成功匹配测试样本 {sample_id}，来自类别 {class_folder_name}，真实标签: {true_label}")

    # 按样本ID排序
    sample_pairs.sort(key=lambda x: int(x["sample_id"]))
    print(f"共读取测试样本 {len(sample_pairs)} 个")
    return sample_pairs