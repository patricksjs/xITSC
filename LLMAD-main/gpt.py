import time
import json
import base64
from PIL import Image
from io import BytesIO
from openai import OpenAI

# 配置OpenAI客户端（保持不变）
gpt_model = "gpt-4o"
gpt_model_2 = "deepseek-chat"
OPENAI_API_KEY = "sk-gLJpO9I10cMfjn0PYz80SSwELl84fmTyKjhYlMUwkyANTfpf"
client = OpenAI(api_key=OPENAI_API_KEY, base_url="https://api.chatanywhere.tech/v1")

background = ("This data is derived from one of the Computers in Cardiology challenges, an annual competition that "
              "runs with the conference series of the same name and is hosted on physionet. Data is taken from ECG "
              "data for multiple torso-surface sites. There are 4 classes (4 different people).")


def local_image_to_base64(image_path, image_format="JPEG"):
    try:
        with Image.open(image_path) as img:
            if image_format == "JPEG" and img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            buffer = BytesIO()
            img.save(buffer, format=image_format, quality=90)
            buffer.seek(0)

            # 编码为Base64字符串，并添加API要求的前缀
            base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return f"data:image/{image_format.lower()};base64,{base64_str}"

    except Exception as e:
        print(f"本地图片处理失败：{str(e)}")
        return None


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def build_multimodal_content(text, image_data_list=None):
    content = [{"type": "text", "text": text}]

    if image_data_list:
        for img_data in image_data_list:
            # 判断是Base64字符串（本地图片）还是在线URL
            if img_data.startswith("data:image/"):
                # Base64格式：直接使用
                content.append({"type": "image_url", "image_url": {"url": img_data}})
            else:
                # 在线URL格式：按原逻辑处理
                content.append({"type": "image_url", "image_url": {"url": img_data}})

    return content


def gpt_chat(content, conversation, max_retries=3):
    """发送聊天请求（支持本地图片的Base64编码）"""

    # print("p", conversation)
    retry_count = 0
    while retry_count < max_retries:
        try:
            if isinstance(content, list):
                user_message = {"role": "user", "content": content}
            else:
                user_message = {"role": "user", "content": content}

            response = client.chat.completions.create(
                model=gpt_model,
                temperature=0.2,
                messages=conversation + [user_message],
                stream=False
            )
            return response.choices[0].message.content

        except Exception as e:
            error_msg = str(e)[:250]
            print(f"API请求失败 (尝试 {retry_count + 1}/{max_retries}): {error_msg}")
            retry_count += 1
            if retry_count < max_retries:
                time.sleep(5)
    print("已达到最大重试次数，请求失败。")
    return None


def gpt_chat_3(content, max_retries=3):
    """发送聊天请求（支持本地图片的Base64编码）"""

    # print(content)
    retry_count = 0
    while retry_count < max_retries:
        try:
            if isinstance(content, list):
                user_message = {"role": "user", "content": content}
            else:
                user_message = {"role": "user", "content": content}

            response = client.chat.completions.create(
                model=gpt_model_2,
                temperature=0.2,
                messages=[user_message],
                stream=False
            )
            return response.choices[0].message.content

        except Exception as e:
            error_msg = str(e)[:250]
            print(f"API请求失败 (尝试 {retry_count + 1}/{max_retries}): {error_msg}")
            retry_count += 1
            if retry_count < max_retries:
                time.sleep(5)
    print("已达到最大重试次数，请求失败。")
    return None


# -------------------------- 多轮对话函数（保持不变） --------------------------
def round_1(conversation, prompt, image_data_list=None):
    if image_data_list:
        content = build_multimodal_content(prompt, image_data_list)
    else:
        content = prompt
    round1_answer = gpt_chat(content, conversation)
    conversation.append({"role": "user", "content": content})
    conversation.append({"role": "assistant", "content": round1_answer})
    return conversation


def round_3(conversation, prompt, image_data_list=None):
    if image_data_list:
        content = build_multimodal_content(prompt, image_data_list)
    else:
        content = prompt
    round2_answer = gpt_chat(content, conversation)
    conversation.append({"role": "user", "content": content})
    conversation.append({"role": "assistant", "content": round2_answer})
    return conversation


def round_4(prompt1, prompt2, prompt3, prompt4):
    content = prompt1 + prompt2 + prompt3 + prompt4
    round4_answer = gpt_chat_3(content)

    return round4_answer


# -------------------------- 主函数（使用本地图片路径） --------------------------
if __name__ == "__main__":
    # 1. 替换为你的本地图片路径（支持JPG/PNG）
    local_image_paths1 = [

        r"spectrogram_plots/cincecgtorso_class_1/cincecgtorso_class_1_sample_596_line.png",
        r"spectrogram_plots/cincecgtorso_class_1/cincecgtorso_class_1_sample_596.png",
        r"spectrogram_plots/cincecgtorso_class_1/cincecgtorso_class_1_sample_596_heatmap.png"
    ]
    local_image_paths2 = [
        r"spectrogram_plots/cincecgtorso_class_1/cincecgtorso_class_1_sample_39_line.png",

        r"spectrogram_plots/cincecgtorso_class_1/cincecgtorso_class_1_sample_220_line.png",

        r"spectrogram_plots/cincecgtorso_class_1/cincecgtorso_class_1_sample_532_line.png"
    ]
    local_image_paths3 = [

        r"spectrogram_plots/cincecgtorso_class_0/cincecgtorso_class_0_sample_1310_line.png"
    ]

    # 2. 将本地图片转换为Base64编码（API可识别）
    image_data_list1 = []
    image_data_list2 = []
    image_data_list3 = []

    for img_path in local_image_paths1:
        # 若图片是PNG，第二个参数传"PNG"；JPG/JPEG传"JPEG"
        if img_path.lower().endswith(".png"):
            base64_str = local_image_to_base64(img_path, image_format="PNG")
        else:
            base64_str = local_image_to_base64(img_path, image_format="JPEG")

        if base64_str:
            image_data_list1.append(base64_str)
        else:
            print(f"跳过无效图片：{img_path}")
    for img_path in local_image_paths2:
        # 若图片是PNG，第二个参数传"PNG"；JPG/JPEG传"JPEG"
        if img_path.lower().endswith(".png"):
            base64_str = local_image_to_base64(img_path, image_format="PNG")
        else:
            base64_str = local_image_to_base64(img_path, image_format="JPEG")

        if base64_str:
            image_data_list2.append(base64_str)
        else:
            print(f"跳过无效图片：{img_path}")
    for img_path in local_image_paths3:
        # 若图片是PNG，第二个参数传"PNG"；JPG/JPEG传"JPEG"
        if img_path.lower().endswith(".png"):
            base64_str = local_image_to_base64(img_path, image_format="PNG")
        else:
            base64_str = local_image_to_base64(img_path, image_format="JPEG")

        if base64_str:
            image_data_list3.append(base64_str)
        else:
            print(f"跳过无效图片：{img_path}")
    # 3. 多轮对话（传入Base64编码的本地图片）
    conversation_1 = []

    print("=== 第一轮对话 ===")
    first_prompt = background + ("#### 2. Provided Visual Information\nI have input an image to "
                                 "you. This image contains 3 types of plots: a line plot and a time-frequency plot. The line "
                                 "plot is obtained by converting 1D time series data into 2D visualization; its x-axis represents "
                                 "the time steps of the time series, and the y-axis represents the values of the time series at "
                                 "corresponding time steps. The time-frequency plot reflects the distribution of the time series' "
                                 "frequency components across different time steps; its x-axis is time steps, y-axis is frequency, "
                                 "and the color intensity represents the energy or amplitude of the corresponding time-frequency "
                                 "component. The third graph is a heat map, representing the time-frequency graph regions that "
                                 "the black box model considers helpful for classification\n"
                                 "#### 3. Tasks to Perform\nDescribe the overall characteristics of the line graph and the "
                                 "time-frequency graph."
                                 "Locate the positions of the red areas in the heat map in the line graph and time-frequency graph, "
                                 "and describe the characteristics of these areas.")

    conversation_1 = round_1(conversation_1, first_prompt, image_data_list1)
    first_round_output = conversation_1[-1]
    conversation_1 = []
    conversation_1.append(first_round_output)
    print("第一轮回答：")
    print(first_round_output["content"])
    print("-" * 80)

    print("\n=== 第二轮对话 ===")
    second_prompt = background + ("Black Box Model Result: Class 1, Category Logits: [-11.4429,  13.9403,   1.3181,  -1.4422], Classification Accuracy: 95%"
                     "I will provide another 3 images of the Class 1 to help you compare"
                     "Please think step by step:\n"
                     "Based on the previous descriptions of the images to be classified, evaluate the degree of "
                     "similarity between them and these 3 images of Class 1"
                     "– Interpret the Model’s Results: [Evaluate the model’s classification result and logits. Assess"
                     "the confidence level of the model’s prediction and how well it aligns with the observed time"
                     "series patterns.]"
                     "- Make a final decision and interpretation: [Please choose one from "
                     "affirmative/skeptical/against as the evaluation of the judgment on the black box model. Based "
                     "on the background dataset knowledge and the heatmap feature, describe the previous sample and "
                     "explain the reason why it belongs to Class 1]")
    conversation_1 = round_1(conversation_1, second_prompt, image_data_list2)
    second_round_output = conversation_1[-1]
    print("第二轮回答：")
    print(second_round_output["content"])
    print("-" * 80)

    third_prompt = ("### 2. Typical Patterns of each category\n"
                    # "Class 0:"
                    # "- **Segments**: "
                    # "- **Low-activity segments**: Consist of repeated low values (e.g., -0.3963, -0.2809)."
                    # "- **High-activity segments**: Sudden spikes or sustained high values (e.g., 2.7052, 4.1319)."
                    # "- **Features**: "
                    # "- Predominantly flat with occasional abrupt spikes."
                    # "- Spikes are short-lived and return to baseline quickly."
                    # "- Low variability outside of spikes."
                    # "- **Differences with other class**: "
                    # "- Less frequent and less intense spikes compared to Class 1."
                    # "- Baseline values are consistently low with minimal fluctuation."
                    # "- **Explanation**: "
                    # "- The time series for Class 0 (Desktop) shows minimal activity with rare, abrupt energy "
                    # "spikes, likely corresponding to intermittent high-power usage events (e.g., turning on/off or "
                    # "heavy computation)."
                    # "Class 1:"
                    # "- **Segments**: Primary ultra-stable baseline (-0.4526±0.0001) occupying >85% of timeline, "
                    # "with secondary micro-fluctuations (-0.1422±0.01) preceding spikes. Spikes (1.0-3.5 range) last "
                    # "2-15 timesteps."
                    # "- **Features**: Baseline shows near-perfect stability during inactive periods. Spikes have steep "
                    # "attack/decay (<2 timesteps) with no ramping. Micro-fluctuations consistently precede major "
                    # "spikes by 2-5 timesteps."
                    # "- **Differences with other class**: Compared to Class 0, Class 1 maintains sub-0.1% baseline "
                    # "variation during inactive periods versus Class 0's 5% variation. Spike magnitudes show tighter "
                    # "clustering (1.4-3.3 IQR vs Class 0's 0.8-2.2)."
                    # "- **Explanation**: Matches laptop power management with deep sleep states (ultra-stable "
                    # "baseline), brief wake-up pre-spike fluctuations (OS checks), and sudden high-power demands ("
                    # "CPU/GPU activation)."
                    "Class 0:"
                    "**Segments:**"  
                    "- Comprised of alternating segments showcasing rapid changes in frequency and amplitude " 
                    "**Features:**"  
                    "- High amplitude peaks appear periodically, prominently interrupting baseline signals"  
                    "- Contrast between distinct sharp peaks and longer flat areas indicating sudden transitions " 
                    "**Differences with other classes:** " 
                    "- Exhibits more pronounced peaks and valleys, amplifying signal shifts compared to other classes"  
                    "- Significantly higher amplitude variance than Categories 2 and 3"  
                    "**Explanation:**"  
                    "- Variability in amplitude and frequency within this category denotes dynamic signal behavior, possibly reflecting different physiological states or sensor misalignment."
                    "Class 1:"
                    "**Segments:** This category consistently showcases minor uniform fluctuations, primarily in the lower amplitude range, indicating stable signal patterns."
                    "**Features:** " 
                    "- Predominantly characterized by lower amplitude values with minor fluctuations."
                    "- Minimal overall variation within each segment, reinforcing a consistent pattern."
                    "- Predominantly showcases consistent low-frequency oscillations and lack of high-frequency noise."  
                    "**Differences with other class:** " 
                    "- Lacks significant positive peaks or pronounced negative variations, unlike Class 0 and Class 3, where the signal shows more abrupt changes."
                    "- Compared to Class 3, Class 1 maintains a consistent low baseline without significant excursions into positive or high values, presenting a much more stable progression. " 
                    "**Explanation:** Class 1 is indicative of stable ECG patterns with consistent, subtle negative peaks, suggesting uniform heart rhythms and electrode stability without significant amplitude changes. This steadiness is reflective of a baseline signal with only mild, controlled deviations, distinguishing it from the more variant behaviors seen in other classes. The consistently low amplitude and lack of significant excursions help differentiate it from more variable classes."
                    "Class 2:"
                    "**Segments:** This class can be segmented into consistent segments with low variability. " 
                    "**Features:** Predominant features include sustained trends, significant baseline activity with fewer drastic peaks or troughs. The sequences seem closer to a baseline with limited significant deviation.  "
                    "**Differences with other class:** Class 2 displays more stable sequences compared to the pronounced fluctuations of Classes 0 and 1, and its overall level is different from the elevated nature of Class 3. " 
                    "**Explanation:** The stability and consistency of Class 2 is reminiscent of relaxed or non-stress conditions in an ECG waveform."
                    "Class 3"
                    "**Segments:** Class 3 exhibits extended periods of low-amplitude outputs, characterized by a distinct pattern of consistent small oscillations that suggest a rhythmic cardiac output with minimal external disturbances. These segments may occasionally display subtle dips that quickly revert to baseline levels, underscoring a stable cardiac rhythm that shows little variability or physiological disruptions across the recording period."
                    "**Features:** Class 3 is marked by notably low mean signal values and minimal amplitude fluctuations, demonstrating a cradling of the cardiac signal indicative of a restful physiological state. This pattern suggests a balance in heart rate and overall cardiovascular stability, reflecting a state of low exertion, markedly different from the higher energy outputs observed in more dynamic classes."
                    "**Differences with other classes:** In contrast to Class 1, Class 3 reveals significantly muted amplitude fluctuations, without the notable peaks and troughs that characterize Class 1's direct and erratic oscillatory behavior. Furthermore, it deviates from Class 2 through its consistently stable signal values, devoid of abrupt negative spikes, which often signify the active physiological disturbances present in Class 2."
                    "**Explanation:** The consistent absence of marked peaks and troughs within Class 3 reinforces its classification as indicative of stable cardiac functionality, implying the subject is likely in a state of restful activity or minimized physiological exertion. This low-amplitude profile sharply contrasts with other classes that reflect more active or dynamic physiological states, thereby providing a distinct signature for classification purposes of varied cardiac conditions."
                    "####  3. Tasks to Perform\n"
                    "Please think step by step:\n"
                    "1. Describe the overall characteristics of given line graph"
                    "2. Based on the background dataset knowledge make an initial classification decision. "
                    "Provide a brief explanation for this decision."
                    "3. Review the feature of the given image and compare with the typical patterns of each category."
                    "4. Make your final classification decision and explain the reason")

    conversation_3 = []
    conversation_3 = round_3(conversation_3, third_prompt, image_data_list3)
    third_round_output = conversation_3[-1]
    print("第三轮回答：")
    print(third_round_output["content"])

    final_prompt = background + ("####  2. Tasks to Perform\n"
                    "I will give you two classification explanations. The first one is an explanation based on the "
                    "black box model, and the second one is an explanation based on data features. ")

    final_task = ("\nPlease integrate "
                  "the two classifications and explanations, and in combination with the background information of "
                  "the dataset, provide a final classification, which should include: "
                  "Features consistent with the selected category"
                  ", whether the judgment of the black box model is reasonable, and the explanation of the "
                  "final classification.")

    second_round_output = "\n** Explanation by the black box model: **\n" + second_round_output["content"]
    third_round_output = "\n** Explanation by the typical feature of each category: **\n" + third_round_output[
        "content"]
    conversation_4 = round_4(final_prompt, second_round_output, third_round_output, final_task)
    final_round_output = conversation_4
    print("第四轮回答：")
    print(final_round_output)
