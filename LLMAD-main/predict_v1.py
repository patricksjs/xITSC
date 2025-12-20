import os
import base64
from openai import OpenAI
from sklearn.metrics import accuracy_score, classification_report
import time
import numpy as np
# -----------------------------
# 配置
# -----------------------------
gpt_model = "gpt-4o"
OPENAI_API_KEY = "sk-VdhI38Ualpqsc1yxyYN3Is82AWrNOtvhMjgxUihamFZSAf7k"
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://api.chatanywhere.tech/v1"
)

root_dir = "plots/Computers_test_nolabel"
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


def first_round():
    # 构建系统提示（关键：强制输出格式）
    SYSTEM_PROMPT = f"""
    You are an expert in time series analysis. 
    I will give you a line plot of a time series from the 'Computers' dataset.
    {subset_description}
    Your task is to classify it.
    Respond ONLY with the class label number (0 or 1). 
    Do not output any other text, explanation, or punctuation.
    """

    # -----------------------------
    # 图像编码函数
    # -----------------------------
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    # -----------------------------
    # 遍历图像并调用 GPT-4o
    # -----------------------------
    all_preds = []
    all_labels = []

    total = 0
    for class_dir in class_dirs:
        true_label = class_to_label[class_dir]
        class_path = os.path.join(root_dir, class_dir)
        image_files = sorted(
            [f for f in os.listdir(class_path) if f.endswith(".png")],
            key=lambda x: int(x.split(".")[0])
        )

        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)
            base64_image = encode_image(img_path)

            try:
                response = client.chat.completions.create(
                    model="gpt-4o",  # 或 "gpt-4o-mini" 更便宜
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    temperature=0.2
                )

                pred_text = response.choices[0].message.content.strip()
                # 尝试解析为整数
                try:
                    pred = int(pred_text)
                    if pred in valid_labels:
                        all_preds.append(pred)
                    else:
                        print(f"⚠️ Invalid prediction: {pred_text} (out of range), fallback to 0")
                        all_preds.append(0)
                except ValueError:
                    print(f"⚠️ Failed to parse prediction: '{pred_text}', fallback to 0")
                    all_preds.append(0)

                all_labels.append(true_label)
                total += 1
                print(f"[{total}] True: {true_label}, Pred: {all_preds[-1]}")

                # 避免 API 限速（可选）
                time.sleep(0.5)

            except Exception as e:
                print(f"❌ Error on {img_path}: {e}")
                # 可选择跳过或填默认值
                all_preds.append(0)
                all_labels.append(true_label)

    return all_preds, all_labels





all_preds, all_labels = first_round()
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
