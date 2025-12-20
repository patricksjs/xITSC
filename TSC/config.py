# config.py
import os
import re

model_file_path = r"C:\Users\34517\Desktop\zuhui\xITSC\computer_transformer.json"
# 文件路径配置
ROOT_SPECTROGRAM_FOLDER = r"C:\Users\34517\Desktop\zuhui\xITSC\data\image1"
TEST_IMAGE_FOLDER = r"C:\Users\34517\Desktop\zuhui\xITSC\data\image1_test"
dataset = "computer"
# 第一阶段输出文件
STAGE1_OUTPUT_FILE = r"C:\Users\34517\Desktop\zuhui\xITSC\result\stage1.51.json"

# 第二阶段输出文件
STAGE2_OUTPUT_FILE = r"C:\Users\34517\Desktop\zuhui\xITSC\result\stage2.json"

# 分类结果输出文件
CLASSIFICATION_RESULTS_FILE = r"C:\Users\34517\Desktop\zuhui\xITSC\result\classification_results.json"

# 模型配置
GPT_MODEL = "gpt-5-mini"
#gpt-5-mini
OPENAI_API_KEY =

OPENAI_BASE_URL = "https://api.chatanywhere.tech/v1"
#OPENAI_BASE_URL ="https://api.agicto.cn/v1"
# 样本配置
CLASS_COUNT = 2
SAMPLES_PER_CLASS = 4
MANUAL_IDS = {0: [0,105, 16, 102],
              1: [204,246,247,220]
             }

# Transformer参数
NUM_LAYERS = 2
D_MODEL = 64
NHEAD = 8
DIM_FEEDFORWARD = 256
DROPOUT = 0.2

RESULT_PATTERNS = {
    # round3: 找到"result"和"score"之间的第一个数字
    'round3': re.compile(r'result(.*?)score', re.IGNORECASE),

    # round4_result: 找到"result"和"model"之间的第一个数字
    'round4_result': re.compile(r'result(.*?)model', re.IGNORECASE),

    # round4_model: 找到"model"和"rationale"之间的第一个数字
    'round4_model': re.compile(r'model(.*?)rationale', re.IGNORECASE),

}
