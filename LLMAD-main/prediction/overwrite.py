import json

# 示例数据（你可以替换成从文件读取的）
with open('ArrowHead2_main.json', 'r') as outfile:
    data = json.load(outfile)

# 1. prediction 长度
pred_len = len(data["predictions"])


# 初始化计数器
gaicuo = 0
gaidui = 0
correct = 0
discover = 0
yizhi=0
# 遍历每个位置
for i in range(pred_len):
    pred = data["predictions"][i]
    label = data["labels"][i]
    dtw = data["dtw"][i]

    # if pred == label:
    #     correct+=1
    #
    # if dtw[0] != pred:
    #     yizhi  += 1
    #
    # if label == dtw[0] and pred != label:
    #     gaicuo += 1
    #
    # if label != dtw[0] and pred == label:
    #     gaidui +=1
    #
    # if label != dtw[0] and l
    # abel != dtw[1] and label != dtw[2] and pred == label:
    #     discover +=1

    # if pred == label and pred == dtw[0]:
    #     gaidui += 1
    #
    # if pred == dtw[0]:
    #     gaicuo += 1

    # if pred == label and pred == dtw[1] and dtw[0] != dtw[1]:
    #     gaidui += 1
    #
    # if pred == dtw[1] and dtw[0] != dtw[1]:
    #     gaicuo += 1

    # if pred == label and pred == dtw[2] and dtw[2] != dtw[1] and dtw[2] != dtw[0]:
    #     gaidui += 1
    #
    # if pred == dtw[2] and dtw[2] != dtw[1] and dtw[2] != dtw[0]:
    #     gaicuo += 1

    if pred != dtw[0] and pred != dtw[1] and pred != dtw[2]:
        gaicuo += 1

    if pred != dtw[0] and pred != dtw[1] and pred != dtw[2] and pred == label:
        gaidui += 1


# 输出结果
print(f"Prediction 长度: {pred_len}")
print(correct/pred_len)
print(f"Gaicuo: {gaicuo}")
print(f"Gaidui: {gaidui}")
print(f"Discover: {discover}")
print(yizhi/pred_len)