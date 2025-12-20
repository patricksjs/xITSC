# from fastdtw import fastdtw
#
# from dataloader_v2 import load_data
#
# dataset_name = "CBF"
# train_data_tensor, train_labels_tensor, _ = load_data(data_name=dataset_name)['train']
# test_data_tensor, test_labels_tensor, _ = load_data(data_name=dataset_name)['test']
#
# # 计算 DTW 距离并排序
#
# correct = 0
# count = 0
# for idx in range(len(test_data_tensor)):
#     series = test_data_tensor[idx].numpy()
#     distances = []
#
#
#     def scalar_euclidean(u, v):
#         return abs(u - v)
#
#
#     for train_idx in range(len(train_data_tensor)):
#         dist, _ = fastdtw(train_data_tensor[train_idx].flatten(), series.flatten(), dist=scalar_euclidean)
#         distances.append((dist, train_idx))
#
#     # 取距离最近的3个
#     distances.sort(key=lambda x: x[0])
#     top3_indices = [idx for _, idx in distances]
#     print("pred", train_labels_tensor[top3_indices[0]].item())
#     print("gt", test_labels_tensor[idx].item())
#
#     if train_labels_tensor[top3_indices[0]] == test_labels_tensor[idx]:
#         correct += 1
#         print("correct", correct)
#     count += 1
#
# print(correct / count)


from fastdtw import fastdtw
from dataloader_v2 import load_data

dataset_name = "EOGHorizontalSignal"
train_data_tensor, train_labels_tensor, _ = load_data(data_name=dataset_name)['train']
test_data_tensor, test_labels_tensor, _ = load_data(data_name=dataset_name)['test']

# === 配置 ===
MAX_TRAIN_PER_CLASS = 20
MAX_TEST_PER_CLASS = 50


# === 工具函数：按类别采样最多 N 个样本索引 ===
def sample_per_class(data_tensor, labels_tensor, max_per_class):
    class_to_indices = {}
    for idx, label in enumerate(labels_tensor):
        label_val = label.item()
        if label_val not in class_to_indices:
            class_to_indices[label_val] = []
        if len(class_to_indices[label_val]) < max_per_class:
            class_to_indices[label_val].append(idx)

    # 合并所有保留的索引（保持原始顺序）
    sampled_indices = []
    for indices in class_to_indices.values():
        sampled_indices.extend(indices)

    return sampled_indices, class_to_indices


# === 构建子采样后的训练集和测试集索引 ===
filtered_train_indices, _ = sample_per_class(
    train_data_tensor, train_labels_tensor, MAX_TRAIN_PER_CLASS
)
filtered_test_indices, test_class_dict = sample_per_class(
    test_data_tensor, test_labels_tensor, MAX_TEST_PER_CLASS
)

print(f"Train: original={len(train_data_tensor)}, filtered={len(filtered_train_indices)}")
print(f"Test:  original={len(test_data_tensor)}, filtered={len(filtered_test_indices)}")


# === DTW 距离函数 ===
def scalar_euclidean(u, v):
    return abs(u - v)


# === KNN 分类（1-NN）===
correct_first = 0
correct_second = 0
correct_third = 0
correct_forth = 0
correct_fifth = 0
count = 0

for test_idx in filtered_test_indices:  # 只遍历采样后的测试样本
    if test_labels_tensor[test_idx] >= 9:
        continue
    if test_idx == 2 :
        print(test_idx)
    series = test_data_tensor[test_idx].numpy()
    distances = []

    # 在采样后的训练集上计算 DTW
    for train_idx in filtered_train_indices:
        dist, _ = fastdtw(
            train_data_tensor[train_idx].flatten(),
            series.flatten(),
            dist=scalar_euclidean
        )
        distances.append((dist, train_idx))

    # 找最近邻（1-NN）
    distances.sort(key=lambda x: x[0])
    first_train_idx = distances[0][1]
    second_train_idx = distances[1][1]
    third_train_idx = distances[2][1]
    forth_train_idx = distances[3][1]
    fifth_train_idx = distances[4][1]
    top10 = [idx for _, idx in distances[:10]]
    pred_label = train_labels_tensor[top10]

    first_pred_label = train_labels_tensor[first_train_idx].item()
    second_pred_label = train_labels_tensor[second_train_idx].item()
    third_pred_label = train_labels_tensor[third_train_idx].item()
    forth_pred_label = train_labels_tensor[forth_train_idx].item()
    fifth_pred_label = train_labels_tensor[fifth_train_idx].item()
    true_label = test_labels_tensor[test_idx].item()

    print("pred", pred_label)
    print("gt", true_label)

    if first_pred_label == true_label:
        correct_first += 1
    elif second_pred_label == true_label:
        correct_second += 1
    elif third_pred_label == true_label:
        correct_third += 1
    elif forth_pred_label == true_label:
        correct_forth += 1
    elif fifth_pred_label == true_label:
        correct_fifth += 1
    count += 1

    print("count", count)
    print("correct_first", correct_first)
    print("correct_second", correct_second)
    print("correct_third", correct_third)
    print("correct_forth", correct_forth)
    print("correct_fifth", correct_fifth)
