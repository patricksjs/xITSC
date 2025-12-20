def find_extreme_indices(lists):
    """
    输入：lists，包含三个等长（或不等长）数值列表的列表。
    输出：{
        "max_indices": [[...], [...], [...]],  # 每个子列表对应原列表中“显著最大”的索引
        "min_indices": [[...], [...], [...]]   # 每个子列表对应原列表中“显著最小”的索引
    }
    “显著”定义为：该位置的值比其他两个列表在同一位置的值大（或小）至少2。
    """
    if len(lists) != 3:
        raise ValueError("输入必须包含恰好三个列表")

    # 取最短长度，避免越界
    min_len = min(len(lst) for lst in lists)

    # 初始化结果：每个列表对应一个空索引列表
    max_indices = [[] for _ in range(3)]
    min_indices = [[] for _ in range(3)]

    # 遍历每个索引位置
    for i in range(min_len):
        a, b, c = lists[0][i], lists[1][i], lists[2][i]
        vals = [a, b, c]

        for idx in range(3):
            others = [vals[j] for j in range(3) if j != idx]
            current = vals[idx]

            # 检查是否显著大于其他两个
            if all(current > other + 2 for other in others):
                max_indices[idx].append(i)

            # 检查是否显著小于其他两个
            if all(current < other - 2 for other in others):
                min_indices[idx].append(i)

    return {
        "max_indices": max_indices,
        "min_indices": min_indices
    }

lists = [
    [2, 3, 5, 2, 4, 0, 0, 1, 3, 0, 4, 1, 5, 1, 5, 0, 0, 0, 1, 5, 0, 5, 3, 0, 1, 4, 2, 5, 0, 2, 1, 1, 3, 1, 0, 2, 2, 4, 5, 5, 4, 0, 0, 0, 0, 1, 5, 4, 4, 0, 2, 5, 3, 5, 0, 0, 3, 1, 0, 3, 5, 5, 0, 5, 0, 0, 0, 2, 3, 0, 0, 5, 3, 0, 2, 0, 5, 3, 5, 1, 0, 5, 0, 0, 0, 2, 4, 0, 4, 0, 3, 5, 2, 0, 1, 1, 0, 0, 2, 1, 5, 5, 5, 5],
    [1, 1, 5, 4, 4, 0, 0, 2, 1, 0, 5, 1, 5, 0, 4, 1, 0, 0, 0, 5, 0, 2, 2, 0, 2, 4, 3, 5, 0, 4, 4, 2, 0, 2, 1, 3, 3, 4, 5, 5, 5, 0, 0, 2, 0, 0, 5, 4, 5, 4, 0, 5, 4, 5, 2, 0, 5, 2, 0, 2, 5, 5, 0, 3, 0, 0, 0, 4, 4, 1, 0, 4, 4, 1, 4, 0, 5, 5, 5, 2, 0, 5, 0, 1, 3, 4, 5, 0, 5, 0, 4, 5, 0, 0, 4, 0, 1, 0, 1, 3, 5, 5, 4, 5],
    [1, 1, 3, 2, 4, 0, 0, 3, 2, 0, 4, 0, 4, 1, 4, 1, 1, 0, 1, 5, 0, 3, 1, 0, 3, 2, 3, 5, 0, 3, 0, 1, 1, 4, 0, 3, 4, 2, 5, 5, 3, 0, 0, 1, 0, 2, 5, 4, 5, 3, 1, 5, 5, 5, 0, 0, 4, 2, 0, 3, 5, 4, 0, 3, 0, 0, 0, 2, 3, 0, 0, 2, 3, 1, 2, 0, 5, 5, 5, 3, 0, 5, 1, 1, 3, 2, 4, 0, 4, 0, 4, 5, 1, 0, 2, 1, 2, 1, 1, 1, 5, 5, 2, 5]]

result = find_extreme_indices(lists)
print("max_indices:", result["max_indices"])
print("min_indices:", result["min_indices"])