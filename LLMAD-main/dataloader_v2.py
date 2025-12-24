import os
import random

import torch
from matplotlib import ticker
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from sklearn.neighbors import LocalOutlierFactor
from torch.utils.data import Dataset
import pandas as pd
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np


class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels, idx):
        self.data = data
        self.labels = labels
        self.idx = idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        z = self.idx[idx]
        return x, y, z


def load_data(data_name):
    print(f"Loading dataset: {data_name}")

    if data_name == "Computers":
        file_path_train = 'data/Computers/Computers_TRAIN.txt'
        file_path_test = 'data/Computers/Computers_TEST.txt'

        df_train = pd.read_csv(file_path_train, header=None, sep='\s+')
        df_test = pd.read_csv(file_path_test, header=None, sep='\s+')

        # Train
        labels_train = df_train.iloc[:, 0].values - 1
        data_train = df_train.iloc[:, 1:].values

        # Test
        labels_test = df_test.iloc[:, 0].values - 1
        data_test = df_test.iloc[:, 1:].values

        train_selected_indices = np.arange(len(data_train), dtype=np.int64)

        test_selected_indices = np.arange(len(data_test), dtype=np.int64)

        # All = train + test
        data_all = np.concatenate([data_train, data_test], axis=0)
        labels_all = np.concatenate([labels_train, labels_test], axis=0)
        global_selected_indices = np.arange(len(data_all), dtype=np.int64)


    elif data_name == "BME":
        file_path_train = 'data/BME/BME_TRAIN.txt'
        file_path_test = 'data/BME/BME_TEST.txt'

        df_train = pd.read_csv(file_path_train, header=None, sep='\s+')
        df_test = pd.read_csv(file_path_test, header=None, sep='\s+')

        # Train
        labels_train = df_train.iloc[:, 0].values - 1
        data_train = df_train.iloc[:, 1:].values

        # Test
        labels_test = df_test.iloc[:, 0].values - 1
        data_test = df_test.iloc[:, 1:].values

        train_selected_indices = np.arange(len(data_train), dtype=np.int64)

        test_selected_indices = np.arange(len(data_test), dtype=np.int64)

        # All = train + test
        data_all = np.concatenate([data_train, data_test], axis=0)
        labels_all = np.concatenate([labels_train, labels_test], axis=0)
        global_selected_indices = np.arange(len(data_all), dtype=np.int64)



    elif data_name == "ElectricDevices":
        file_path_train = 'data/ElectricDevices/ElectricDevices_TRAIN.txt'
        file_path_test = 'data/ElectricDevices/ElectricDevices_TEST.txt'

        df_train = pd.read_csv(file_path_train, header=None, sep='\s+')
        df_test = pd.read_csv(file_path_test, header=None, sep='\s+')

        # Raw data
        labels_train = df_train.iloc[:, 0].values - 1
        data_train = df_train.iloc[:, 1:].values

        labels_test = df_test.iloc[:, 0].values - 1
        data_test = df_test.iloc[:, 1:].values

        train_selected_indices = np.arange(len(data_train), dtype=np.int64)

        test_selected_indices = np.arange(len(data_test), dtype=np.int64)

        # All = train + test
        data_all = np.concatenate([data_train, data_test], axis=0)
        labels_all = np.concatenate([labels_train, labels_test], axis=0)
        global_selected_indices = np.arange(len(data_all), dtype=np.int64)





    elif data_name == "ECG200":
        file_path_train = 'data/ECG200/ECG200_TRAIN.txt'
        file_path_test = 'data/ECG200/ECG200_TEST.txt'

        df_train = pd.read_csv(file_path_train, header=None, sep='\s+')
        df_test = pd.read_csv(file_path_test, header=None, sep='\s+')

        # Train
        labels_train = (df_train.iloc[:, 0].values + 1) / 2
        data_train = df_train.iloc[:, 1:].values

        # Test
        labels_test = (df_test.iloc[:, 0].values + 1) / 2
        data_test = df_test.iloc[:, 1:].values

        train_selected_indices = np.arange(len(data_train), dtype=np.int64)

        test_selected_indices = np.arange(len(data_test), dtype=np.int64)

        # All = train + test
        data_all = np.concatenate([data_train, data_test], axis=0)
        labels_all = np.concatenate([labels_train, labels_test], axis=0)
        global_selected_indices = np.arange(len(data_all), dtype=np.int64)


    elif data_name == "CBF":
        file_path_train = 'data/CBF/CBF_TRAIN.txt'
        file_path_test = 'data/CBF/CBF_TEST.txt'

        df_train = pd.read_csv(file_path_train, header=None, sep='\s+')
        df_test = pd.read_csv(file_path_test, header=None, sep='\s+')

        # Train
        labels_train = df_train.iloc[:, 0].values - 1
        data_train = df_train.iloc[:, 1:].values

        # Test
        labels_test = df_test.iloc[:, 0].values - 1
        data_test = df_test.iloc[:, 1:].values

        train_selected_indices = np.arange(len(data_train), dtype=np.int64)
        test_selected_indices = np.arange(len(data_test), dtype=np.int64)

        # max_per_class_test = 40
        # test_selected_indices = []  # indices in original df_train
        #
        # for label in np.unique(labels_test):
        #     idx = np.where(labels_test == label)[0]  # 所有该类别的行号
        #     test_selected_indices.extend(idx[:max_per_class_test])  # 取前30个
        #
        # test_selected_indices = np.array(test_selected_indices)
        #
        # data_test_tensor = torch.tensor(data_test, dtype=torch.float32)
        # labels_test_tensor = torch.tensor(labels_test, dtype=torch.long) - 1

        # All = train + test
        data_all = np.concatenate([data_train, data_test], axis=0)
        labels_all = np.concatenate([labels_train, labels_test], axis=0)
        global_selected_indices = np.arange(len(data_all), dtype=np.int64)

    elif data_name == "Trace":
        file_path_train = 'data/Trace/Trace_TRAIN.txt'
        file_path_test = 'data/Trace/Trace_TEST.txt'

        df_train = pd.read_csv(file_path_train, header=None, sep='\s+')
        df_test = pd.read_csv(file_path_test, header=None, sep='\s+')

        # Train
        labels_train = df_train.iloc[:, 0].values - 1
        data_train = df_train.iloc[:, 1:].values

        # Test
        labels_test = df_test.iloc[:, 0].values - 1
        data_test = df_test.iloc[:, 1:].values

        train_selected_indices = np.arange(len(data_train), dtype=np.int64)
        test_selected_indices = np.arange(len(data_test), dtype=np.int64)

        # All = train + test
        data_all = np.concatenate([data_train, data_test], axis=0)
        labels_all = np.concatenate([labels_train, labels_test], axis=0)
        global_selected_indices = np.arange(len(data_all), dtype=np.int64)


    elif data_name == "DodgerLoopWeekend":
        file_path_train = 'data/DodgerLoopWeekend/DodgerLoopWeekend_TRAIN.txt'
        file_path_test = 'data/DodgerLoopWeekend/DodgerLoopWeekend_TEST.txt'

        df_train = pd.read_csv(file_path_train, header=None, sep='\s+')
        df_test = pd.read_csv(file_path_test, header=None, sep='\s+')

        # Train
        labels_train = df_train.iloc[:, 0].values
        data_train = df_train.iloc[:, 1:].values

        # Test
        labels_test = df_test.iloc[:, 0].values
        data_test = df_test.iloc[:, 1:].values

        # Convert to tensors (labels are 0-indexed)
        data_train_tensor = torch.tensor(data_train, dtype=torch.float32)
        labels_train_tensor = torch.tensor(labels_train, dtype=torch.long) - 1
        train_selected_indices = torch.arange(len(data_train_tensor), dtype=torch.long)

        data_test_tensor = torch.tensor(data_test, dtype=torch.float32)
        labels_test_tensor = torch.tensor(labels_test, dtype=torch.long) - 1
        test_selected_indices = torch.arange(len(data_test_tensor), dtype=torch.long)

        # All = train + test
        data_all_tensor = torch.cat([data_train_tensor, data_test_tensor], dim=0)
        labels_all_tensor = torch.cat([labels_train_tensor, labels_test_tensor], dim=0)
        global_selected_indices = torch.arange(len(data_all_tensor), dtype=torch.long)


    elif data_name == "AbnormalHeartBeat":
        file_path_train = 'data/AbnormalHeartBeat/HeartBeat_TRAIN.ts'
        file_path_test = 'data/AbnormalHeartBeat/HeartBeat_TEST.ts'

        def load_ts_file(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Step 1: Parse metadata and find @data
            series_length = None
            data_started = False
            data_lines = []

            for line in lines:
                stripped = line.strip()
                if not stripped or stripped.startswith('%'):
                    continue
                if stripped.startswith('@seriesLength'):
                    series_length = int(stripped.split()[-1])
                elif stripped.startswith('@data'):
                    data_started = True
                    continue
                elif not data_started:
                    continue  # skip other @ lines
                else:
                    # In @data section
                    if stripped:
                        data_lines.append(stripped)

            if series_length is None:
                raise ValueError(f"@seriesLength not found in {file_path}")
            if not data_lines:
                raise ValueError(f"No data lines found in {file_path}")

            # Step 2: Parse each time series line (format: val1,val2,...,valN:label)
            features = []
            labels = []

            for line in data_lines:
                # Fix possible Chinese colon
                if '：' in line:
                    line = line.replace('：', ':')
                if ':' not in line:
                    raise ValueError(f"Missing ':' in data line: {line}")

                ts_part, label_part = line.rsplit(':', 1)
                try:
                    # Parse label (should be integer)
                    # label = int(label_part.strip())
                    if label_part == 'normal':
                        label = 0
                    else:
                        label = 1
                    # Parse time series values
                    values = list(map(float, ts_part.split(',')))
                    if len(values) != series_length:
                        print(f"Warning: Expected {series_length} values, got {len(values)}. Skipping line: {line}")
                        continue
                    features.append(values)
                    labels.append(label)
                except Exception as e:
                    print(f"Skipping malformed line: {line} | Error: {e}")
                    continue

            if not features:
                raise RuntimeError(f"No valid samples loaded from {file_path}")

            return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int64)

        # Load data from .ts files
        data_train, labels_train = load_ts_file(file_path_train)
        data_test, labels_test = load_ts_file(file_path_test)

        # Convert to PyTorch tensors
        data_train_tensor = torch.tensor(data_train, dtype=torch.float32)
        labels_train_tensor = torch.tensor(labels_train, dtype=torch.long)

        data_test_tensor = torch.tensor(data_test, dtype=torch.float32)
        labels_test_tensor = torch.tensor(labels_test, dtype=torch.long)

        # Optional: combine all
        data_all_tensor = torch.cat([data_train_tensor, data_test_tensor], dim=0)
        labels_all_tensor = torch.cat([labels_train_tensor, labels_test_tensor], dim=0)



    elif data_name == "ECG":
        file_path_train = 'data/ECG/ECG_TRAIN.ts'
        file_path_test = 'data/ECG/ECG_TEST.ts'

        def load_ts_file(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Step 1: Parse metadata and find @data
            series_length = None
            data_started = False
            data_lines = []

            for line in lines:
                stripped = line.strip()
                if not stripped or stripped.startswith('%'):
                    continue
                if stripped.startswith('@seriesLength'):
                    series_length = int(stripped.split()[-1])
                elif stripped.startswith('@data'):
                    data_started = True
                    continue
                elif not data_started:
                    continue  # skip other @ lines
                else:
                    # In @data section
                    if stripped:
                        data_lines.append(stripped)

            if series_length is None:
                raise ValueError(f"@seriesLength not found in {file_path}")
            if not data_lines:
                raise ValueError(f"No data lines found in {file_path}")

            # Step 2: Parse each time series line (format: val1,val2,...,valN:label)
            features = []
            labels = []

            for line in data_lines:
                # Fix possible Chinese colon
                if '：' in line:
                    line = line.replace('：', ':')
                if ':' not in line:
                    raise ValueError(f"Missing ':' in data line: {line}")

                ts_part, label_part = line.rsplit(':', 1)
                try:
                    # Parse label (should be integer)
                    label = int(label_part.strip())
                    # Parse time series values
                    values = list(map(float, ts_part.split(',')))
                    if len(values) != series_length:
                        print(f"Warning: Expected {series_length} values, got {len(values)}. Skipping line: {line}")
                        continue
                    features.append(values)
                    labels.append(label)
                except Exception as e:
                    print(f"Skipping malformed line: {line} | Error: {e}")
                    continue

            if not features:
                raise RuntimeError(f"No valid samples loaded from {file_path}")

            return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int64)

        # Load data from .ts files
        data_train, labels_train = load_ts_file(file_path_train)
        data_test, labels_test = load_ts_file(file_path_test)

        # Convert to PyTorch tensors
        data_train_tensor = torch.tensor(data_train, dtype=torch.float32)
        labels_train_tensor = torch.tensor(labels_train, dtype=torch.long)

        data_test_tensor = torch.tensor(data_test, dtype=torch.float32)
        labels_test_tensor = torch.tensor(labels_test, dtype=torch.long)

        # Optional: combine all
        data_all_tensor = torch.cat([data_train_tensor, data_test_tensor], dim=0)
        labels_all_tensor = torch.cat([labels_train_tensor, labels_test_tensor], dim=0)



    elif data_name == "TwoLeadECG":
        file_path_train = 'data/TwoLeadECG/TwoLeadECG_TRAIN.txt'
        file_path_test = 'data/TwoLeadECG/TwoLeadECG_TEST.txt'

        df_train = pd.read_csv(file_path_train, header=None, sep='\s+')
        df_test = pd.read_csv(file_path_test, header=None, sep='\s+')

        # Train
        labels_train = df_train.iloc[:, 0].values
        data_train = df_train.iloc[:, 1:].values

        # Test
        labels_test = df_test.iloc[:, 0].values
        data_test = df_test.iloc[:, 1:].values

        # Convert to tensors (labels are 0-indexed)
        data_train_tensor = torch.tensor(data_train, dtype=torch.float32)
        labels_train_tensor = torch.tensor(labels_train, dtype=torch.long) - 1

        data_test_tensor = torch.tensor(data_test, dtype=torch.float32)
        labels_test_tensor = torch.tensor(labels_test, dtype=torch.long) - 1

        # All = train + test
        data_all_tensor = torch.cat([data_train_tensor, data_test_tensor], dim=0)
        labels_all_tensor = torch.cat([labels_train_tensor, labels_test_tensor], dim=0)


    elif data_name == "ArrowHead":
        file_path_train = 'data/ArrowHead/ArrowHead_TRAIN.txt'
        file_path_test = 'data/ArrowHead/ArrowHead_TEST.txt'

        df_train = pd.read_csv(file_path_train, header=None, sep='\s+')
        df_test = pd.read_csv(file_path_test, header=None, sep='\s+')

        # Train
        labels_train = df_train.iloc[:, 0].values
        data_train = df_train.iloc[:, 1:].values

        # Test
        labels_test = df_test.iloc[:, 0].values
        data_test = df_test.iloc[:, 1:].values

        train_selected_indices = np.arange(len(data_train), dtype=np.int64)
        test_selected_indices = np.arange(len(data_test), dtype=np.int64)
        data_all = np.concatenate([data_train, data_test], axis=0)
        labels_all = np.concatenate([labels_train, labels_test], axis=0)
        global_selected_indices = np.arange(len(data_all), dtype=np.int64)


    elif data_name == "StarLightCurves":
        file_path_train = 'data/StarLightCurves/StarLightCurves_TRAIN.txt'
        file_path_test = 'data/StarLightCurves/StarLightCurves_TEST.txt'

        df_train = pd.read_csv(file_path_train, header=None, sep='\s+')
        df_test = pd.read_csv(file_path_test, header=None, sep='\s+')

        # Train
        labels_train = df_train.iloc[:, 0].values-1
        data_train = df_train.iloc[:, 1:].values

        # Test
        labels_test = df_test.iloc[:, 0].values-1
        data_test = df_test.iloc[:, 1:].values

        # max_per_class_train = 50
        # train_selected_indices = []  # indices in original df_train
        #
        # for label in np.unique(labels_train):
        #     idx = np.where(labels_train == label)[0]  # 所有该类别的行号
        #     train_selected_indices.extend(idx[:max_per_class_train])  # 取前30个
        #
        # train_selected_indices = np.array(train_selected_indices)

        # Subset train data using selected indices
        # data_train_selected = data_train[train_selected_indices]
        # labels_train_selected = labels_train[train_selected_indices]
        # data_train_tensor = torch.tensor(data_train_selected, dtype=torch.float32)
        # labels_train_tensor = torch.tensor(labels_train_selected, dtype=torch.long) - 1

        # data_train_tensor = torch.tensor(data_train, dtype=torch.float32)
        # labels_train_tensor = torch.tensor(labels_train, dtype=torch.long) - 1
        #
        # max_per_class_test = 100
        # test_selected_indices = []  # indices in original df_train
        #
        # for label in np.unique(labels_test):
        #     idx = np.where(labels_test == label)[0]  # 所有该类别的行号
        #     test_selected_indices.extend(idx[:max_per_class_test])  # 取前30个
        #
        # test_selected_indices = np.array(test_selected_indices)
        #
        # # Subset train data using selected indices
        # data_test_selected = data_test[test_selected_indices]
        # labels_test_selected = labels_test[test_selected_indices]
        #
        # data_test_tensor = torch.tensor(data_test_selected, dtype=torch.float32)
        # labels_test_tensor = torch.tensor(labels_test_selected, dtype=torch.long) - 1

        train_selected_indices = np.arange(len(data_train), dtype=np.int64)
        test_selected_indices = np.arange(len(data_test), dtype=np.int64)

        # All = train + test
        data_all = np.concatenate([data_train, data_test], axis=0)
        labels_all = np.concatenate([labels_train, labels_test], axis=0)
        global_selected_indices = np.concatenate([train_selected_indices, test_selected_indices])

    elif data_name == "EOGHorizontalSignal":
        file_path_train = 'data/EOGHorizontalSignal/EOGHorizontalSignal_TRAIN.txt'
        file_path_test = 'data/EOGHorizontalSignal/EOGHorizontalSignal_TEST.txt'

        df_train = pd.read_csv(file_path_train, header=None, sep='\s+')
        df_test = pd.read_csv(file_path_test, header=None, sep='\s+')

        # Train
        labels_train = df_train.iloc[:, 0].values
        data_train = df_train.iloc[:, 1:].values

        # Test
        labels_test = df_test.iloc[:, 0].values
        data_test = df_test.iloc[:, 1:].values

        train_selected_indices = np.arange(len(data_train), dtype=np.int64)
        test_selected_indices = np.arange(len(data_test), dtype=np.int64)

        # All = train + test
        data_all = np.concatenate([data_train, data_test], axis=0)
        labels_all = np.concatenate([labels_train, labels_test], axis=0)
        global_selected_indices = np.concatenate([train_selected_indices, test_selected_indices])


    elif data_name == "Chinatown":
        file_path_train = 'data/Chinatown/Chinatown_TRAIN.txt'
        file_path_test = 'data/Chinatown/Chinatown_TEST.txt'

        df_train = pd.read_csv(file_path_train, header=None, sep='\s+')
        df_test = pd.read_csv(file_path_test, header=None, sep='\s+')

        # Train
        labels_train = df_train.iloc[:, 0].values - 1
        data_train = df_train.iloc[:, 1:].values

        # Test
        labels_test = df_test.iloc[:, 0].values - 1
        data_test = df_test.iloc[:, 1:].values

        # Convert to tensors (labels are 0-indexed)
        data_train_tensor = torch.tensor(data_train, dtype=torch.float32)
        labels_train_tensor = torch.tensor(labels_train, dtype=torch.long)

        data_test_tensor = torch.tensor(data_test, dtype=torch.float32)
        labels_test_tensor = torch.tensor(labels_test, dtype=torch.long)

        # All = train + test
        data_all_tensor = torch.cat([data_train_tensor, data_test_tensor], dim=0)
        labels_all_tensor = torch.cat([labels_train_tensor, labels_test_tensor], dim=0)


    elif data_name == "Epilepsy":

        file_path_train = 'data/Epilepsy/EpilepsyDimension1_TRAIN.arff'

        file_path_test = 'data/Epilepsy/EpilepsyDimension1_TEST.arff'

        def load_arff_file(file_path):

            with open(file_path, 'r') as f:

                lines = f.readlines()

            # Step 1: Find @data section

            data_started = False

            data_lines = []

            for line in lines:

                line = line.strip()

                if not line or line.startswith('%'):
                    continue  # skip comments

                if line.lower().startswith('@data'):
                    data_started = True

                    continue

                if not data_started:
                    continue  # skip @relation, @attribute, etc.

                # Now in @data section

                if line:  # non-empty

                    data_lines.append(line)

            # Step 2: Parse data lines

            features = []

            labels = []

            label_map = {"EPILEPSY": 0, "WALKING": 1, "RUNNING": 2, "SAWING": 3}

            for line in data_lines:

                parts = line.split(',')

                if len(parts) < 2:
                    continue

                # All but last are numeric features

                try:

                    feat_vals = [float(x) for x in parts[:-1]]

                    label_str = parts[-1].strip()

                    if label_str not in label_map:
                        raise ValueError(f"Unknown label: {label_str}")

                    features.append(feat_vals)

                    labels.append(label_map[label_str])

                except Exception as e:

                    print(f"Skipping malformed line: {line} | Error: {e}")

                    continue

            return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int64)

        # Load data

        data_train, labels_train = load_arff_file(file_path_train)

        data_test, labels_test = load_arff_file(file_path_test)

        train_selected_indices = np.arange(len(data_train), dtype=np.int64)
        test_selected_indices = np.arange(len(data_test), dtype=np.int64)

        data_all = np.concatenate([data_train, data_test], axis=0)
        labels_all = np.concatenate([labels_train, labels_test], axis=0)
        global_selected_indices = np.concatenate([train_selected_indices, test_selected_indices], axis=0)




    else:
        raise ValueError(f"Unsupported dataset: {data_name}")

    return {
        'train': (data_train, labels_train, train_selected_indices),
        'test': (data_test, labels_test, test_selected_indices),
        'all': (data_all, labels_all, global_selected_indices),
        # 'all_test': (data_all_test_tensor, labels_all_test_tensor)
    }


def random_choice(labels, select_indice, top_k_random=10):
    # 获取 select_indice 对应的标签子集
    subset_labels = labels[select_indice]  # shape: (M,)
    unique_labels = np.unique(labels)  # 保持所有类别顺序（即使某些类在 select_indice 中不存在）

    random_indices = []

    for label in unique_labels:
        # 找出 subset 中属于当前类别的位置（在 select_indice 中的位置）
        mask = (subset_labels == label)

        # 获取这些样本在 select_indice 中的局部索引
        local_indices = np.where(mask)[0]  # shape: (K,)
        select_indice = np.array(select_indice)
        global_indices = select_indice[local_indices]  # 映射回原始全局索引

        if len(global_indices) <= top_k_random:
            selected_global = global_indices
        else:
            # 随机打乱并选前 top_k_random 个
            perm = torch.randperm(len(global_indices))
            selected_global = global_indices[perm[:top_k_random]]

        random_indices.append(selected_global.tolist())

    return random_indices


def cluster_per_class(data_np, labels_np, selected_indice, n_clusters=5, top_k=10, random_state=42):
    """
    对每个类别分别进行 KMeans 聚类，**仅使用 selected_indice 中的样本**。

    Args:
        data_tensor: (N, T) tensor of time series
        labels_tensor: (N,) tensor of class labels (0-indexed)
        selected_indice: list or array of indices (subset of [0, N)) to consider for clustering
        n_clusters: number of clusters per class
        top_k: total number of samples to return per cluster
        random_state: for reproducible random sampling

    Returns:
        dict: {
            class_label: [
                [global_idx1, global_idx2, ..., global_idx_topk],  # cluster 0
                ...
            ]
        }
    """
    selected_indice = np.array(selected_indice)

    # 提取子集
    subset_data = data_np[selected_indice]  # shape: (M, T)
    subset_labels = labels_np[selected_indice]  # shape: (M,)

    unique_labels = np.unique(subset_labels)
    result = {}
    rng = np.random.default_rng(random_state)

    for label in unique_labels:
        mask = (subset_labels == label)
        class_data = subset_data[mask]  # (C, T)
        local_indices = np.where(mask)[0]  # indices in subset_data
        global_indices = selected_indice[mask]  # map back to original global indices

        if len(class_data) == 0:
            result[label] = [[] for _ in range(n_clusters)]
            continue

        if len(class_data) < n_clusters:
            # 样本数 < n_clusters：每个样本作为一个聚类
            clusters_topk = []
            for i in range(n_clusters):
                if i < len(global_indices):
                    clusters_topk.append([int(global_indices[i])] if top_k >= 1 else [])
                else:
                    clusters_topk.append([])
            result[label] = clusters_topk
            continue

        # KMeans 聚类（在 class_data 上）
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(class_data)
        centroids = kmeans.cluster_centers_

        clusters_topk = []
        for i in range(n_clusters):
            cluster_mask = (cluster_labels == i)
            cluster_samples = class_data[cluster_mask]  # (S, T)
            cluster_global_indices = global_indices[cluster_mask]  # 原始全局索引！

            if len(cluster_samples) == 0:
                clusters_topk.append([])
                continue

            total_samples = len(cluster_samples)
            if total_samples <= top_k:
                # 返回全部
                clusters_topk.append(cluster_global_indices.tolist())
                continue

            # 计算到质心的距离，选最近的 top_k 个（你原逻辑是只选最近，不加随机）
            centroid = centroids[i].reshape(1, -1)
            distances = euclidean_distances(cluster_samples, centroid).flatten()
            nearest_local = np.argsort(distances)[:top_k]  # local in cluster_samples
            nearest_global = cluster_global_indices[nearest_local]

            clusters_topk.append(nearest_global.tolist())

        result[label] = clusters_topk

    return result


def save_all_train_plots(data_all, label_all, index, save_dir, max_per_class):
    # Step 1: 按类别收集索引
    class_to_indices = {}
    for idx in index:
        label = label_all[idx]
        label = int(label)

        if label not in class_to_indices:
            class_to_indices[label] = []
        class_to_indices[label].append(idx)

    # Step 2: 对每个类别随机采样最多 max_per_class 个索引
    selected_indices = []
    for label, indices in class_to_indices.items():
        sampled = random.sample(indices, min(max_per_class, len(indices)))
        selected_indices.extend(sampled)

    # 可选：打乱顺序（非必须）
    # random.shuffle(selected_indices)

    # Step 3: 遍历选中的索引进行绘图
    for idx in selected_indices:
        data = data_all[idx]
        label = label_all[idx]

        label = int(label)

        # 创建分类目录
        class_save_dir = os.path.join(save_dir, f"class{label}")
        os.makedirs(class_save_dir, exist_ok=True)

        # 绘图
        plt.figure(figsize=(8, 4))
        if data.ndim == 1:
            plt.plot(data)
        else:
            for c in range(data.shape[0]):
                plt.plot(data[c], label=f"Channel {c}")
            plt.legend()

        ax = plt.gca()
        ax.xaxis.set_major_locator(ticker.MultipleLocator(25))

        # def scale_x_ticks(x, pos):
        #     return f"{x * 0.01:.2f}"
        # ax.xaxis.set_major_formatter(ticker.FuncFormatter(scale_x_ticks))

        plt.xlabel("Time(s)")
        plt.ylabel("Value", rotation=0, labelpad=20)
        plt.ylim(-3, 4)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        filename = os.path.join(class_save_dir, f"{idx}.png")
        plt.savefig(filename, dpi=150)
        plt.close()


# def save_all_train_plots(data_all, label_all, index, save_dir, max):
#     # 记录每个类别的保存图像数量
#     saved_plots_count = {}
#
#     for idx in index:
#         data = data_all[idx]
#         label = label_all[idx]
#
#         # 转为 numpy
#         if torch.is_tensor(data):
#             data = data.cpu().numpy()
#         if torch.is_tensor(label):
#             label = label.item()
#
#         # 检查是否已达到每个类别的最大保存图像数量
#         if label not in saved_plots_count:
#             saved_plots_count[label] = 0
#
#         if saved_plots_count[label] >= max:
#             continue  # 跳过此数据点
#
#         # 创建分类目录
#         class_save_dir = os.path.join(save_dir, f"class{int(label)}")
#         os.makedirs(class_save_dir, exist_ok=True)
#
#         # 绘图
#         plt.figure(figsize=(8, 4))
#         if data.ndim == 1:
#             plt.plot(data)
#         else:
#             # 如果是多通道 (C, T)，画在同一图上（可用不同颜色）
#             for c in range(data.shape[0]):
#                 plt.plot(data[c], label=f"Channel {c}")
#             plt.legend()
#
#         ax = plt.gca()  # 获取当前图像的 axes 对象
#         ax.xaxis.set_major_locator(ticker.MultipleLocator(20))  # 设置主刻度间隔为 100
#         plt.xlabel("Time Step")
#         plt.ylabel("Value", rotation=0, labelpad=20)
#         plt.ylim(-2, 10)
#         plt.grid(True, linestyle='--', alpha=0.6)
#         plt.tight_layout()
#
#         # 保存
#         filename = os.path.join(class_save_dir, f"{idx}.png")
#         plt.savefig(filename, dpi=150)
#         plt.close()
#
#         # 更新已保存图像计数
#         saved_plots_count[label] += 1


def compute_lof_per_class(data, label, n_neighbors=5):
    """
    对每个类别分别计算 LOF 分数。

    Parameters:
        data: np.ndarray, shape (n_samples, n_features)
        label: np.ndarray, shape (n_samples,)
        n_neighbors: int, LOF 的邻居数（注意不能超过某类样本数）

    Returns:
        lof_scores: np.ndarray, shape (n_samples,)，与输入 data 顺序一致
    """
    n_samples = data.shape[0]
    lof_scores = np.full(n_samples, np.nan)  # 初始化为 NaN

    unique_labels = np.unique(label)

    for cls in unique_labels:
        # 获取当前类的索引
        mask = (label == cls)
        indices = np.where(mask)[0]
        X_cls = data[mask]  # shape: (n_cls, 151)

        n_cls = X_cls.shape[0]
        if n_cls <= 1:
            # 无法计算 LOF，跳过或设为 1（正常）
            lof_scores[indices] = 1.0
            continue

        # 设置合适的 n_neighbors（不能超过 n_cls - 1）
        k = min(n_neighbors, n_cls - 1)
        if k < 1:
            k = 1

        # 计算 LOF（novelty=False，因为是在训练集自身上计算）
        lof = LocalOutlierFactor(n_neighbors=k, novelty=False)
        lof.fit(X_cls)

        # 注意：sklearn 返回的是负的 LOF 值
        scores = -lof.negative_outlier_factor_
        lof_scores[indices] = scores

    return lof_scores


if __name__ == '__main__':
    data_name = "Trace"  # EOGHorizontalSignal Epilepsy
    # 加载数据
    data, label, index = load_data(data_name)['train']
    data = np.array(data, dtype=np.float32)
    label = np.array(label, dtype=int)
    index = np.array(index)  # 即使是整数 ID，也转成 array

    # x = data[110]
    # stat(x)
    neighbor = len(label) // len(np.unique(label)) // 4
    lof_scores = compute_lof_per_class(data, label, n_neighbors=neighbor)

    # 创建掩码：保留 LOF <= 2 的样本
    mask = lof_scores <= 2
    outlier_indices = index[~mask]
    print(outlier_indices.tolist())
    outlier_label = label[~mask]
    print(outlier_label.tolist())

    # 过滤 data, label, index
    data_clean = data[mask]
    label_clean = label[mask]
    index_clean = index[mask]

    print(f"原始样本数: {len(lof_scores)}")
    print(f"清洗后样本数: {len(data_clean)}")
    print(f"移除了 {np.sum(~mask)} 个异常样本")

    print(len(index))

    save_all_train_plots(data, label, index, save_dir=f"plots/{data_name}_train_nolabel", max_per_class=50)
