# 标准库导入
from logging import root
import math
import os
import re
import sys
import time
import numbers
import warnings
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import Pool
from typing import Tuple
import pickle
from pathlib import Path

# 第三方库导入
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import (PCA, SparsePCA, FactorAnalysis, KernelPCA, TruncatedSVD)
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, mean_squared_error, r2_score)
from sklearn.model_selection import (train_test_split, cross_val_score, StratifiedKFold, RandomizedSearchCV)
from sklearn.naive_bayes import (GaussianNB, MultinomialNB, CategoricalNB, ComplementNB)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import (SMOTE, ADASYN, BorderlineSMOTE)
from imblearn.combine import SMOTEENN
from scipy.stats import uniform, randint
import shap

# 本地库导入
from self_ada import self_ada
from self_bayesian import self_BO
from self_knn import self_knn
from self_lr import self_lr
from self_mlp import self_mlp
from self_rf import self_rf
from self_svm import self_svm
from self_xgb import self_xgb
from self_mlp_pytorch import pytorch_model_sklearn, self_mlp_torch, Model


# 将上一级目录添加到 sys.path
from utils import grid_param_builder, load_iris_shuffle, write_content_to_file

from params import *

# ---------------------------------------------------------------------------------
# 通用pytorch模型训练方法
def general_torch_model_fit(model: nn.Module, data: torch.Tensor, targets:torch.Tensor, batch_size: int, num_epochs: int, criterion, optimizer: torch.optim.Optimizer, use_cpu: bool = False, validation_split: float = 0.2, early_stop = None) -> nn.Module:
    """
    训练模型的方法

    参数:
    model (nn.Module): 要训练的模型实例
    data (Union[torch.Tensor, list]): 输入数据
    targets (Union[torch.Tensor, list]): 目标标签
    batch_size (int): 批量大小
    num_epochs (int): 训练轮数
    criterion (Callable): 损失函数
    optimizer (torch.optim.Optimizer): 优化器
    use_cpu (bool): 是否使用CPU进行训练，如果为False且GPU可用，则使用GPU
    validation_split (float): 验证集比例
    early_stop (Optional[int]): 早停的阈值（验证损失未改善的最大轮数），如果为None则不使用早停
    """
    # 选择设备
    device = torch.device('cuda' if torch.cuda.is_available() and not use_cpu else 'cpu')

    # 将数据转换为Tensor并移动到指定设备
    data = data.clone().detach().to(torch.float32).to(device)
    targets = targets.clone().detach().to(torch.float32).to(device)

    # 切分训练集和验证集
    train_data, val_data, train_targets, val_targets = train_test_split(data, targets, test_size=validation_split)
    # print(type(train_data))
    train_data, train_targets = train_data.to(device), train_targets.to(device)
    val_data, val_targets = val_data.to(device), val_targets.to(device)
    # print(type(train_data))
    # 将模型移动到指定设备
    model = model.to(device)

    # 训练函数
    def fit(model: nn.Module, train_data: torch.Tensor, train_targets: torch.Tensor, val_data: torch.Tensor, val_targets: torch.Tensor, criterion, optimizer: torch.optim.Optimizer, num_epochs: int, batch_size: int, early_stop) -> None:
        """
        执行模型训练

        参数:
        model (nn.Module): 要训练的模型
        train_data (Tensor): 训练数据张量
        train_targets (Tensor): 训练目标张量
        val_data (Tensor): 验证数据张量
        val_targets (Tensor): 验证目标张量
        criterion (Callable): 损失函数
        optimizer (torch.optim.Optimizer): 优化器
        num_epochs (int): 训练轮数
        batch_size (int): 批量大小
        early_stop (Optional[int]): 早停的阈值（验证损失未改善的最大轮数），如果为None则不使用早停
        """
        model.train()
        dataset_size = len(train_data)
        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(num_epochs):
            # 随机打乱训练数据
            permutation = torch.randperm(dataset_size)
            shuffled_data = train_data[permutation]
            shuffled_targets = train_targets[permutation]

            for i in range(0, dataset_size, batch_size):
                inputs = shuffled_data[i:i+batch_size]
                batch_targets = shuffled_targets[i:i+batch_size]

                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, batch_targets)

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # 验证集上的损失计算
            model.eval()
            with torch.no_grad():
                val_outputs = model(val_data)
                val_loss = criterion(val_outputs, val_targets)
            model.train()

            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, Validation Loss: {val_loss.item()}')

            # 早停机制
            if early_stop is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= early_stop:
                    print(f'Early stopping triggered after {epoch+1} epochs.')
                    break
    # 训练模型
    fit(model, train_data, train_targets, val_data, val_targets, criterion, optimizer, num_epochs, batch_size, early_stop)
    return model

# ---------------------------------------------------------------------------------
# 数据读取与预处理


def load_cell_data(cell_data_path:str) -> np.ndarray:
    return np.loadtxt(cell_data_path)

def load_moles_data(mole_data_path:str) -> np.ndarray:
    return np.loadtxt(mole_data_path)

def pca_method(x_data, n_components, scaler_path, pca_path):
    """
    使用缓存模型进行归一化和PCA降维
    """
    # 定义lambda方法
    normalize_data = lambda x: (StandardScaler().fit_transform(x), StandardScaler().fit(x))
    perform_pca = lambda x, n: (PCA(n_components=n).fit_transform(x), PCA(n_components=n).fit(x))
    save_model = lambda model, path: pickle.dump(model, open(path, 'wb'))
    load_model = lambda path: pickle.load(open(path, 'rb'))
    models_exist = lambda sp, pp: os.path.exists(sp) and os.path.exists(pp)

    if models_exist(scaler_path, pca_path):
        # 加载模型
        scaler = load_model(scaler_path)
        pca = load_model(pca_path)
    else:
        # 归一化数据并保存模型
        x_normalized, scaler = normalize_data(x_data)
        save_model(scaler, scaler_path)

        # 执行PCA并保存模型
        x_pca, pca = perform_pca(x_normalized, n_components)
        save_model(pca, pca_path)

    # 对输入数据进行转换
    x_normalized = scaler.transform(x_data)
    x_pca = pca.transform(x_normalized)

    return x_pca, pca.components_, pca.explained_variance_ratio_


def binning(y_data:np.ndarray, num_buckets=10, method='uniform', min_bucket_size=5):
    '''分桶
        num_buckets 分为几桶
        min_bucket_size 最小分桶数据阈值
        method 等频或等距
        返回：分桶结果，每一桶的平均值
    '''
    def bin_data(y_data, num_buckets=10, method='uniform'):
        """
        Bin the data using specified method.
        """
        if method == 'uniform':
            bucketed_data = pd.cut(y_data, bins=num_buckets, labels=False, retbins=True)
        elif method == 'quantile':
            bucketed_data = pd.qcut(y_data, q=num_buckets, labels=False, duplicates='drop', retbins=True)
        else:
            raise ValueError("method must be 'uniform' or 'quantile'")

        return bucketed_data[0], bucketed_data[1]

    def handle_small_bins(bucket_dict, num_buckets, min_bucket_size=5):
        """
        Handle bins with data below the minimum threshold.
        """
        for i in range(num_buckets):
            if len(bucket_dict[i]) < min_bucket_size:
                if i == 0:  # First bin
                    bucket_dict[i+1].extend(bucket_dict[i])
                elif i == num_buckets-1:  # Last bin
                    bucket_dict[i-1].extend(bucket_dict[i])
                else:
                    half = len(bucket_dict[i]) // 2
                    bucket_dict[i-1].extend(bucket_dict[i][:half])
                    bucket_dict[i+1].extend(bucket_dict[i][half:])
                bucket_dict[i] = []
        return bucket_dict

    bucketed_data, bins = bin_data(y_data, num_buckets, method)

    bucket_dict = {i: [] for i in range(num_buckets)}

    for idx, bucket in enumerate(bucketed_data):
        bucket_dict[bucket].append(y_data[idx])

    bucket_dict = handle_small_bins(bucket_dict, num_buckets, min_bucket_size)
    # 计算每一桶的平均值
    means = np.array([np.mean(bucket_dict[i]) if len(bucket_dict[i]) > 0 else 0 for i in range(num_buckets)])

    new_bucketed_data = []
    for idx, bucket in enumerate(bucketed_data):
        if len(bucket_dict[bucket]) > 0:
            new_bucketed_data.append(bucket)
        else:
            if bucket == 0:
                new_bucketed_data.append(1)
            elif bucket == num_buckets - 1:
                new_bucketed_data.append(num_buckets - 2)
            else:
                new_bucketed_data.append(bucket - 1 if len(bucket_dict[bucket - 1]) > 0 else bucket + 1)

    return np.array(new_bucketed_data), means


# ---------------------------------------------------------------------------------

class voting_model:
    
    def __init__(self, model_list:list, essemble_method='hard') -> None:
        self.model_list = model_list
        self.essemble_method = essemble_method
        self.pertuba_accu_list = []
        self.shap_accu_list = []

    def clear_attribute(self):
        self.model_list = []
        self.essemble_method = ''
        self.pertuba_accu_list = []
        self.shap_accu_list = []


    def fit(self, x_data:np.ndarray, y_data:np.ndarray):
        for model in self.model_list:
            # print(model)
            # print(x_data.shape, y_data.shape)
            model.fit(x_data, y_data)

    def cross_fitting(self, x_data, y_data):
        ...

    def predict(self, x_data:np.ndarray, method = "voting"):
        """method : voting soft_voting average"""
        predict_proba_list = self.predict_proba(x_data)
        if method == "voting":
            # Hard voting
            predictions = np.array([np.argmax(probas, axis=1) for probas in predict_proba_list])
            # Mode across predictions
            final_predictions = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)

        elif method == "soft_voting":
            # Soft voting
            avg_probas = np.mean(predict_proba_list, axis=0)
            final_predictions = np.argmax(avg_probas, axis=1)

        elif method == "average":
            # Average of predictions
            predictions = np.array([np.argmax(probas, axis=1) for probas in predict_proba_list])
            final_predictions = np.mean(predictions, axis=0)
            final_predictions = np.round(final_predictions).astype(int)

        else:
            raise ValueError("Unsupported method : %s"%(method))

        return final_predictions

    def predict_proba(self, x_data:np.ndarray):
        '''判断准确率'''
        proba_list = []
        for model in self.model_list:
            proba_list.append(model.predict_proba(x_data))
        return proba_list


    def regenerate_feature_importance(self, reducer, feature_importance_list, x_test: np.ndarray, y_test: np.ndarray, method, reductor_method: str) -> np.ndarray:
        """
        将降维后的特征重要性还原到原始的特征空间。
        """
        if method == 'hard':
            results = []
            for feature_importances in feature_importance_list:
                if reductor_method == 'pca':
                    results.append(reducer.inverse_transform(feature_importances))
                elif reductor_method == 'KernelPCA':
                    results.append(reducer.inverse_transform(feature_importances))
                elif reductor_method == 'TruncatedSVD':
                    results.append(reducer.inverse_transform(feature_importances))
                else:
                    raise ValueError("Unsupported reduction method: %s"%(reductor_method))
            return np.vstack(results).mean(axis=0)

        elif method == 'soft':
            # 软加权，重要性的权重来自于模型的R²
            accu_list = np.zeros(len(self.model_list))
            for i in range(len(self.model_list)):
                y_predict = self.model_list[i].predict(x_test)
                accu_list[i] = r2_score(y_test, y_predict)

            weights = accu_list / accu_list.sum()
            feature_importance_array = np.array(feature_importance_list)
            print(f"Feature importance list shape: {feature_importance_array.shape}")
            print(f"Accu list.shape: {accu_list.shape}")
            print(f"Weights shape: {weights.shape}")
            # print(f"Feature importance list: {feature_importance_list}")

            if feature_importance_array.shape[0] != len(weights):
                # for line in feature_importance_array:
                    # print(line)
                raise ValueError(f"The length of weights {len(weights)} is not compatible with the first dimension of feature importance list {feature_importance_array.shape[0]}.")
            
            if feature_importance_array.shape[0] == 0:
                for line in feature_importance_array:
                    print(line)
                raise ValueError("The feature importance list is empty.")

            results = np.average(feature_importance_array, axis=0, weights=weights)
            return results
        else:
            raise ValueError("Unsupported method")


    def calculate_perturbation_accuracy(self, x_test: np.ndarray, y_test: np.ndarray):
        '''微扰法，重要性计算'''
        self.pertuba_accu_list = []
        cnt = 0
        for model in self.model_list:
            # print(cnt, '/',len(self._model_list), x_test.shape)
            cnt+=1
            # 判断是回归模型还是分类模型
            is_classification = hasattr(model, 'predict_proba') or hasattr(model, 'classes_')
            if is_classification:  # 分类模型
                baseline_score = accuracy_score(y_test, model.predict(x_test))
                score_func = accuracy_score
            else:  # 回归模型
                baseline_score = mean_squared_error(y_test, model.predict(x_test))
                score_func = mean_squared_error
            feature_importances = []
            for i in range(x_test.shape[1]):
                X_test_perturbed = x_test.copy()
                np.random.shuffle(X_test_perturbed[:, i])
                perturbed_score = score_func(y_test, model.predict(X_test_perturbed))
                if is_classification:
                    importance = baseline_score - perturbed_score
                else:
                    importance = perturbed_score - baseline_score
                feature_importances.append(importance)

            self.pertuba_accu_list.append(feature_importances)
        # 基于软/硬投票，获取平均特征重要性

    def calculate_shap_importance(self, x_train: np.ndarray, x_test: np.ndarray, sample_size=100, workers=4, method='soft'):
        def calculate_shap_for_model(model, x_train, x_test, sample_size):
            if isinstance(model, RandomForestClassifier):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(x_test)
                result = np.abs(shap_values).mean(axis=0)
            elif isinstance(model, nn.Module):
                explainer = shap.DeepExplainer(model, torch.FloatTensor(x_train))
                shap_values = explainer.shap_values(torch.FloatTensor(x_test))
                if isinstance(shap_values, list):
                    shap_values = np.sum(np.abs(shap_values), axis=0)
                result = np.abs(shap_values).mean(axis=0)
            elif isinstance(model, (GaussianNB, XGBClassifier)):
                background = x_train[np.random.choice(x_train.shape[0], sample_size, replace=False)]
                explainer = shap.KernelExplainer(model.predict_proba, background)
                shap_values = explainer.shap_values(x_test)
                result = np.abs(shap_values).mean(axis=0)
            else:
                background = shap.kmeans(x_train, 10)
                explainer = shap.KernelExplainer(model.predict, background)
                shap_values = explainer.shap_values(x_test)
                result = np.abs(shap_values).mean(axis=0)
            return result
        # 创建一个锁对象
        lock = threading.Lock()
        sample_size = min(sample_size, x_train.shape[0])
        self.shap_accu_list = []

        with ThreadPoolExecutor(max_workers=workers) as executor:  # 调整max_workers以适应你的系统
            futures = [executor.submit(calculate_shap_for_model, model, x_train, x_test, sample_size) for model in self.model_list]
            for future in as_completed(futures):
                results = future.result()
                with lock:
                    self.shap_accu_list.append(results)

    #----------------------------------------
    # 获取内部参数方法

def grid_params_search(param_model_pairs:list):
    '''独立方法，对于每一个模型进行超参数筛选'''
    results = []
    for i in range(len(param_model_pairs)):
        param_model_pairs[i][0].fit(param_model_pairs[i][1])
        results.append(param_model_pairs[i][0].get_result())
    return results

def merge_multiple_2d_lists_with_labels(lists):
    merged_list = []
    for idx, lst in enumerate(lists):
        for row in lst:
            # 在每行的末尾添加一个标记，标记该行来自于第几个列表（从1开始）
            merged_list.append(row + [idx])
    return merged_list

def sort_merged_list(merged_list, column_index):
    # 如果 column_index 是负数，计算倒数列的索引
    if column_index < 0:
        column_index = len(merged_list[0]) + column_index
    return sorted(merged_list, key=lambda x: x[column_index])

def group_by_column(data, column_index):
    # 创建一个默认字典，其中值为列表
    grouped_data = defaultdict(list)

    # 遍历数据，将每行添加到相应的组中
    for row in data:
        if column_index < 0:
            # 从右侧开始计数
            adjusted_index = len(row) + column_index
        else:
            # 从左侧开始计数
            adjusted_index = column_index

        # 检查索引是否在当前行的范围内
        if 0 <= adjusted_index < len(row):
            key = row[adjusted_index]
            grouped_data[key].append(row)
        else:
            # 如果索引不在当前行的范围内，则跳过该行
            continue

    # 将字典转换为列表
    result = list(grouped_data.values())
    return result

def convert_to_number(s):
    """
    尝试将字符串转换为数字（整数或浮点数）。
    如果转换失败，则返回原字符串。
    """
    try:
        if '.' in s:
            return float(s)
        else:
            return int(s)
    except ValueError:
        return s

def get_key(row, column_index):
    """
    获取排序键值，处理负数索引和数据转换。
    """
    try:
        # 处理负数索引
        if column_index < 0:
            index = len(row) + column_index
        else:
            index = column_index
        return convert_to_number(row[index]) if len(row) > index else ""
    except IndexError:
        return ""

def sort_nested_list(nested_list, column_index, reverse=False):
    """
    对嵌套列表按指定列排序，支持负数索引，并处理行数据不等长情况。
    同时处理字符串、整数和浮点数排序。

    :param nested_list: List[List[str]] 需要排序的嵌套列表
    :param column_index: int 指定的列索引
    :param reverse: bool 是否倒序排序
    :return: List[List[str]] 排序后的嵌套列表
    """
    try:
        # 使用sorted函数和lambda表达式对嵌套列表按指定列排序
        sorted_list = sorted(nested_list, key=lambda row: (isinstance(get_key(row, column_index), str), get_key(row, column_index)), reverse=reverse)
        return sorted_list
    except IndexError:
        print(f"Error: Column index {column_index} is out of range.")
        return nested_list

def merge_results(lists:list[list], reserve_model_size=0, reserve_r2_ratio=-sys.float_info.max) -> Tuple[list[list], list]:
    '''数据预处理, reserve_accu_ratio 正确率阈值（可能是r2，或者RMSE等）'''
    def is_num(value) -> bool:
        if isinstance(value, numbers.Number):
            return not math.isnan(value)
        elif isinstance(value, str):
            # 使用正则表达式检查字符串是否是合法的数字（包括整数和浮点数）
            return bool(re.match(r'^-?\d+(\.\d+)?$', value))
        else:
            return False
    
    merged_list = merge_multiple_2d_lists_with_labels(lists)
    # 排序
    merged_list = sort_nested_list(merged_list, -2, True)
    
    # 重组还原
    
    # 确保r2值小于1
    if(reserve_r2_ratio > 1):
        raise ValueError("Invalid reserve_r2_ratio")
    # 确保模型输入数值合法
    if(reserve_model_size < 0):
        raise ValueError("Invalid reserve_data_size")
    # 默认值0 不限制模型大小
    elif(reserve_model_size == 0):
        reserve_model_size = len(merged_list)

    reconstruct_model_list = [[] for _ in range(len(lists))] # 结果
    # 如果通过r2保留的模型列表，其尺寸仍然大于reserve_data_size的限制，则进一步限缩模型大小，但当其取值为0时，表示使用所有模型
    model_count = 0
    for model in merged_list:
        # 有时，收敛失败导致模型结果异常，需要排除
        if(is_num(model[-2])):
            # 正确性保留阈值
            if(model_count < reserve_model_size and float(model[-2]) > reserve_r2_ratio):
                reconstruct_model_list[model[-1]].append(model[:-2])
                model_count += 1        # 确保保留的模型尺寸不超过规定的最大模型限制 
    
    # 展开的结果 
    cross_results = []
    for model in merged_list:
        cross_results.append('\t'.join(model[:-1]))  # 将子列表转换为字符串并用空格分隔
    
    return reconstruct_model_list, cross_results


def essemble_model_builder_core(x_data:np.ndarray, y_data:np.ndarray, params_list, model_list, max_reserved_model_size, reserve_r2_threshod) -> Tuple[voting_model, list]:
    '''电池集成模型构建'''
    def find_best_sing_model(essemble_results:list[list[list]]):
        '''子程序 从各个结果中找到单一模型最优结果'''
        best_result = []
        for results_in_sig_model in essemble_results:
            if(len(results_in_sig_model) > 0):
                best_result.append(sort_nested_list(results_in_sig_model, -1, True)[0])
            else:
                best_result.append(["Empty"])
        return best_result
    
    def rebuild_models(actual_models) -> list:
        '''子程序：重组模型, 构建模型列表，用于集成模型输入'''
        models = []
        for block in actual_models:
            if(len(block) > 0):
                for model in block:
                    # 防止有些可能的模型筛选出来为0
                    models.append(model)
        # actual_models = models
        return models
    
    selected_params_list = []
    for param_block in range(len(params_list)):
        model = type(model_list[param_block])()
        grid_params = grid_param_builder(params_list[param_block])
        model.fit(x_data, y_data, grid_params)
        selected_params_list.append(model.get_residual_param().tolist())
    
    # 每一种类的模型中最好的模型
    best_sigle_model_params = find_best_sing_model(selected_params_list)
    print('\n------------------------------------------\nBest model in each single kinds:')
    print(best_sigle_model_params)
    print("--------------------------------------------")
    
    # 筛选较好的数据，如果没有规定集成模型大小，则使用所有模型参数(此判断在方法merge_results中进行)
    # if(max_reserved_model_size == 0):
    #     for model_param_list in selected_params_list:
    #         max_reserved_model_size += len(model_param_list)
    
    # 模型参数列表，通过交叉验证的优劣排序
    selected_params_list, cross_results = merge_results(selected_params_list, max_reserved_model_size, reserve_r2_threshod)
    actual_models = []
    for i in range(len(selected_params_list)):
        actual_models.append(model_list[i].return_full_model(selected_params_list[i]))
    
    # 重新合并模型
    actual_models = rebuild_models(actual_models)
    print('Actual essemble model size: %d, max_reserved_size:%d'%(len(actual_models), max_reserved_model_size))
    
    # 构建新集成模型
    e_model = voting_model(actual_models)
    
    # 返回集成模型(未经训练)
    return e_model, cross_results

def essemble_builder(x_data: np.ndarray ,y_data: np.ndarray, max_essemble_submodel_size=0, essemble_r2_reserve_threshod=0.9) -> Tuple[voting_model, list]:
    # 网格参数选择
    mlp_params = mlp_torch_param.copy()
    mlp_params[0] = [x_data.shape[1]]
    mlp_params[4] = [len(set(y_data))]
    # 参数列表
    params_list = [xgb_sk_params, ada_param, knn_param, lr_param, rf_param, svm_param, mlp_params]
    # 模型列表
    model_list = [self_xgb(), self_ada(), self_knn(), self_lr(), self_rf(), self_svm(), self_mlp_torch()]
    # 参数列表
    # params_list = [lr_param, knn_param, mlp_params]
    # 模型列表
    # model_list = [self_lr(), self_knn(), self_mlp_torch()]
    # 参数列表
    params_list = [lr_param, mlp_params]
    # 模型列表
    model_list = [self_lr(), self_mlp_torch()]

    # params_list = [svm_param]
    # model_list = [self_svm()]
    # --------------------------------------------------------------
    # 模型构建
    e_model, cross_results = essemble_model_builder_core(x_data, y_data, params_list, model_list, max_essemble_submodel_size, essemble_r2_reserve_threshod)
    return e_model, cross_results

#------------------------------------------------------------
# 降维部分
def perform_pca(X, n_components):
    print("开始PCA降维...")
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    print(f"PCA降维后的数据维度: {X_reduced.shape}")
    return X_reduced, pca

def perform_kernel_pca(X, n_components, kernel='rbf'):
    print("开始KernelPCA降维...")
    kp = KernelPCA(n_components, kernel=kernel, fit_inverse_transform=True)
    X_reduced = kp.fit_transform(X)
    print(f"KernelPCA降维后的数据维度: {X_reduced.shape}")
    return X_reduced, kp

def perform_truncated_pca(X, n_components):
    print("开始TruncatedSVD降维...")
    tp = TruncatedSVD(n_components) 
    X_reduced = tp.fit_transform(X)
    print(f"TruncatedSVD降维后的数据维度: {X_reduced.shape}")
    return X_reduced, tp

def dimension_reduction(x_data:np.ndarray, reduction_method:str, n_components=1000):
    # 选择降维方法
    if reduction_method == 'pca':
        X_reduced, reducer = perform_pca(x_data, n_components=n_components)
    elif reduction_method == 'KernelPCA':
        X_reduced, reducer = perform_kernel_pca(x_data, n_components=n_components)
    elif reduction_method == 'TruncatedSVD':
        X_reduced, reducer = perform_truncated_pca(x_data, n_components=n_components)
    else:
        raise ValueError("Unsupported reduction method")
    return X_reduced, reducer

# 类别不平衡处理部分
def balance_classes_smote(X : np.ndarray, y : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    print("使用SMOTE处理类别不平衡...")
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    print(f"平衡后的类别分布: {np.bincount(y_balanced)}")
    return X_balanced, y_balanced

def balance_classes_borderlinesmote(X, y) -> Tuple[np.ndarray, np.ndarray]:
    print("使用BorderlineSMOTE处理类别不平衡...")
    bsmote = BorderlineSMOTE(random_state=42)
    X_balanced, y_balanced = bsmote.fit_resample(X, y)
    print(f"平衡后的类别分布: {np.bincount(y_balanced)}")
    return X_balanced, y_balanced

def balance_classes_adasyn(X, y) -> Tuple[np.ndarray, np.ndarray]:
    print("使用ADASYN处理类别不平衡...")
    adasyn = ADASYN(random_state=42)
    try:
        X_balanced, y_balanced = adasyn.fit_resample(X, y)
        print(f"平衡后的类别分布: {np.bincount(y_balanced)}")
        return X_balanced, y_balanced
    except ValueError as e:
        print(f"ADASYN无法生成样本: {e}")
        print(f"未处理的类别分布: {np.bincount(y)}")
        return X, y

def imbalance_process(x_data:np.ndarray, y_data:np.ndarray, imbalance_method:str) -> Tuple[np.ndarray, np.ndarray]:
    # 选择类别不平衡处理方法
    if imbalance_method == 'smote':
        return balance_classes_smote(x_data, y_data)
    elif imbalance_method == 'borderlinesmote':
        return balance_classes_borderlinesmote(x_data, y_data)
    elif imbalance_method == 'adasyn':
        return balance_classes_adasyn(x_data, y_data)
    else:
        raise ValueError("Unsupported imbalance method: %s"%(imbalance_method))

def remove_multicollinearity(X_encoded, threshold=0.95) -> Tuple[np.ndarray, list]:
    '''消除多重共线性'''
    print("开始移除多重共线性特征...")
    X_dense = X_encoded.astype(np.float32)
    corr_matrix = np.corrcoef(X_dense, rowvar=False)
    abs_corr_matrix = np.abs(corr_matrix)
    upper_triangle_indices = np.triu_indices_from(abs_corr_matrix, k=1)
    high_corr_var_indices = np.where(abs_corr_matrix[upper_triangle_indices] > threshold)[0]
    cols_to_remove = np.unique(upper_triangle_indices[1][high_corr_var_indices])
    cols_to_keep = [i for i in range(X_dense.shape[1]) if i not in cols_to_remove]
    X_no_multicollinearity = X_dense[:, cols_to_keep]
    print(f"移除共线性特征后的数据维度: {X_no_multicollinearity.shape}")
    return X_no_multicollinearity, cols_to_keep # 保留特征的索引

# 特征重要性映射回原始特征
def map_feature_importance_back(feature_importances:np.ndarray, reducer, X_original:np.ndarray, method:str) -> np.ndarray:
    if method == 'pca':
        print("将特征重要性映射回原始特征 (PCA)...")
        X_recovered = reducer.inverse_transform(feature_importances)
    elif method == 'KernelPCA':
        print("将特征重要性映射回原始特征 (KernelPCA)...")
        X_recovered = reducer.inverse_transform(feature_importances)
    elif method == 'TruncatedSVD':
        print("将特征重要性映射回原始特征 (TruncatedSVD)...")
        X_recovered = reducer.inverse_transform(feature_importances)
    else:
        raise ValueError("Unsupported reduction method")
    return X_recovered

def onehot_importance_reflection(importance_list, index_list:list, index_maps:dict) -> list:
    index_kinds = set()
    for index in index_list:
        index_kinds.add(index_maps[index])
    onehot_importances = [0 for _ in range(len(index_kinds))]
    for i in range(len(importance_list)):
        onehot_importances[index_maps[index_list[i]]] += importance_list[i]
    return onehot_importances

def model_bulder(x_data_raw:np.ndarray, y_data:np.ndarray, models_saving_root_dir=os.path.dirname(os.path.abspath(__file__)), dimension_reduction_components=3, reserve_r2_ratio=0.9, max_reserve_model_size=200):
    def check_and_create_directory(path:str, replacement_dir:str) -> str:
        directory_path = Path(path)
        
        # 检查路径是否已经存在
        if directory_path.exists():
            # 如果路径存在，但它不是目录，使用脚本所在目录
            if not directory_path.is_dir():
                return replacement_dir
            return path
        else:
            # 如果路径不存在，且路径是一个目录，则创建该目录
            directory_path.mkdir(parents=True, exist_ok=True)
            return path

    def load_index_map() -> dict:
        '''加载映射回onehot文件的数据'''
        return dict()

    # 特征重要性映射回原始特征
    def map_feature_importance_back(feature_importances:np.ndarray, reducer, method:str) -> np.ndarray:
        # 确保 feature_importances 是 2 维数组
        if feature_importances.ndim == 1:
            feature_importances = feature_importances.reshape(1, -1)
        if method == 'pca':
            print("将特征重要性映射回原始特征 (PCA)...")
            X_recovered = reducer.inverse_transform(feature_importances)
        elif method == 'KernelPCA':
            print("将特征重要性映射回原始特征 (KernelPCA)...")
            X_recovered = reducer.inverse_transform(feature_importances)
        elif method == 'TruncatedSVD':
            print("将特征重要性映射回原始特征 (TruncatedSVD)...")
            X_recovered = reducer.inverse_transform(feature_importances)
        else:
            raise ValueError("Unsupported reduction method")
        return X_recovered
   
    def get_scaler(method:str):
        if(method == 'MinMaxScaler'):
            return MinMaxScaler()
        elif(method == 'StandardScaler'):
            return StandardScaler()
        else:
            raise ValueError("Error Scaler")
    
    # 构建合法的存储目录
    models_saving_root_dir = check_and_create_directory(models_saving_root_dir, os.path.dirname(os.path.abspath(__file__)))
    print('Model Records Storage dir:%s'%(models_saving_root_dir))
    # 降维方式
    reduction_methods = ['pca', 'KernelPCA', 'TruncatedSVD']
    # 不平衡数据处理方式
    imbalance_methods = ['smote', 'borderlinesmote', 'adasyn']
    scaler_methods = ['MinMaxScaler','StandardScaler']
    #------------------------------------------------------
    # onehot索引分类
    index_maps = load_index_map()

    #  不需要通过此方法降低共线性，因为onehot使得特征正交了
    # x_data, remained_feature_pos = remove_multicollinearity(x_data)
    remained_feature_pos = np.arange(0, len(x_data_raw), 1).tolist()
    # 数据归一化
    for scaler in scaler_methods: 
        std_standarder = get_scaler(scaler)
        x_data = std_standarder.fit_transform(x_data_raw)
        # print("归一化后的数据范围: [{}, {}]".format(x_data.min(), x_data.max()))
        # 遍历降维方法和不平衡数据填充方法
        for reduction_method in reduction_methods:
            for imbalance_method in imbalance_methods:
                print('--------------------------------------------------------------------')
                print('--------------------------------------------------------------------')
                print('--------------------------------------------------------------------')
                print('--------------------------------------------------------------------')
                 
                a = time.time()
                # 降维
                x_reduced, reducer = dimension_reduction(x_data, reduction_method, dimension_reduction_components)
                # 重采样
                x_resampled, y_resampled = imbalance_process(x_reduced, y_data, imbalance_method)
                print(x_resampled.shape, y_resampled.shape)
                # 切分训练集，验证集
                x_resampled, x_test, y_resampled, y_test = train_test_split(x_resampled, y_resampled, test_size=0.3, random_state=42)
                # 构建集成模型
                e_model, cross_results = essemble_builder(x_resampled, y_resampled, max_reserve_model_size, reserve_r2_ratio)
                # 模型进行训练
                e_model.fit(x_resampled, y_resampled)
        
        
                #--------------------------------------------------------------------
                # 以下计算特征重要性代码，不是把把都需要算的，而是在确定好模型后再算
                # 交叉验证完结果后，进行模型训练
                #print('Model build finished', time.time()-a, 's')
                ##--------------------------------------------------------------------
                #e_model.calculate_perturbation_accuracy(x_test, y_test)
                #print('Per built')
                #e_model.calculate_shap_importance(x_resampled[:int(len(x_resampled)/10)], x_test, workers=4)
                #print('SHAP built')
                ## 特征重要性集成，获取平均特征重要性列表
                ## 构建特征重要性            
                #per_importance_list = e_model.regenerate_feature_importance(reducer, e_model.pertuba_accu_list, x_test, y_test, 'soft', reduction_method)
                #shap_importance_list = e_model.regenerate_feature_importance(reducer, e_model.shap_accu_list, x_test, y_test, 'soft', reduction_method)
                ## # 特征重要些还原到原始特征中
                #per_importance_list_re = map_feature_importance_back(per_importance_list, reducer, reduction_method)
                #shap_importance_list_re = map_feature_importance_back(shap_importance_list, reducer, reduction_method)
                #print('特征列表1:')
                #print(per_importance_list_re)
                #print('特征列表2:')
                #print(shap_importance_list_re)
                #print('--------------------------------------------------------------------')
                #print('--------------------------------------------------------------------')
                #print('--------------------------------------------------------------------')
                #print('--------------------------------------------------------------------')
                ## 重新映射到onehot中
                # onehot_importance_per = onehot_importance_reflection(per_importance_list_re, remained_feature_pos, index_maps)
                # onehot_importance_shap = onehot_importance_reflection(shap_importance_list_re, remained_feature_pos, index_maps)
                
        
        
        
                # ---------------------------------------------------------------
                # 存储模型
                root_name = models_saving_root_dir + '/' + scaler + '_' + reduction_method+'_'+imbalance_method
                model_storage_name = root_name+'_model.pkl' 
                reducer_storage_name = root_name+'_reducer.pkl' 
                scalar_storage_name = root_name+'_scalar.pkl' 
                model_params_list_name = root_name+'_model_params_list.txt' 
                model_basic_info_name = root_name+'_model_basic_info.txt'
                
                # 打印模型列表
                with open(model_params_list_name, 'w') as f:
                    for line in cross_results:
                        f.writelines(line+'\n')

                # 存储模型
                with open(model_storage_name, 'wb') as f:
                    pickle.dump(e_model, f)
                
                # 存储模型相关说明
                with open(model_basic_info_name, 'w') as f:
                    f.write('Total Model Numbers: %d\n--------------------------------------\n'%(len(e_model.model_list)))
                    for model in e_model.model_list:
                        f.write(str(model)+'\n')

                # 存储放大器
                with open(reducer_storage_name, 'wb') as f:
                    pickle.dump(std_standarder, f)

                # 存储降维器
                with open(scalar_storage_name, 'wb') as f:
                    pickle.dump(reducer, f)
                
                # 清除集成模型中的数据，用于下一轮训练
                e_model.clear_attribute()

def predict_model_1(e_model, e_model_2):

    def load_predict_data():
      ...

    def load_sams_data():
      ...

    x_data, y_data = [], []
    x_data_2, y_data_2 = [], []
    with open('w', 'rb') as f:
        e_model = pickle.load(f)
    e_model.predict(x_data, y_data)

    # y_data_diff = y_data_2-y_data
    # e_model_2.predict(x_data_2)

    # sum = y_predict_1 + y_predict_2


def SAMs_model_builder():
  def load_data():
    ...
    return np.zeros(0), np.zeros(0)


  x_data, y_data = load_data()

  # 训练模型2

  # 全尺寸预测

  # 删除过劣点 R达到0.75以上

  # 重构

def build_cell_models():
    '''入口'''
    def load_data():
        '''读取数据，并处理为标准输入形式（iris集形式）'''
        # 读取
        # 分桶
        # 重采样
        x_data, y_data = load_iris_shuffle()
        return x_data, y_data
    
    # 数据生成和预处理部分
    def generate_onehot_data(n_samples, n_features, n_classes):
        np.random.seed(42)
        data = np.random.randint(2, size=(n_samples, n_features))
        labels = np.random.randint(n_classes, size=n_samples)
        print(f"原始数据维度: {data.shape}")
        return data, labels
    
    # 保留模型r2阈值（0.9）
    reserve_r2_ratio=0.9
    # 保留模型最大数目，0代表保留所有模型
    max_reserve_model_size=0
    
    # 降维后的维数
    n_components = 3
    # 加载数据
    n_samples = 42000
    n_features = 15000
    n_classes = 24
    x_data, y_data = generate_onehot_data(n_samples, n_features, n_classes)

    # 原始数据 
    x_data, y_data = load_data()
    # 结果记录的目录
    record_dir = os.path.dirname(os.path.abspath(__file__))+'/record_dir'
    # 构建模型1
    model_bulder(x_data, y_data, record_dir, n_components, reserve_r2_ratio, max_reserve_model_size)


def build_SAMs_model():
    def load_data():
        '''读取数据，并处理为标准输入形式（iris集形式）'''
        # 读取
        # 分桶
        # 重采样
        x_data, y_data = load_iris_shuffle()
        return x_data, y_data
    
    # 数据生成和预处理部分
    def generate_onehot_data(n_samples, n_features, n_classes):
        np.random.seed(42)
        data = np.random.randint(2, size=(n_samples, n_features))
        labels = np.random.randint(n_classes, size=n_samples)
        print(f"原始数据维度: {data.shape}")
        return data, labels
    
    # 保留模型r2阈值（0.9）
    reserve_r2_ratio=0.7
    # 保留模型最大数目，0代表保留所有模型
    max_reserve_model_size=0
    
    # 降维后的维数
    n_components = 3
    # 加载数据
    n_samples = 42000
    n_features = 15000
    n_classes = 24
    x_data, y_data = generate_onehot_data(n_samples, n_features, n_classes)
    record_dir = os.path.dirname(os.path.abspath(__file__))+'/record_dir_sams'
    # 构建SAMs的输入
    cell_model_path = ''
    cell_model_scaler = ''
    cell_model_reducer = ''
    with open(cell_model_path, 'rb') as f:
        cell_e_model:voting_model = pickle.load(f)
    y_predict = cell_e_model.predict(x_data)
    y_input = y_data-y_predict
    model_bulder(x_data, y_input, record_dir, n_components, reserve_r2_ratio, max_reserve_model_size)

def model_rating_and_importance_analysis():
    def load_data():
        '''读取数据，并处理为标准输入形式（iris集形式）'''
        # 读取
        # 分桶
        # 重采样
        x_data, y_data = load_iris_shuffle()
        return x_data, y_data

    # 特征重要性映射回原始特征
    def map_feature_importance_back(feature_importances:np.ndarray, reducer, method:str) -> np.ndarray:
        # 确保 feature_importances 是 2 维数组
        if feature_importances.ndim == 1:
            feature_importances = feature_importances.reshape(1, -1)
        if method == 'pca':
            print("将特征重要性映射回原始特征 (PCA)...")
            X_recovered = reducer.inverse_transform(feature_importances)
        elif method == 'KernelPCA':
            print("将特征重要性映射回原始特征 (KernelPCA)...")
            X_recovered = reducer.inverse_transform(feature_importances)
        elif method == 'TruncatedSVD':
            print("将特征重要性映射回原始特征 (TruncatedSVD)...")
            X_recovered = reducer.inverse_transform(feature_importances)
        else:
            raise ValueError("Unsupported reduction method")
        return X_recovered

    # 测试数据 
    x_test, y_test = load_data() 
    x_resampled = x_test.copy()
    # 背景数据（用于重要性分析）
    cell_model_path = ''
    cell_model_scaler = ''
    cell_model_reducer = ''
    
    # 获取模型
    with open(cell_model_path, 'rb') as f:
        e_model:voting_model = pickle.load(f) 
    # 获取降维器，scalar    
    with open(cell_model_path, 'rb') as f:
        scalar = pickle.load(f) 
    with open(cell_model_path, 'rb') as f:
        reducer = pickle.load(f) 
    
    reduction_method = ''

    # 以下计算特征重要性代码，不是把把都需要算的，而是在确定好模型后再算
    #--------------------------------------------------------------------
    e_model.calculate_perturbation_accuracy(x_test, y_test)
    print('Per built')
    e_model.calculate_shap_importance(x_resampled[:int(len(x_resampled)/10)], x_test, workers=4)
    print('SHAP built')
    # 特征重要性集成，获取平均特征重要性列表
    # 构建特征重要性            
    per_importance_list = e_model.regenerate_feature_importance(reducer, e_model.pertuba_accu_list, x_test, y_test, 'soft', reduction_method)
    shap_importance_list = e_model.regenerate_feature_importance(reducer, e_model.shap_accu_list, x_test, y_test, 'soft', reduction_method)
    # # 特征重要些还原到原始特征中
    per_importance_list_re = map_feature_importance_back(per_importance_list, reducer, reduction_method)
    shap_importance_list_re = map_feature_importance_back(shap_importance_list, reducer, reduction_method)
    print('特征列表1:')
    print(per_importance_list_re)
    print('特征列表2:')
    print(shap_importance_list_re)
    # 重新映射到onehot中
    onehot_importance_per = onehot_importance_reflection(per_importance_list_re, remained_feature_pos, index_maps)
    onehot_importance_shap = onehot_importance_reflection(shap_importance_list_re, remained_feature_pos, index_maps)
                
if __name__ == "__main__":
    build_cell_models()
    build_SAMs_model()
    model_rating_and_importance_analysis() 
