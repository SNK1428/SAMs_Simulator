# 标准库导入
from ctypes import Union
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
from typing import Union
# 第三方库导入
import numpy as np
import pandas as pd
from pandas.core.indexes.base import astype_array
from scipy.optimize._lsq.common import print_iteration_nonlinear
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA, SparsePCA, FactorAnalysis, KernelPCA, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB, MultinomialNB, CategoricalNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
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
from utils import grid_param_builder, load_iris_shuffle, write_content_to_file, check_and_create_directory

from params import *

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


def binning_num_arr(y_data:np.ndarray, num_buckets=10, method='uniform', min_bucket_size=5):
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
    
    # check binning range
    new_bucketed_data = np.array(new_bucketed_data)  
    upper_range = dict()
    for i in range(len(y_data)):
        if(new_bucketed_data[i] in upper_range):
            if(y_data[i] > upper_range[new_bucketed_data[i]]):
                upper_range[new_bucketed_data[i]] = y_data[i]
        else:
            upper_range[new_bucketed_data[i]] = y_data[i]
    lower_range = dict()
    for i in range(len(y_data)):
        if(new_bucketed_data[i] in lower_range):
            if(y_data[i] < lower_range[new_bucketed_data[i]]):
                lower_range[new_bucketed_data[i]] = y_data[i]
        else:
            lower_range[new_bucketed_data[i]] = y_data[i]
    data_range = []
    key = list(sorted(upper_range.keys()))
    for i in range(len(key)):
        pair = []
        pair.append(key[i])
        pair.append(lower_range[key[i]])
        pair.append(upper_range[key[i]])
        data_range.append(pair)
    means = sorted(means)
    return new_bucketed_data, means, data_range


def binning_based_on_range(y_data:Union[np.ndarray, list], data_range:list[list[int]]) -> np.ndarray:
    # data_range need to be sorted pos1: classification label pos2 left edge(left-incursive) pos3 right edge
    # build bin range
    # without error check
    sorted(data_range, key=lambda x: x[0])
    bin_range = []
    labels = []
    for i in range(len(data_range)-1):
        if i == 0:
            bin_range.append(-float('inf'))
        bin_range.append(data_range[i+1][1])
        labels.append(data_range[i][0])
    bin_range.append(float('inf'))
    labels.append(data_range[-1][0])
    data_bin = pd.cut(y_data, bin_range, labels=labels, right=False).tolist()
    return np.ndarray(data_bin)

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


def essemble_model_builder_core(x_data:np.ndarray, y_data:np.ndarray, imbalance_method:str, params_list, model_list, max_reserved_model_size, reserve_r2_threshod) -> Tuple[voting_model, list]:
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
        # Initialize the model, and then set the params of argumentation method
        model = type(model_list[param_block])(argumentation_method=imbalance_method)
        grid_params = grid_param_builder(params_list[param_block])
        # Search best hyper params
        model.fit(x_data, y_data, grid_params, imbalance_method)
        selected_params_list.append(model.get_residual_param().tolist())
    
    # 每一种类的模型中最好的模型
    best_sigle_model_params = find_best_sing_model(selected_params_list)
    print('\n------------------------------------------\nBest model in each single kinds:')
    print(best_sigle_model_params)
    print("--------------------------------------------")
    
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

def essemble_builder(x_data: np.ndarray ,y_data: np.ndarray, imbalance_method:str, max_essemble_submodel_size=0, essemble_r2_reserve_threshod=0.9) -> Tuple[voting_model, list]:
    # 网格参数选择
    mlp_params = mlp_torch_param.copy()
    mlp_params[0] = [x_data.shape[1]]
    mlp_params[4] = [len(set(y_data))]
    # 参数列表
    params_list = [xgb_sk_params, ada_param, knn_param, lr_param, rf_param, svm_param, mlp_params]
    # 模型列表
    model_list = [self_xgb(), self_ada(), self_knn(), self_lr(), self_rf(), self_svm(), self_mlp_torch()]
    # 参数列表
    params_list = [lr_param, mlp_params]
    # 模型列表
    model_list = [self_lr(), self_mlp_torch()]

    # --------------------------------------------------------------
    # 模型构建
    e_model, cross_results = essemble_model_builder_core(x_data, y_data, imbalance_method, params_list, model_list, max_essemble_submodel_size, essemble_r2_reserve_threshod)
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
    elif reduction_method == 'None':
        return x_data, None
    else:
        raise ValueError("Unsupported reduction method")
    return X_reduced, reducer

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

def model_builder(x_data_raw:np.ndarray, y_data:np.ndarray, models_saving_root_dir:str,
                reduction_methods:list[str], imbalance_methods:list[str], scaler_methods:list[str], 
                dimension_reduction_components=3, reserve_r2_ratio=0.9, max_reserve_model_size=200):
    
    #  特征重要性映射回原始特征
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
    #------------------------------------------------------
    # remove multi-colliear datas
    # x_data, remained_feature_pos = remove_multicollinearity(x_data)
    # 数据归一化
    for scaler in scaler_methods: 
        std_standarder = get_scaler(scaler)
        x_data = std_standarder.fit_transform(x_data_raw)
        # print("归一化后的数据范围: [{}, {}]".format(x_data.min(), x_data.max()))
        # 遍历降维方法和不平衡数据填充方法
        for reduction_method in reduction_methods:
            for imbalance_method in imbalance_methods:
                print('--------------------------------------------------------------------')
                reducer = None
                #降维
                if(len(reduction_methods) != 0):
                    x_reduced, reducer = dimension_reduction(x_data, reduction_method, dimension_reduction_components)
                else:
                    x_reduced = x_data
                # 构建集成模型
                e_model, cross_results = essemble_builder(x_data, y_data, imbalance_method, max_reserve_model_size, reserve_r2_ratio)
                # 模型进行训练(Using argumentated data)
                e_model.fit(x_reduced, y_data)
        
                # ---------------------------------------------------------------
                # 存储模型
                root_name = models_saving_root_dir + '/' + scaler + '_' + reduction_method+'_'+imbalance_method
                model_storage_name = root_name+'_model.pkl' 
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
                with open(scalar_storage_name, 'wb') as f:
                    pickle.dump(std_standarder, f)

                # 存储降维器
                if(reducer is not None):
                    reducer_storage_name = root_name+'_reducer.pkl' 
                    with open(reducer_storage_name, 'wb') as f:
                        pickle.dump(reducer, f)
                
                # 清除集成模型中的数据，用于下一轮训练
                e_model.clear_attribute()

def build_cell_models(x_data, y_data, max_reserve_model_size, reserve_r2_ratio, model_saving_dir:str, 
                      reduction_methods:list, imbalance_methods:list, scaler_methods:list, reduce_components = 3):
    # 保留模型r2阈值（0.9）
    # 保留模型最大数目，0代表保留所有模型
    # reduce_components: 降维后维数
    # 数据生成和预处理部分
    # def generate_onehot_data(n_samples, n_features, n_classes):
    #     np.random.seed(42)
    #     data = np.random.randint(2, size=(n_samples, n_features))
    #     labels = np.random.randint(n_classes, size=n_samples)
    #     print(f"原始数据维度: {data.shape}")
    #     return data, labels
    
    # 临时数据
    #---------------------------------------------------- 
    # 加载数据
    # n_samples = 42000
    # n_features = 15000
    # n_classes = 24
    # x_data, y_data = generate_onehot_data(n_samples, n_features, n_classes)
    # 原始数据 
    x_data, y_data = load_data()

    #-----------------------------------------------------
    
    # 结果记录的目录
    # model_saving_dir = os.path.dirname(os.path.abspath(__file__))+'/record_dir'
    # 构建模型1
    model_builder(x_data, y_data, model_saving_dir,reduction_methods, imbalance_methods, scaler_methods, reduce_components, reserve_r2_ratio, max_reserve_model_size)

def build_SAMs_model(x_data, y_data, mole_data:np.ndarray, max_reserve_model_size, reserve_r2_ratio, cell_model_path, model_saving_dir:str):
    # 数据生成和预处理部分
    def generate_onehot_data(n_samples, n_features, n_classes):
        np.random.seed(42)
        data = np.random.randint(2, size=(n_samples, n_features))
        labels = np.random.randint(n_classes, size=n_samples)
        print(f"原始数据维度: {data.shape}")
        return data, labels
    
    #---------------------------------
    # 降维后的维数
    n_components = 3
    # 加载数据
    n_samples = 42000
    n_features = 15000
    n_classes = 24
    x_data, y_data = generate_onehot_data(n_samples, n_features, n_classes)
    #---------------------------------

    model_saving_dir = os.path.dirname(os.path.abspath(__file__))+'/record_dir_sams'
    # 构建SAMs的输入
    with open(cell_model_path, 'rb') as f:
        cell_e_model:voting_model = pickle.load(f)
    # 差减值
    y_predict = cell_e_model.predict(x_data)
    y_input = y_data-y_predict
    # 模型构建

    # 降维方式：在外部确定，因为SAMs不需要
    reduction_methods = ['None']
    # 不平衡数据处理方式
    imbalance_methods = ['smote', 'borderlinesmote', 'adasyn']
    scaler_methods = ['MinMaxScaler','StandardScaler']
    # 去除数据多重共线性 
    x_data,_ = remove_multicollinearity(x_data)
    
    model_builder(x_data, y_input, model_saving_dir, reduction_methods, imbalance_methods, scaler_methods, n_components, reserve_r2_ratio, max_reserve_model_size)

def load_data():
    '''读取数据，并处理为标准输入形式（iris集形式）'''
    # 读取
    # 分桶
    # 重采样
    x_data, y_data = load_iris_shuffle()
    return x_data, y_data

def model_rating_and_importance_analysis(x_train:np.ndarray, x_test:np.ndarray, y_test:np.ndarray, model_path:str, reducer_path:str, onehot_reflection_list:None):
    # 评价每一个模型的特征重要性
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
    
    def get_reducer_type() -> str:
      ...

    # 获取模型
    with open(model_path, 'rb') as f:
        e_model:voting_model = pickle.load(f) 
    # 获取降维器，scalar    
    with open(model_path, 'rb') as f:
        scalar = pickle.load(f) 
    with open(model_path, 'rb') as f:
        reducer = pickle.load(f) 
    
    reduction_method = ''

    # 以下计算特征重要性代码，不是把把都需要算的，而是在确定好模型后再算
    #--------------------------------------------------------------------
    e_model.calculate_perturbation_accuracy(x_test, y_test)
    print('Per built')
    e_model.calculate_shap_importance(x_train[:int(len(x_train)/10)], x_test, workers=4)
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
    if(onehot_reflection_list is not None):
        onehot_importance_per = onehot_importance_reflection(per_importance_list_re, onehot_reflection_list, index_maps)
        onehot_importance_shap = onehot_importance_reflection(shap_importance_list_re, onehot_reflection_list, index_maps)

device_data_path = ''
mole_device_data_path=''
mole_data_path = ''
predict_data_path = ''
onehot_reflection_data_path = ''

cell_training_data_path = ''
cell_test_data_path = ''
sam_training_data_path = ''
sam_test_data_path = ''

cell_model_dir = ''
sams_model_dir = ''

predict_result_saving_path = ''

actual_cell_model_path = ''
actual_sams_model_path =''
actual_cell_reducer_path = ''
actual_cell_scalar_path = ''
actual_sams_reducer_path = ''
acatual_sams_scalar_path = ''

binning_means_path = ''
binning_range_path = ''

cell_reserve_r2_ratio=0.9
sam_reserve_r2_ratio=0.75
cell_model_max_size =1000
sam_model_max_size = 1000


reduction_component = 1800  # components reserved scale in dimention reduction procedure

def build_cell_model(cell_x_data:np.ndarray, cell_y_data:np.ndarray):
    # 降维方式：在外部确定，因为sams不需要
    cell_reduction_methods = ['pca', 'kernelpca', 'truncatedsvd']
    # 不平衡数据处理方式
    cell_imbalance_methods = ['smote', 'borderlinesmote', 'adasyn']
    # 
    cell_scaler_methods = ['MinMaxScaler','StandardScaler']
    model_builder(cell_x_data, cell_y_data, cell_model_dir, cell_reduction_methods, cell_imbalance_methods, cell_scaler_methods, reduction_component, cell_reserve_r2_ratio, cell_model_max_size)

def build_sams_model(cell_x_data:np.ndarray, cell_y_data:np.ndarray, mole_x_data:np.ndarray):
    # 降维方式：在外部确定，因为SAMs不需要
    sams_reduction_methods = ['None']
    # 不平衡数据处理方式
    sams_imbalance_methods = ['smote', 'borderlinesmote', 'adasyn']
    # rescale the data
    sams_scaler_methods = ['MinMaxScaler','StandardScaler']
    # load model of cell
    with open(actual_cell_model_path, 'rb') as f:
        cell_model:voting_model = pickle.load(f)
    # build valid_target_value 
    cell_performance = cell_model.predict(cell_x_data) 
    mole_y_diff = cell_y_data - cell_performance
    # search best model
    model_builder(mole_x_data, mole_y_diff,sams_model_dir, sams_reduction_methods, sams_imbalance_methods, sams_scaler_methods, reduction_component, sam_reserve_r2_ratio, sam_model_max_size)

def rating_feature_importance():
    # load training and test data from given path
    ... 
    # load model from given path
    with open(actual_cell_model_path, 'rb') as f:
        cell_model:voting_model = pickle.load(f)
    with open(actual_sams_model_path, 'rb') as f:
        sams_model:voting_model = pickle.load(f) 
    
    # load reducer
    with open(actual_cell_model_path, 'rb') as f:
        cell_reducer = pickle.load(f)
    with open(actual_sams_model_path, 'rb') as f:
        sams_reducer:voting_model = pickle.load(f) 
   
    # load scalar
    with open(actual_cell_model_path, 'rb') as f:
        cell_model:voting_model = pickle.load(f)
    with open(actual_sams_model_path, 'rb') as f:
        sams_model:voting_model = pickle.load(f) 
    
    # load reflector
    with open(actual_cell_model_path, 'rb') as f:
        cell_model:voting_model = pickle.load(f)
    with open(actual_sams_model_path, 'rb') as f:
        sams_model:voting_model = pickle.load(f) 
    
    # ---------------------------------
    # rating cell model features
    model_rating_and_importance_analysis(cell_x_train, cell_x_test, cell_y_test, cell_model, actual_cell_scalar_path, actual_cell_reducer_path, feature_reflection_list) 
    # rating sams model features
    model_rating_and_importance_analysis(mole_x_train, mole_x_test, mole_y_test, sams_model, actual_sams_reducer_path, acatual_sams_scalar_path, feature_reflection_list)

def predict_data():
    '''using given essemble model for data predicting'''
    def predict_data_core(model:voting_model, x_data:np.ndarray, y_data:np.ndarray, means):
        ...
    # get datas from file
    cell_data_predict = np.loadtxt(predict_data_path)
    cell_x_data_predict = np.concatenate((cell_data_predict[:, :50], cell_data_predict[:,59:]), axis=1)
    cell_y_data_predict = cell_data_predict[:, 51:58]
    # load mole data again
    mole_data = np.loadtxt(mole_data_path)
    
    # load model
    with open(actual_cell_model_path, 'rb') as f:
        cell_model:voting_model = pickle.load(f)
    with open(actual_sams_model_path, 'rb') as f:
        sam_model:voting_model = pickle.load(f) 

    # predict avg cell performance
    cell_result = cell_model.predict(cell_x_data_predict)

    # got results of mole data
    sam_result = sam_model.predict(mole_data)

    # sort both data, find best comparasion(mark the original sequence)
    cell_index_array = np.arange(cell_result.shape[0]).reshape(-1, 1)
    sam_index_array = np.arange(sam_result.shape[0]).reshape(-1, 1)
    cell_result = np.hstack((cell_result, cell_index_array))
    sam_result = np.hstack((sam_result, sam_index_array))
    cell_result = cell_result[cell_result[:, 0].argsort()] 
    sam_result = sam_result[sam_result[:, 0].argsort()]
    write_content_to_file(predict_result_saving_path+'cell_result.txt', cell_result)
    write_content_to_file(predict_result_saving_path+'sam_result.txt', cell_result)

def main():
    # cell data
    cell_device_data = np.loadtxt(device_data_path)
    cell_x_data = np.concatenate((cell_device_data[:, :50], cell_device_data[:,59:]), axis=1)
    cell_y_data = cell_device_data[:, 51:58]
    # sams data
    sam_device_data = np.loadtxt(mole_device_data_path)
    sam_cell_x_data = np.concatenate((sam_device_data[:, :50], sam_device_data[:,59:]), axis=1)
    sam_cell_y_data = sam_device_data[:, 51:58]
    mole_data = np.loadtxt(mole_data_path) 
     
    #-----------------------------------
    # binning
    cell_y_data, cell_mean_list, binning_range = binning_num_arr(cell_y_data)
    # saving means and range data from binning
    write_content_to_file(binning_means_path, cell_mean_list)
    write_content_to_file(binning_range_path, binning_range)
    
    sam_cell_y_data = binning_based_on_range(sam_cell_y_data, binning_range)
    
    #-----------------------------------
    # build train and test data
    cell_x_train, cell_x_test, cell_y_train, cell_y_test = train_test_split(cell_x_data, cell_y_data, test_size=0.2, random_state=42, stratify=cell_y_data) 
    # build train and test data for sams
    # first merge
    device_data_size = sam_cell_x_data.shape[1]
    merged_mole_data = np.concatenate((sam_cell_x_data, mole_data), axis=1)
    merged_cell_x_train, merged_cell_x_test, sam_cell_y_train, sam_cell_y_test = train_test_split(merged_mole_data, sam_mole_y_data, test_size=0.15, random_state=42, stratify=mole_y_data)
    # devide the merged results 
    sam_cell_x_train = merged_cell_x_train[:, :device_data_size]
    sam_cell_x_test = merged_cell_x_test[:, :device_data_size]
    mole_x_train = merged_cell_x_train[:, device_data_size:]
    mole_x_test = merged_cell_x_test[:, device_data_size:]
    # saving devided training and test data
    ...
    selection = 1
    if(selection == 1):
        # cell model selection
        build_cell_model(cell_x_train, cell_y_train)
    elif(selection == 2):
        # mole model selection
        build_sams_model(sam_cell_x_train, sam_cell_y_train, mole_x_train, mole_data)
    elif(selection == 3):
        # rating
        rating_feature_importance()
    elif(selection == 4):
        predict_data()

if __name__ == "__main__":
    main()
