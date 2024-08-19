import time
import os
import sys
from typing import Tuple
import pickle
from datetime import datetime
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE

from typing import Union
import itertools as it
import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import datasets       #导入数据模块

def convert_to_numbers(input_list):
    output_list = []
    for item in input_list:
        if isinstance(item, str):
            # 将字符串转换为数字
            number = float(item) if '.' in item else int(item)
            output_list.append(number)
        else:
            # 如果已经是数字，直接添加到输出列表
            output_list.append(item)
    return output_list


def write_content_to_file(file_path:str,data:Union[list,np.ndarray,torch.Tensor], seperator='\t',fmt='%s', mode = 'w') -> None:
    '''将数据写入文件 2D以下 写入txt 以上写入.npy, 不可用numpy存储的数据按列表写入,  mode（部分实现）: w写入 a覆盖写入'''
    #获取文件标题头 file_path 无需检查，savetxt会报错
    file_path = os.path.splitext(file_path)
    #定义的lambda函数 用于写入numpy矩阵
    def record_matrix_arr(data_np : np.ndarray):
        if(len(data_np.shape) <= 2):
            np.savetxt(file_path[0]+'.txt', data_np, delimiter=seperator,fmt=fmt)#保留三位小数
        else:#写入npy文件
            np.save(file_path[0], data_np, allow_pickle=True)
    def record_list(data : list, mode = 'w'):
        file_exist = os.path.isfile(file_path[0]+'.txt')    # 判断文件是否存在
        with open(file_path[0]+".txt", mode) as f:
            if mode == 'a' and file_exist:
                f.write('\n')       # 给新加入的文件以新行起步
            for i in range(len(data)):
                f.write(seperator.join(map(str,data[i])))
                if(i < len(data)-1):
                    f.write("\n")
    #写入不同类型的数据
    if(isinstance(data, list)):#list 尺寸大于2 则存储为npy
        try:
            data = np.array(data)
            record_matrix_arr(data)
        except ValueError:
            record_list(data, mode)
    elif(isinstance(data, torch.Tensor)):    #torch
        record_matrix_arr(data.clone().cpu().numpy())
    elif(isinstance(data, np.ndarray)):    #numpy
        record_matrix_arr(data)

def generate_data(path_seq_str:str, path_seq_num:str, path_scatter_string:str, path_scatter_number:str,
                    path_mole:str, path_outputs:str) ->list:
    '''
    和generate_data_4处理数据的逻辑完全相同，只是不存储数组，而是直接将数据返回，因为numpy的问题\n
    将数据从标准文件\n
        output_data_original.txt\n
        scatter_number_data_original.txt\n
        scatter_string_data_original.txt\n
        seq_number_data_original.txt\n
        seq_string_data_original.txt\n
    获取，进行对齐等预处理操作\n
    并组织成为list数组，其中包含的元素是torch\n
    '''
    
    ###############################################################
    #lambda方法
    def open_data_txt(path_data):
        with open(path_data, "r") as f:
            data = f.read()
            data = data.split('\n')
        for i in range(len(data)):
            data[i] = data[i].split('\t')
        return data    

    ###############################################################

    print("generate has been execute : ")
    t_begin = time.time()
    data_seq_str = open_data_txt(path_seq_str)
    data_seq_num = open_data_txt(path_seq_num)
    data_sca_str = open_data_txt(path_scatter_string)
    data_sca_num = open_data_txt(path_scatter_number)
    data_out = open_data_txt(path_outputs)
    # 归一化output data

    data_seq_str_reshape = []
    data_seq_num_reshape = []
    split_num = len(data_sca_str)  # 分割长度
    # print(split_num)
    for i in range(0, len(data_seq_str), split_num):
        data_seq_str_reshape.append(data_seq_str[i:i+split_num])
    for i in range(0, len(data_seq_num), split_num):
        data_seq_num_reshape.append(data_seq_num[i:i+split_num])
    data_seq_str = data_seq_str_reshape
    data_seq_num = data_seq_num_reshape
    del data_seq_str_reshape
    del data_seq_num_reshape
    # 合并
    for i in range(len(data_seq_str)):
        data_seq_str[i] = np.array(data_seq_str[i])
        data_seq_num[i] = np.array(data_seq_num[i])
        # print(data_seq_str[i].shape, data_seq_num[i].shape)
        data_seq_str[i] = np.hstack((data_seq_str[i], data_seq_num[i]))
        data_seq_str[i] = data_seq_str[i][data_seq_str[i] !=
                                          ''].astype(np.float32)  # 去除空字符串（来源于str seq中的空格）
        data_seq_str[i] = data_seq_str[i].reshape(len(data_sca_str),
                                                  # 去除完成后，原始数组被展开，需要reshape(不使用-1,在于确定空格数是否被平行批量删除，而不是文件内部存在bug的空字符)
                                                  int(len(data_seq_str[i])/len(data_sca_str)))
    # 创建分子信息（空）
    data_mole = np.zeros((len(data_sca_str), 167)).astype(np.float32)
    data_seq_str.append(data_mole)
    data_seq_str.append(np.array(data_sca_str).astype(np.float32))
    data_seq_str.append(np.array(data_sca_num).astype(np.float32))
    data_seq_str.append(np.array(data_out).astype(np.float32))

    for i in range(len(data_seq_str)):
        data_seq_str[i] = torch.tensor(data_seq_str[i])
    print('generate_data_5 execute time : %s ms' % ((time.time() - t_begin)*1000))
    return data_seq_str

def load_data_from_path(method : int, data_load_path : str, data_out_path = "", data_source_path = "") -> tuple[torch.Tensor, torch.Tensor]:
    '''
    参数\n
    method 1 : 按旧方式（input_data.npy） 读取数据 2 按新方式（data_in data_out）+ 数据预处理方式读取数据\n
    data_load_path 如果存在，则从此文件夹读取data_in 和data_out 文件(方法1，2)\n
    data_out_path 输出数据的路径(方法2专用)\n
    data_source_path 加载原始文件的路径 给generate_data_5用(方法2专用)\n
    
    如果原始文件一致，两种方法等效 因为新版numpy不支持存储不等长数组，故需要直接返回（也就是必须使用方法2）
    这一次以后，所有的数据不处理为input_data的集成形式，而改为input和output分离的，二维数组的处理形式
    返回值也改为data_raw和labels_raw 相较于原函数更进一步
    标准读取/写入文件形式
        旧版本 input_data.npy（读取，写入通过generate_data_4()） 
        新版本 data_in.npy data_out.npy(写入(data_out_path)/读取(data_load_path))
    '''
    
    def data_pre(data_ori : np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        lambda方法
        处理旧的数据（numpy形式,generate_data_4产生） 为第一种处理方法
        处理新的数据（list,generate_data_5产生） 为第二种处理方法
        '''
        data_in = data_ori[0]  # 输入数据
        # print(data_in.shape)
        if isinstance(data_ori, np.ndarray):
            for i in range(1, data_ori.shape[0] - 4):
                data_in = torch.cat((data_in, data_ori[i]), dim=1)
            data_out = data_ori[data_ori.shape[0] - 1]  # 输出总数据
            #print(data_in.shape)
            return data_in, data_out
        elif isinstance(data_ori, list):
            for i in range(1, len(data_ori)-4):
                data_in = torch.cat((data_in, data_ori[i]), dim=1)
            data_out = torch.Tensor(data_ori[len(data_ori)-1])
            return data_in, data_out

    #旧方法：直接读取input_data.npy 前提是generate_data_4能够正常工作
    if (method == 1):
        #data_path = data_load_path
        data_ori =  np.load(data_load_path, allow_pickle=True)
        return data_pre(data_ori)
    elif (method == 2):
        #如果待读取的文件存在，则直接读取并返回
        if(os.path.exists(data_load_path+'/data_in.npy') and os.path.exists(data_load_path+'/data_out.npy')):
            data_in =  np.load(data_load_path+'/data_in.npy')
            data_out = np.load(data_load_path+'/data_out.npy')
            return torch.from_numpy(data_in),torch.from_numpy(data_out)
        else:    
            path_seq_str = data_source_path+"/seq_string_data_original.txt"
            path_seq_num = data_source_path+"/seq_number_data_original.txt"
            path_scatter_string = data_source_path+"/scatter_string_data_original.txt"
            path_scatter_number = data_source_path+"/scatter_number_data_original.txt"
            path_mole = data_source_path+"/mole_data_original.txt"
            path_outputs = data_source_path+"/output_data_original.txt"
            data_ori = generate_data(
                path_seq_str, path_seq_num, path_scatter_string, 
                path_scatter_number, path_mole, path_outputs)
            data_in,data_out =  data_pre(data_ori)
            #存储数据
            np.save(data_out_path+'/data_in.npy', data_in.detach().cpu().numpy())
            np.save(data_out_path+'/data_out.npy', data_out.detach().cpu().numpy())
            return data_in,data_out

def data_pca_process(pca_model_path : str, sc_model_path : str, X_data : torch.Tensor, process_method = 0) -> np.ndarray:
    '''
    用于数据的降维操作
    pca_model_path : 读取/写入pca模型
    sc_model_path : 配合PCA模型的StandardScaler模型路径
    X_data 待降维数据
    process_method 
        0(any) 写入pca路径，但是如果路径存在文件，则直接读取模型
        1 强制覆盖写入pca文件
    保留置信0.9 基本可以保留641个数据
    '''
    def pca_process(this_pca : PCA, is_write=True) -> np.ndarray:
        '''
        PCA实际处理过程
        this_pca 使用的pca
        is_write 是否记录pca
        '''
        sc = StandardScaler()       # 数据归一化方法
        print('standard scaler begin ...')
        X_data_std = sc.fit_transform(X_data)
        if(is_write):
            print('actual PCA begin ...')
            X_data_std_pca = this_pca.fit_transform(X_data_std)
            print('PCA record begin ... ')
            # 存储PCA模型
            with open(pca_model_path, 'wb') as f:
                pickle.dump(this_pca, f)
            # 存储归一化模型
            with open(sc_model_path, 'wb') as f:
                pickle.dump(sc, f)
        else:
            X_data_std_pca = this_pca.transform(X_data_std)
        print('PCA finished')
        return X_data_std_pca
    
    if(process_method == 1):                #强制更新PCA    
        print('PCA model 1')
        if(os.path.exists(pca_model_path)):
            os.remove(pca_model_path)
        return pca_process(PCA(n_components=0.87))
    if(os.path.exists(pca_model_path)):     #如果文件存在，则读取，否则写入
        print('PCA model 0 with PCA files')
        with open(pca_model_path, 'rb') as f:
            pca = pickle.load(f)
            f.close()
        return pca_process(pca, False)                    # 其他未定义数字
    else:                                   # 如果文件不存在，写入pca，并且返回被处理完的数据s
        print('PCA model 0 without PCA files')
        return pca_process(PCA(n_components=0.87))

def equal_range_binning(data_tar : torch.Tensor, devide_size : int, min_length : int):
    '''
    等间隔分桶 data_tar 用于分桶的数据
    '''
    # 等距
    data_tar = torch.squeeze(data_tar)
    data_tar = data_tar.numpy()
    data_dis_bins = pd.cut(np.squeeze(data_tar),
                           bins=devide_size).codes  # 17类(bins=80) 73类（320）
    data_dis_bins = data_dis_bins.copy().astype(np.int64)
    # 贪心匹配，最小长度
    cls_ptr = 0
    cls_count = 0
    tmp_cls = set()
    new_cls_count = 0
    avg_value = 0
    dic = dict()
    while (cls_ptr <= max(data_dis_bins)):
        for i in range(len(data_dis_bins)):
            # 1. 遍历整体找到类
            if (data_dis_bins[i] == cls_ptr):
                cls_count += 1
                # 2. 加入临时数组
                tmp_cls.add(cls_ptr)
                avg_value += data_tar[i]
        # 3. 加入到上限
        if (cls_count > min_length):
            # 替换原先的类图标
            avg_value /= cls_count
            for i in range(len(data_dis_bins)):
                if (data_dis_bins[i] in tmp_cls):
                    data_dis_bins[i] = new_cls_count
                    dic[new_cls_count] = avg_value.item()
            new_cls_count += 1
            tmp_cls = set()
            cls_count = 0
            avg_value = 0
        # 5. 指针向前，重新开始
        cls_ptr += 1
    # 清理尾类
    if cls_count != 0:
        avg_value /= cls_count
        for i in range(len(data_dis_bins)):
            if (data_dis_bins[i] in tmp_cls):
                data_dis_bins[i] = new_cls_count
                dic[new_cls_count] = avg_value.item()
    return dic, data_dis_bins


# 从文本中读取数据
def read_data_from_file(file_name : str) -> list:
    if(os.path.exists(file_name)):
        file = open(file_name, 'r')
        file_data = file.read()
        file.close()
        return file_data.split('\n')
    else:
        return []
    
'''
批量切分字符串数组，使其变为二维list
'''
def split_vector(data : list, seperator='\t') -> list:
    data_np = []
    #依据提供的seperator 切分每一行
    for ele in data:
        line_data = ele.split(seperator)
        data_np.append(line_data)
    return data_np
    
'''
批量切分字符串数组，并将其转变为numpy数组
'''
def split_vector_numpy(data : list, seperator='\t') -> np.ndarray:
    data_np = split_vector(data, seperator)
    # 检查每一行长度是否相同，如果不同，停止程序运行
    line_equal = True
    for line in data_np:
        if(len(line)!=len(data_np[0])):
            line_equal = False
            break
    # 返回numpy结果
    if(line_equal):
        return np.array(data_np)
    else:
        #不合法，报错，退出程序
        print('Fatal Error : could not transfer 2d list to np.array')
        exit(-1)

def remove_list_ele(list_for_delete : list, ele) -> list:
    '''删除list中所有指定值'''
    def remove_item(n):
        return n != ele
    return list(filter(remove_item, list_for_delete))

#-------------------------------------------------

def grid_para_search(range_arr : np.ndarray, iter_num = 0) -> list:
    '''构建网格搜索范围（基本型） range_arr n*2数组，每一列限定了取值数量，行数为总待筛选参数数量，不会出现重复'''
    
    def combine_arr_mode_0(vec : Union[list, np.ndarray], mat :Union[list,np.ndarray]) -> np.ndarray:
        '''为 utils.utils.combine_arr方法中的mode0方法，用于将两个数组进行穷举组合，合并输出为一个二维矩阵
        vec_1 具有 n*1 形式; mat 具有 m*k 形式 不限其元素type，type可为int float char等形式 输出形式为 np.ndarray(n*m, 1+k)数组'''
        combinations = it.product(vec, mat)
        arr = list(combinations)            # 将迭代器转换为列表
        # if(mode == 0):                      # 一亿个参数 44 秒 (np.shape : 10000 , np.shape 10000*5) 2500万 11秒（默认方法）
        r_arr = np.array([arr[i][1] for i in range(len(arr))])
        l_arr = np.array([arr[i][0] for i in range(len(arr))])[:, np.newaxis]
        if len(r_arr.shape) == 1:
            r_arr = r_arr[:, np.newaxis]
        return np.concatenate((l_arr, r_arr), axis=1)

    result_arr = [ele for ele in range(range_arr[iter_num][0], range_arr[iter_num][1])]
    if iter_num < len(range_arr) - 1:
        param = grid_para_search(range_arr, iter_num+1)
        return combine_arr_mode_0(result_arr, param).tolist()
    return result_arr

def grid_param_builder(param_list_range : list) -> np.ndarray:
    '''依据传入的超参数表，返回构建的网格超参数结果（网格搜索，构建超参数列表的入口函数）\n
        示例： 如果传入超参数列表为 list = [[1,2,3],[4,'345']] 则返回 \n
        [['1' '4'], ['1' '345'], ['2' '4'], ['2' '345'], ['3' '4'], ['3' '345']]
        输入值为二维list'''
    start = time.time()     # 计时器
    hyper_param_list_np = list()
    actual_param_list = list()
    hyper_param_range = np.zeros((len(param_list_range), 2), dtype=np.int64)    # 获取每个超参数的个数，作为超参数选择范围的两端
    for i in range(len(param_list_range)):
        hyper_param_range[i][1] = len(param_list_range[i])                      # 构建超参数选择范围
        hyper_param_list_np.append(np.array(param_list_range[i]))
    raw_hyper_param = np.array(grid_para_search(hyper_param_range))                       # 未进行映射的超参数
    for i in range(raw_hyper_param.shape[1]):
        actual_param_list.append(hyper_param_list_np[i][raw_hyper_param[:,i]])
    print('Time consumption of grid params generation : %0.3f s'%(time.time() - start))
    return np.vstack(actual_param_list).T

def load_iris_shuffle():
    '''获取打乱的iris集合'''
    iris = datasets.load_iris()
    iris_x = iris.data
    iris_y = iris.target
    data = np.concatenate((iris_x, iris_y[:,np.newaxis]),axis=1)
    np.random.seed(datetime.now().microsecond)
    np.random.shuffle(data)
    iris_x = data[:, :4]
    iris_y = data[:,4].astype(np.int64)
    np.random.seed(0)
    return iris_x, iris_y

def build_k_fold_data(k_fold : int, data_x : torch.Tensor, data_y : torch.Tensor) -> list:  ###此过程主要是步骤（2）
    '''依据Tensor的input和target数据，构建n折交叉验证集合'''
    if(len(data_x.shape) == 1):
        data_x = torch.unsqueeze(data_x, dim=1)
    if(len(data_y.shape) == 1):
        data_y = torch.unsqueeze(data_y, dim=1)
    k_fold_data = []
    for i in range(k_fold):
        # 返回第i折交叉验证时所需要的训练和验证数据，分开放，X_train为训练数据，X_valid为验证数据
        assert k_fold > 1
        fold_size = data_x.shape[0] // k_fold  # 每份的个数:数据总条数/折数（组数）
        X_train, y_train = None, None
        for j in range(k_fold):
            idx = slice(j * fold_size, (j + 1) * fold_size)  #slice(start,end,step)切片函数
            ##idx 为每组 valid
            X_part, y_part = data_x[idx, :], data_y[idx]
            if j == i: ###第i折作valid
                X_valid, y_valid = X_part, y_part
            elif X_train is None:
                X_train, y_train = X_part, y_part
            else:
                X_train = torch.cat((X_train, X_part), dim=0) #dim=0增加行数，竖着连接
                y_train = torch.cat((y_train, y_part), dim=0)
        # 格式转换
        
        k_fold_data.append([X_train, y_train, X_valid,y_valid])
    return k_fold_data

def str_2_bool(bool_str : str) -> bool:
    '''布尔字符串转布尔值'''
    if(bool_str == 'True'):
        return True
    elif(bool_str == "False"):
        return False
    else:
        raise ValueError("Invalid bool str value: %s"%(bool_str))

# 类别不平衡处理部分
def balance_classes_smote(X : np.ndarray, y : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # print("SMOTE...")
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    # print(f"Balanced Classes: {np.bincount(y_balanced)}")
    return X_balanced, y_balanced

def balance_classes_borderlinesmote(X, y) -> Tuple[np.ndarray, np.ndarray]:
    # print("BorderlineSMOTE...")
    bsmote = BorderlineSMOTE(random_state=42)
    X_balanced, y_balanced = bsmote.fit_resample(X, y)
    # print(f"平衡后的类别分布: {np.bincount(y_balanced)}")
    return X_balanced, y_balanced

def balance_classes_adasyn(X, y) -> Tuple[np.ndarray, np.ndarray]:
    adasyn = ADASYN(random_state=42)
    try:
        X_balanced, y_balanced = adasyn.fit_resample(X, y)
        # print(f"平衡后的类别分布: {np.bincount(y_balanced)}")
        return X_balanced, y_balanced
    except ValueError as e:
        # print(f"ADASYN无法生成样本: {e}")
        # print(f"未处理的类别分布: {np.bincount(y)}")
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


