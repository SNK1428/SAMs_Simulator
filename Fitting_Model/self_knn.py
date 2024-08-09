# https://blog.csdn.net/weixin_37763870/article/details/105160899
import sys
import os

import time
from queue import Queue

import numpy as np

from sklearn import datasets       #导入数据模块
from datetime import datetime
from sklearn.model_selection import KFold#导入切分训练集、测试集模块
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from utils import grid_param_builder, load_iris_shuffle, load_data_from_path
from general_model import general_model
from params import knn_param

class self_knn(general_model):
    '''KNeighborsClassifier的封装类，用于进行基于交叉验证的超参数筛选'''
    '''https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html'''
    
    def _cross_valid_mtd(self, clf, x_data: np.ndarray, y_data: np.ndarray) -> float:
        return cross_val_score(clf, x_data, y_data, cv=self.cv, scoring='r2').mean()

    def _param_filter(self, param: np.ndarray) -> np.ndarray:
        if param[4] != 'minkowski':
            param[5] = '1'
        return param
    
    def _build_clf(self, param : np.ndarray) -> KNeighborsClassifier:
        if(param[4] == 'minkowski'):
            return KNeighborsClassifier(
                n_neighbors=int(param[0]),weights=param[1],algorithm=param[2], 
                leaf_size=int(param[3]), metric=param[4], p=int(param[5]), n_jobs=-1)
        else:
            return KNeighborsClassifier(
                n_neighbors=int(param[0]),weights=param[1],algorithm=param[2], 
                leaf_size=int(param[3]), metric=param[4], n_jobs=-1)

def demo() -> int:
    b_time = time.time() 
    iris_x, iris_y = load_iris_shuffle()
    param_list = grid_param_builder(knn_param)
    np.random.shuffle(param_list)
    knn_model = self_knn()
    knn_model.fit(iris_x, iris_y,param_list)
    knn_model.save_residual_param(abs_dir + '/knn_grid_result.txt')
    best_model = knn_model.get_best_clf()
    r2 = cross_val_score(best_model, iris_x, iris_y, cv=5, scoring='r2')
    print("R2 (cross vali): %0.3f (+/- %0.3f)" % (r2.mean(), r2.std() * 2))
    print('time:', time.time() - b_time)
    return 0

def main():
    data_src_dir = abs_dir+'/../data/cell_input_data_new_1'
    data_out_dir = abs_dir+'/../data/cell_input_data_new_1/out'
    x_data, y_data = load_data_from_path(2, data_out_dir, data_out_dir, data_src_dir)
    data = np.concatenate((x_data, y_data), axis=1)
    np.random.shuffle(data)
    x_data = data[:,:-8]
    y_data = data[:,-8:]

if __name__ == "__main__":
    abs_dir = os.path.dirname(os.path.abspath(__file__))
    demo()
