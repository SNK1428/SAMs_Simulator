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
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold

from utils import grid_param_builder, load_iris_shuffle, imbalance_process
from general_model import general_model
from params import knn_param

class self_knn(general_model):
    '''KNeighborsClassifier的封装类，用于进行基于交叉验证的超参数筛选'''
    '''https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html'''
    
    def _cross_valid_mtd(self, clf, x_data: np.ndarray, y_data: np.ndarray) -> float:
        # cross valiadtion with argumentation of training data
        accuracy_scores = []
        cv = StratifiedKFold(n_splits=self._fold_num)
        for train_index, test_index in cv.split(x_data, y_data):
            x_train, x_test = x_data[train_index], x_data[test_index]
            y_train, y_test = y_data[train_index], y_data[test_index]
            x_resampled, y_resampled = imbalance_process(x_train, y_train, self.argumentation_method) 
            clf.fit(x_resampled, y_resampled)
            y_pred = clf.predict(x_test)
            accuracy_scores.append(r2_score(y_test, y_pred))

        return float(np.mean(accuracy_scores)) 


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

def main() -> int:
    b_time = time.time() 
    iris_x, iris_y = load_iris_shuffle()
    param_list = grid_param_builder(knn_param)
    np.random.shuffle(param_list)
    knn_model = self_knn(fold_num=5, argumentation_method='smote')
    knn_model.fit(iris_x, iris_y,param_list)
    knn_model.save_residual_params(abs_dir + '/knn_grid_result.txt')
    best_model = knn_model.best_clf
    r2 = cross_val_score(best_model, iris_x, iris_y, cv=5, scoring='r2')
    print("R2 (cross vali): %0.3f (+/- %0.3f)" % (r2.mean(), r2.std() * 2))
    print('time:', time.time() - b_time)
    return 0

if __name__ == "__main__":
    abs_dir = os.path.dirname(os.path.abspath(__file__))
    main()
