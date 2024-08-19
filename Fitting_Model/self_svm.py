#SVM超参数构建 网格搜索
import sys
import os

from datetime import datetime

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC, SVC      

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)      # 关闭轮数自动增加警告

from utils import grid_param_builder, load_iris_shuffle, str_2_bool
from general_model import general_model
from params import svm_param

class self_svm(general_model):
    def __init__(self) -> None:
        super().__init__(output_interval=0.01)

    def _cross_valid_mtd(self, clf, x_data: np.ndarray, y_data: np.ndarray) -> float:
        return cross_val_score(clf, x_data, y_data, cv=self.cv, scoring='r2').mean()
    
    def _param_filter(self, param: np.ndarray) -> np.ndarray:
        if(param[0] == 'linear'):   # linearSVC
            param[7] = '10'
            param[8] = '5'
            param[9] = 'auto'
            param[10] = '5'
            param[11] = 'True'
            param[12] = 'True'
            param[13] = 'None'
            param[14] = 'ovo'
            param[15] = 'True'
            if(param[1] == 'l1' and param[3] == 'hinge'):
                param[3] = 'squared_hinge'
            if(param[1] == 'l1' and param[3] == 'squared_hinge'):
                param[4] = 'False'
            if(param[1] == 'l2' and param[3] == 'hinge'):
                param[4] = 'True'
            return param
        else:                       # SVC
            param[1] = 'l1'
            param[2] = '0.3'
            param[3] = 'squared_hinge'
            param[4] = 'True'
            param[5] = 'ovr'
            param[6] = 'None'
            if(param[14] == 'ovo'):
                param[15] = 'False'
            return param 

    def _build_clf(self, param: np.ndarray):
        if(param[0] == 'linear'):
            if(param[4] != 'auto'):
                if(param[13] == 'None'):
                    return LinearSVC(penalty=param[1], C=float(param[7]), loss=param[3], dual=str_2_bool(param[4]), multi_class=param[5], class_weight=None)
                else:
                    return LinearSVC(penalty=param[1], C=float(param[7]), loss=param[3], dual=str_2_bool(param[4]), multi_class=param[5], class_weight=param[13])
            else:
                if(param[13] == 'None'):
                    return LinearSVC(penalty=param[1], C=float(param[7]), loss=param[3], dual=param[4], multi_class=param[5], class_weight=None)
                else:
                    return LinearSVC(penalty=param[1], C=float(param[7]), loss=param[3], dual=param[4], multi_class=param[5], class_weight=param[13])
        else:
            if(param[13] == 'None'):
                return SVC(C=float(param[7]), degree=int(param[8]), gamma=param[9], coef0=int(param[10]), shrinking=str_2_bool(param[11]), 
                       probability=str_2_bool(param[12]), class_weight=None, decision_function_shape=param[14], break_ties=str_2_bool(param[15]))
            else:
                return SVC(C=float(param[7]), degree=int(param[8]), gamma=param[9], coef0=int(param[10]), shrinking=str_2_bool(param[11]), 
                       probability=str_2_bool(param[12]), class_weight=param[13], decision_function_shape=param[14], break_ties=str_2_bool(param[15]))

def main() -> int:
    param_list = grid_param_builder(svm_param)
    x_data, y_data = load_iris_shuffle()
    model = self_svm()
    model.fit(x_data,y_data, param_list)
    best_clf = model.best_clf
    model.save_residual_params(abs_dir + '/svm_grid_params.txt')
    scores = cross_val_score(best_clf, x_data, y_data, scoring='r2')
    print("R2(cross vali): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return 0

if __name__ == "__main__":
    abs_dir = os.path.dirname(os.path.abspath(__file__))
    main()
