# 使用常规网格筛选
import sys
import os
import time

import numpy as np
from xgboost.sklearn import XGBClassifier, XGBRFClassifier #xgboost的 sklearn接口

from utils import grid_param_builder, load_iris_shuffle
from general_model import general_model
from params import xgb_sk_params
from sklearn.model_selection import cross_val_score

# https://zhuanlan.zhihu.com/p/365030773
# https://blog.csdn.net/qq_41076797/article/details/102710299
# https://www.zhihu.com/question/479283391 eta参数 学习率
# https://blog.csdn.net/weixin_43298886/article/details/109523109
xgb_param = [
    ['gbtree','gblinear'],
    [0.01,0.015,0.025,0.05,0.1],
    [0,0.05,0.1,0.5,0.7,0.9,1],
    [0,0.01,0.05,0.1,1],
    [0,0.1,0.5,1],
    [4,6,8,10,12],
    [1,3,5,7],
    [0.3,0.5,0.7]           #base_score 先验分数
]

class self_xgb(general_model):
    def __init__(self) -> None:
        '''打印间隔设为0.5%'''
        super().__init__( output_interval=0.005)
    
    def _cross_valid_mtd(self, clf, x_data: np.ndarray, y_data: np.ndarray) -> float:
        return cross_val_score(clf, x_data, y_data, cv = 5, scoring='r2').mean()
    
    def _param_filter(self, param: np.ndarray) -> np.ndarray:
        if(param[1] == 'gbtree'):
            param[15] = 'shotgun'
            param[16] = 'cyclic'
            param[17] = 'tree'
        elif(param[1] == 'dart'):
            param[16] = 'shotgun'
            param[17] = 'cyclic'
        elif(param[1] == 'gblinear'):
            param[8] = '3'
            param[9] = '0.01'
            param[10] = '1'
            param[11] = '1'
            param[15] = 'tree'
        if(param[1] == 'gblinear' and param[15] == 'tree' and (param[17] not in {'cyclic', 'shuffle'})):
            param[17] = 'cyclic'
        return param

    def _build_clf(self, params: np.ndarray):
        # print(params)
        if(params[1] == 'gbtree'):      # 弱分类器 gbtree
            if(params[0] == '0'):       # XGBClassifier
                # print('mtd1')
                return XGBClassifier(n_estimators=int(params[2]), booster=params[1], reg_alpha=float(params[3]), reg_lambda=float(params[4]), 
                                     objective=params[5], base_score=float(params[6]), max_depth = int(params[8]), learning_rate=float(params[9]), 
                                     gamma=float(params[10]), min_child_weight=int(params[11]),subsample=int(params[12]), verbosity=0,n_jobs=-1)
            else:                    # XGBRFClassifier
                # print('mtd2')
                return XGBClassifier(n_estimators=int(params[2]), booster=params[1], reg_alpha=float(params[3]), reg_lambda=float(params[4]), 
                                     objective=params[5], base_score=float(params[6]), max_depth = int(params[8]), learning_rate=float(params[9]), 
                                     gamma=float(params[10]), min_child_weight=int(params[11]),subsample=int(params[12]), verbosity=0, n_jobs=-1)
        elif(params[1] == 'dart'):      # 弱分类器 dart
            if(params[0] == '0'):       # XGBClassifier
                # print('mtd5')
                return XGBClassifier(n_estimators=int(params[2]), booster=params[1], reg_alpha=float(params[3]), reg_lambda=float(params[4]), 
                                     objective=params[5], base_score=float(params[6]), max_depth = int(params[8]), learning_rate=float(params[9]), 
                                     gamma=float(params[10]), min_child_weight=int(params[11]), subsample=int(params[12]), normalize_type=params[15], verbosity=0, n_jobs=-1)
            else:                       # XGBRFClassifier
                # print('mtd6')
                return XGBClassifier(n_estimators=int(params[2]), booster=params[1], reg_alpha=float(params[3]), reg_lambda=float(params[4]), 
                                     objective=params[5], base_score=float(params[6]), max_depth = int(params[8]), learning_rate=float(params[9]), 
                                     gamma=float(params[10]), min_child_weight=int(params[11]),subsample=int(params[12]), verbosity=0, n_jobs=-1)
        elif(params[1] == 'gblinear'):  # 弱分类器 gblinear
            if(params[0] == '0'):       # XGBClassifier
                # print('mtd3')
                return XGBClassifier(n_estimators=int(params[2]), booster=params[1], reg_alpha=float(params[3]), reg_lambda=float(params[4]), 
                                     objective=params[5], base_score=float(params[6]), updater=params[16], feature_selector=params[17], verbosity=0,n_jobs=-1)
            else:                       # XGBRFClassifier
                # print('mtd4')
                return XGBRFClassifier(n_estimators=int(params[2]), booster=params[1], reg_alpha=float(params[3]), reg_lambda=float(params[4]), 
                                       objective=params[5], base_score=float(params[6]), verbosity=0, n_jobs=-1)

def main()->int:
    x_data, y_data = load_iris_shuffle()
    xgb_param = grid_param_builder(xgb_sk_params)
    clf = self_xgb()
    clf.fit(x_data, y_data, xgb_param)
    clf.save_residual_params(abs_dir+'/xgb_grid_result.txt')
    return 0

def demo_1():
    x_data, y_data = load_iris_shuffle()
    params = ['0','gblinear','10','0.0','0.0','binary:logistic','0.3','0','3','0.01','1','1','1','1','auto','tree','shotgun','greedy']
    # print(int(params[2]), params[1], float(params[3]), float(params[4]), params[5], float(params[6]), params[16], params[17])
    clf =  XGBClassifier(n_estimators=int(params[2]), booster=params[1], reg_alpha=float(params[3]), reg_lambda=float(params[4]), 
                                     objective=params[5], base_score=float(params[6]), updater=params[16], feature_selector=params[17], verbosity=0)
    clf.fit(x_data, y_data)

if __name__ == "__main__":
    abs_dir = os.path.dirname(os.path.abspath(__file__))
    main()
    # demo_1()
    # demo_1()
    # arr = grid_param_builder(xgb_param)
    # print(arr.shape)

