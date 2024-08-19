import sys
import os
import random
import datetime

from sklearn import datasets
from sklearn.neural_network import MLPClassifier
import numpy as np

from utils import grid_param_builder, load_iris_shuffle, remove_list_ele, imbalance_process
from sklearn.model_selection import StratifiedKFold
from general_model import general_model
from params import mlp_sk_weak_list, mlp_param_list
from sklearn.metrics import r2_score



'''
# https://blog.csdn.net/weixin_38278334/article/details/83023958
此MLP用作弱分类器(基本废止)
使用的超参数和范围
hidden_layer_sizes tuple 作为弱分类器 一至两层 数量 0 20 50 100
activate logistic 作为基分类器{'identity','logistic','tanh','relu'}
solver : {'lbfgs','sgd','adam'}
alpha 0 (正则化参数) 0 0.0001 0.0002 0.0004 0.0008
batch_size 64 128 256 512 1024 2048(使用auto,默认值)
learning_rate : 当sgd时(使用自适应 adaptive)
learning_rate_init 初始学习率, 使用默认值0.001, 自动调参
power_t: invscaling时,学习率衰减速率
max_iter 最大迭代次数 指定为5000
shuffle 默认true
monument 0.7 0.8 0.9 1 1.1 1.2
beta 1 beta 2 使用默认值
n_iter_no_change 最大无变化早停数 设为30(默认10)
https://zhuanlan.zhihu.com/p/675570928
'''

def uniform_params_generator(size : int) -> np.ndarray:
    '''超参数发生器（均匀分布）'''
    random.seed(datetime.datetime.now().microsecond)
    def inner_generator() -> np.ndarray:
        '''产生一个随机数列'''
        param = np.zeros(17)
        param[0] = random.randint(0,2)                  # activation  均匀分布
        param[1] = random.randint(0,1)                  # solver(sgd adam)
        param[2] = 0                                    # L2 punishment (alpha)
        #param[3] = pow(2, np.random.randint(4,11))     # batch_size
        param[3] = 512                                  # batch_size
        #param[4] = random.uniform(0,2)                 # learning_rate
        param[4] = 2                                    # learning_rate(固定为自适应)
        param[5] = random.uniform(0, 0.003)             # learning_rate_init
        param[6] = random.uniform(0.2, 1.2)             # power_t
        param[7] = 3500                                 # max_iter(固定3500轮)
        param[8] = random.uniform(0.2,0.95)             # momentum
        if(param[8] > 0 and param[1] == 1):
            param[9] = random.uniform(0,1)              # nesterovs_momentum
        else:
            param[9] = 0
        param[10] = random.uniform(0,1)                 # early_stopping
        if(param[10] == 1):
            param[11] = random.uniform(0.05, 0.25)      # validation_fraction
        else:
            param[11] = 0
        if(param[1] == 1):
            param[12] = random.uniform(0.5,0.999)
            param[13] = random.uniform(0.9,0.9999)
        else:
            param[12] = 0
            param[13] = 0
        for i in range(3):
            param[i+14] = random.randrange(0, 101, 50)
        # hidden_layer
        return param[:,np.newaxis]
    param_list = inner_generator()
    cnt = 1
    while(param_list.shape[1] < size):
        param_list = np.concatenate((param_list, inner_generator()), axis=1)
        param_list = np.unique(param_list, axis=1)
        cnt += 1
    np.savetxt(sys.path[0]+'/self_mlp_para_list.txt', param_list.T, fmt='%0.6f',delimiter='\t')
    print('discard_ratio %0.6f' % (size/cnt)) #被舍弃的比例
    return param_list.T
    #消除重复超参数(转变为字符串再进行比较)

class self_mlp(general_model):
    '''此处为弱分类器'''
    def __init__(self) -> None:
        super().__init__(output_interval =0.01)
    
    def _build_clf(self, param: np.ndarray):
        kind = 0
        if(kind == 0):              # 弱分类器
            layers = remove_list_ele(param[:2].astype(np.int64).tolist(), 0)
            return  MLPClassifier(hidden_layer_sizes = layers, 
                                activation=param[2], solver=param[3], alpha=float(param[4]), 
                                learning_rate=param[5], momentum=float(param[6]), max_iter=5000)
        elif(kind == 1):            # 强分类器
            return MLPClassifier()

    def _param_filter(self, param: np.ndarray) -> np.ndarray:
        param[4] = 0
        if param[0] == '0' and param[1] == '0':
            param[0] = 50
        return param
    
    def _cross_valid_mtd(self, clf, x_data: np.ndarray, y_data: np.ndarray) -> float:
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

def main():
    iris_data = datasets.load_iris()
    x_data = iris_data.data
    y_data = iris_data.target
    hyper_param_list =  grid_param_builder(mlp_sk_weak_list)
    clf = self_mlp()
    clf.fit(x_data, y_data, hyper_param_list)
    clf.save_residual_params(abs_dir + '/mlp_grid_result.txt')

if __name__ == "__main__":
    abs_dir = os.path.dirname(os.path.abspath(__file__))
    main()
