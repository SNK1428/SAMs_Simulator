# https://www.cnblogs.com/pinard/p/6136914.html
# https://blog.csdn.net/TeFuirnever/article/details/99656571 DecisiontreeClassifier参数选择

import os
import numpy as np
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import StratifiedKFold

from utils import grid_param_builder, imbalance_process, remove_list_ele, load_iris_shuffle
from general_model import general_model
from params import ada_param

# 参数 限制使用决策树，mlp 和 决策树
# DecisiontreeClassifier
# MLPClassifier 

class self_ada(general_model):
    def _build_clf(self, param: np.ndarray):
        if(param[0] == '0'):        # 使用决策树作为基分类器
            if(param[8] == 'None'):
                deci_estimator = DecisionTreeClassifier(criterion=param[4], splitter=param[5], min_samples_split=int(param[6]), min_samples_leaf=int(param[7]), class_weight=None)
                return AdaBoostClassifier(estimator=deci_estimator, n_estimators=int(param[1]), algorithm=param[2], learning_rate=float(param[3]))    
            else:                   # 使用MLP作为基分类器（未完成weight部分编写）
                deci_estimator = DecisionTreeClassifier(criterion=param[4], splitter=param[5], min_samples_split=int(param[6]), min_samples_leaf=int(param[7]), class_weight=param[8])
                return AdaBoostClassifier(estimator=deci_estimator, n_estimators=int(param[1]), algorithm=param[2], learning_rate=float(param[3]))    
        elif(param[0] == "1"):
            layers = remove_list_ele(param[9:11].astype(np.int64).tolist(), 0) #去除输入中的零隐藏层，构建单隐藏层
            mlp_estimator = MLPClassifier(hidden_layer_sizes=layers, activation=param[11], solver=param[12], alpha=float(param[13]), learning_rate=param[14],momentum=float(param[15])) 
            return AdaBoostClassifier(estimator=mlp_estimator, n_estimators=int(param[1]), algorithm=param[2], learning_rate=float(param[3]))
    
    def _param_filter(self, param: np.ndarray) -> np.ndarray:
        if(param[0] == '0'):    # 使用决策树
            param[9] = '0'
            param[10] = '10'
            param[11] = 'relu'
            param[12] = 'adam'
            param[13] = '0'
            param[15] = '0.9' 
        elif(param[0] == '1'):  # 使用MLP
            param[4] = 'gini'
            param[5] = 'best'
            param[6] = '2'
            param[7] = '1'
            param[8] = 'balanced'
            # 避免不合法情况全0层出现
            if(param[9] == '0' and param[10] == '0'):
                param[9] = '50'
        return param
    
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

def main() -> None:
    '''示例代码'''
    model = self_ada()
    x_data, y_data = load_iris_shuffle()
    param_list = grid_param_builder(ada_param)
    model.fit(x_data, y_data, param_list)
    model.save_residual_params(abs_dir+'/ada_grid_result.txt')

def demo() -> None:
    x_data, y_data = load_iris_shuffle()
    param = ['1','30','SAMME','0.1','gini','best','2','1','balanced','0','100','identity','adam','0.0','adaptive','0.5']
    layers = remove_list_ele(list(map(int, param[9:11])), 0) #去除输入中的零隐藏层，构建单隐藏层
    mlp_estimator = MLPClassifier(hidden_layer_sizes=layers, activation=param[11], solver=param[12], alpha=float(param[13]), learning_rate=param[14],momentum=float(param[15])) 
    clf =  AdaBoostClassifier(estimator=mlp_estimator, n_estimators=int(param[1]), algorithm=param[2], learning_rate=float(param[3]))        
    clf.fit(x_data, y_data)

if __name__ == "__main__":
    abs_dir = os.path.dirname(os.path.abspath(__file__))
    main()
