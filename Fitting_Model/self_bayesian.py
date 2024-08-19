# 无需超参数筛选
# https://zhuanlan.zhihu.com/p/366787872
import sys
import os
import numpy as np

from sklearn.model_selection import cross_val_score   #导入数据模块
from sklearn.naive_bayes import GaussianNB, MultinomialNB, CategoricalNB, ComplementNB
from sklearn.metrics import r2_score
from utils import grid_param_builder, load_iris_shuffle, str_2_bool, imbalance_process
from general_model import general_model
from params import BO_param

# ComplementNB
from sklearn.model_selection import StratifiedKFold


class self_BO(general_model):
    '''KNeighborsClassifier的封装类，用于进行基于交叉验证的超参数筛选'''
    '''https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html'''

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

    def _param_filter(self, param: np.ndarray) -> np.ndarray:
        # print(param)
        if(param[0] == '0'):        # Gaussian
            param[1] = '0'
            param[2] = '0'
        elif(param[0] == "1"):      # Multi
            param[2] = '0'
        elif(param[0] == "2"):      # Catego
            param[2] = '0'
        return param

    def _build_clf(self, param : np.ndarray):
        if(param[0] == '0'):
            return GaussianNB()
        elif(param[0] == "1"):
            return MultinomialNB(alpha=float(param[1]))
        elif(param[0] == "2"):
            return CategoricalNB(alpha=float(param[1]), force_alpha=True)
        elif(param[0] == "3"):
            return ComplementNB(alpha=float(param[1]), norm=str_2_bool(param[2]),force_alpha=True)

def main() -> int:
    x_data, y_data = load_iris_shuffle()
    param_list = grid_param_builder(BO_param)
    bo_clf = self_BO(fold_num=5, argumentation_method='smote')
    bo_clf.fit(x_data, y_data, param_list)
    bo_clf.save_residual_params(abs_dir + '/bayesian_grid_result.txt')
    return 0

if __name__ == "__main__":
    abs_dir = os.path.dirname(os.path.abspath(__file__))
    main()
