import sys
import os
from  datetime import datetime
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import ElasticNet, Lasso, RidgeClassifier

from sklearn.metrics import r2_score
from utils import grid_param_builder, load_iris_shuffle, imbalance_process
from general_model import general_model
from params import lr_param

class self_lr(general_model):
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
        if(int(param[0]) == 0):     # Elastic
            param[4] = 'balanced'
            param[5] = 'auto'
        elif(int(param[0]) == 1):   # Lasso
            param[3] = '0.5'
            param[4] = 'balanced'
            param[5] = 'auto'
        elif(int(param[0]) == 2):   # Ridge
            param[2] = 'cyclic'
            param[3] = '0.5'
        return param

    def _build_clf(self, param : np.ndarray):
        if(int(param[0]) == 0):
            return ElasticNet(alpha=float(param[1]), selection=param[2], l1_ratio=float(param[3]))
        elif(int(param[0]) == 1):
            return Lasso(alpha=float(param[1]), selection=param[2])
        elif(int(param[0]) == 2):
            if(param[4] == 'None'):
                return RidgeClassifier(alpha=float(param[1]), class_weight=None, solver=param[5])
            else:
                return RidgeClassifier(alpha=float(param[1]), class_weight=param[4], solver=param[5])

def main() ->int:
    para_list = grid_param_builder(lr_param)
    print(para_list.shape)
    x_data, y_data = load_iris_shuffle()
    model = self_lr(argumentation_method='smote')
    model.fit(x_data, y_data, para_list)
    model.save_residual_params(abs_dir + '/lr_grid_result.txt')
    return 0

if __name__ == "__main__":
    abs_dir = os.path.dirname(os.path.abspath(__file__))
    main()
