import sys
import os

from numpy import ndarray
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from utils import grid_param_builder, load_iris_shuffle
from general_model import general_model
from params import rf_param

class self_rf(general_model):
    def __init__(self) -> None:
        super().__init__(output_interval = 0.05)
    
    def _build_clf(self, param: ndarray):
        if(param[5] == 'None'):
            return RandomForestClassifier(
                n_estimators=40, n_jobs=-1, criterion=param[0], max_depth=int(param[1]), min_samples_leaf=int(param[2]), 
                min_samples_split=int(param[3]), max_features=param[4], class_weight=None)
        else:
            return RandomForestClassifier(
                n_estimators=40, n_jobs=-1, criterion=param[0], max_depth=int(param[1]), min_samples_leaf=int(param[2]), 
                min_samples_split=int(param[3]), max_features=param[4], class_weight=param[5])
    def _param_filter(self, param: ndarray) -> ndarray:
        return param

    def _cross_valid_mtd(self, clf, x_data: ndarray, y_data: ndarray) -> float:
        return cross_val_score(clf, x_data, y_data, cv=5, scoring='r2').mean()

def main() -> int:
    para_list = grid_param_builder(rf_param)
    x_data, y_data = load_iris_shuffle()
    rf_clf = self_rf()
    rf_clf.fit(x_data, y_data, para_list)
    rf_clf.save_residual_param(abs_dir + '/rf_grid_result.txt')
    return 0

if __name__ == "__main__":
   abs_dir = os.path.dirname(os.path.abspath(__file__))
   main()

