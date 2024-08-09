from math import e
from typing import Tuple
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torcheval.metrics.functional import r2_score

from utils import grid_param_builder, load_iris_shuffle, convert_to_numbers
from general_model import general_model
import utils as utils

# 单独手动调参
mlp_torch_param = [
    [1],    # input_dim         
    [8192],
    [4096],
    [1024],
    [512],
    [512],
    [512],
    [512],
    [512],
    [512], 
    [1],        # output_dim 10
    [0],        # criterion 0 : CrossEntropyLoss
    [0,1]       # optimizer 0 : Adam 1.SGD(使用Adam)
    ]

class pytorch_model_sklearn(nn.Module):
    '''模拟所有sklearn行为的pytorch模型'''
    epoch_max=1000
    test_check_epoch=50
    batch_size = 64
    early_stop_max = 3
    test_ratio=None
    random_state=None
    using_gpu=True
    calc_device = torch.device('cpu')
    criterion = nn.CrossEntropyLoss()
    optimizer = None

    def __init__(self, layers:list[int]) -> None:
        super(pytorch_model_sklearn, self).__init__()
        # 输入需要纯数字，但是兼容以字符串形式存在的数字list
        layers = convert_to_numbers(layers)
        self.layers = nn.ModuleList()
        self.layers.append(nn.BatchNorm1d(layers[0]))
        self.layers.append(nn.Linear(layers[0], layers[1]))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(layers[1], layers[2]))
        self.layers.append(nn.BatchNorm1d(layers[2]))
        self.layers.append(nn.Linear(layers[2], layers[3]))
        self.layers.append(nn.BatchNorm1d(layers[3]))
        self.layers.append(nn.Linear(layers[3], layers[4]))
        self.layers.append(nn.Linear(layers[4], layers[5]))
        self.layers.append(nn.Linear(layers[5], layers[6]))
        self.layers.append(nn.Linear(layers[6], layers[7]))
        self.layers.append(nn.Linear(layers[7], layers[8]))
        self.layers.append(nn.Linear(layers[8], layers[9]))
        self.layers.append(nn.Linear(layers[9], layers[10]))
        self.dropout = nn.Dropout(0.2)
        self.set_fitting_params()
        self.reset_parameters()

    def set_fitting_params(self, optimizer=None, criterion=None, learn_rate=0.001, max_epoch=300, test_size=0.2, test_interval_epoch=10, batch_size = 32, using_gpu=True, early_stop_max=5, random_state=42):
        '''针对fit方法的参数，模仿sklearn'''
        if(optimizer is None):
            self.optimizer = torch.optim.Adam(self.optimizer_params(), lr=learn_rate)
        else:
            self.optimizer = optimizer
        self.epoch_max = max_epoch
        self.test_check_epoch = test_interval_epoch
        self.batch_size = batch_size
        self.test_ratio = test_size
        self.random_state = random_state
        self.early_stop_max = early_stop_max
        self.using_gpu = using_gpu
        self.calc_device = torch.device('cuda' if self.using_gpu and torch.cuda.is_available() else 'cpu')
        if criterion is not None:
            self.criterion = criterion
    
    def reset_parameters(self):
        '''重置所有参数'''
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.constant_(layer.bias, 0.0)

    def optimizer_params(self):
        '''获取此模型中的原始参数，这些参数是用于输入优化器optimizer的'''
        params = []
        for layer in self.layers:
            if isinstance(layer, nn.Module):
                params.extend(list(layer.parameters()))
        return params

    def forward(self, x_data:torch.Tensor):
        '''原始forward方法'''
        x = F.tanh(x_data)
        x = self.layers[0](x)
        x = self.layers[1](x)
        x = self.layers[2](x)
        x = self.layers[3](x)
        x = self.layers[4](x)
        x = self.layers[5](x)
        x = self.layers[6](x)
        x = self.layers[7](x)
        # x = self.dropout(x)
        return x
        x = self.layers[8](x)
        x = self.layers[9](x)
        x = self.layers[10](x)
        x = self.layers[11](x)
        x = self.layers[12](x)
        x = self.layers[13](x)
        return x

    def _build_input_data(self, x_data:np.ndarray, y_data:np.ndarray, test_ratio) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
        num_samples = x_data.shape[0]                       # 样本总量
        indices = np.arange(num_samples)                    # 序列
        np.random.shuffle(indices)                          #  随机打乱数据
        train_size = int((1-test_ratio)*num_samples)        # 训练集尺寸
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        
        train_X = torch.tensor(x_data[train_indices], dtype=torch.float32)
        test_X = torch.tensor(x_data[test_indices], dtype=torch.float32)
        train_y = torch.tensor(y_data[train_indices], dtype=torch.long)
        test_y = torch.tensor(y_data[test_indices], dtype=torch.long)
        return train_X, test_X, train_y, test_y

    def fit(self, x_data:np.ndarray, y_data:np.ndarray):
        '''模型训练'''
        # 数据进入指定设备
        self.calc_device = torch.device('cuda' if self.using_gpu and torch.cuda.is_available() else 'cpu') 
        self.to(self.calc_device)
        # 模型设置
        self.train()
        best_loss = float('inf')
        early_stop_endurance = 0
        # 数据切分
        x_train, x_test, y_train, y_test = self._build_input_data(x_data, y_data, self.test_ratio)
        '''训练方法'''
        for epoch in range(self.epoch_max):
            # 每一轮训练都重新打乱数据
            indices = torch.randperm(x_train.shape[0])
            for start_pos in range(0, x_train.shape[0], self.batch_size):
                # 数据分批进入模型，并且每一批分批进入GPU，防止超显存
                end_pos = min(start_pos + self.batch_size, x_train.shape[0])
                batch_indices = indices[start_pos:end_pos]
                x_train_batch = x_train[batch_indices].to(self.calc_device)
                y_train_batch = y_train[batch_indices].to(self.calc_device)
                # 训练 
                self.optimizer.zero_grad()
                outputs = self.forward(x_train_batch)
                loss = self.criterion(outputs, y_train_batch)
                loss.backward()
                self.optimizer.step()
                torch.cuda.empty_cache()
                del x_train_batch
                del y_train_batch
            if epoch % self.test_check_epoch == 0 and self.test_ratio != 0:
                val_loss, _ = self.evaluate(x_test, y_test)
                # val_loss, val_accuracy = self.evaluate(x_test, y_test)
                # print(f"Epoch {epoch+1}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}%, Endurance: {early_stop_endurance}")
                if val_loss < best_loss:
                    best_loss = val_loss
                    early_stop_endurance = 0
                else:
                    early_stop_endurance += 1
                if early_stop_endurance > self.early_stop_max:
                    # print('Reach best point, stop')
                    break

    def evaluate(self, x_test:torch.Tensor, y_test:torch.Tensor):
        '''评估模型准确性'''
        x_test = x_test.to(self.calc_device)
        y_test = y_test.to(self.calc_device)
        self.to(self.calc_device)
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x_test)
            loss = self.criterion(outputs, y_test)
            val_loss = loss.item()
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == y_test).sum().item()
        accuracy = 100 * correct / y_test.shape[0]
        self.train()
        return val_loss, accuracy
    
    def predict(self, x_data:np.ndarray):
        probabilities = self.predict_proba(x_data)
        _, predicted_classes = torch.max(probabilities, 1)
        return predicted_classes 

    def predict_proba(self, x_data:np.ndarray):
        # 设备统一
        self.to(self.calc_device)
        x_data = torch.tensor(x_data, dtype=torch.float32).to(self.calc_device)
        # 将模型设置为评估模式
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x_data)
            # 根据损失函数类型选择激活函数
            if isinstance(self.criterion, nn.CrossEntropyLoss):
                probs = F.softmax(outputs, dim=1)
            elif isinstance(self.criterion, nn.BCEWithLogitsLoss):
                probs = torch.sigmoid(outputs)
            else:
                raise ValueError("Unsupported criterion for predict_proba")
        return probs

# 单独训练
# 超参数固定（使用2023年8月参数）

class Model(nn.Module):
    '''基于pytorch原始的多层神经网络'''
    def __init__(self, input_dim,hidden_dim_1,hidden_dim_2,hidden_dim_3, hidden_dim_4,
                hidden_dim_5,hidden_dim_6, hidden_dim_7,hidden_dim_8,hidden_dim_9,output_dim):
        super(Model, self).__init__()
        self.BN = nn.BatchNorm1d(input_dim, input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.BN_2_3 = nn.BatchNorm1d(hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, hidden_dim_3)
        self.BN_3_4 = nn.BatchNorm1d(hidden_dim_3)
        self.fc4 = nn.Linear(hidden_dim_3, hidden_dim_4)
        self.fc5 = nn.Linear(hidden_dim_4, hidden_dim_5)
        self.fc6 = nn.Linear(hidden_dim_5, hidden_dim_6)
        self.fc7 = nn.Linear(hidden_dim_6, hidden_dim_7)
        self.fc8 = nn.Linear(hidden_dim_7, hidden_dim_8)
        self.fc9 = nn.Linear(hidden_dim_8, hidden_dim_9)
        self.fc10 = nn.Linear(hidden_dim_9, output_dim)
        self.dropout = nn.Dropout(0.2)
        # 重置模型参数
        self.reset_parameters()

    def reset_parameters(self):
        '''重置模型参数'''
        '''https://deepinout.com/pytorch/pytorch-questions/1_pytorch_reset_parameters_of_a_neural_network_in_pytorch.html'''
        nn.init.constant_(self.BN.bias, 0.0)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.constant_(self.BN_2_3.bias, 0.0)
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.constant_(self.BN_3_4.bias, 0.0)
        nn.init.xavier_normal_(self.fc4.weight)
        nn.init.xavier_normal_(self.fc5.weight)
        nn.init.xavier_normal_(self.fc6.weight)
        nn.init.xavier_normal_(self.fc7.weight)
        nn.init.xavier_normal_(self.fc8.weight)
        nn.init.xavier_normal_(self.fc9.weight)
        nn.init.xavier_normal_(self.fc10.weight)
        nn.init.constant_(self.BN.bias, 0.0)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.constant_(self.fc2.bias, 0.0)
        nn.init.constant_(self.BN_2_3.bias, 0.0)
        nn.init.constant_(self.fc3.bias, 0.0)
        nn.init.constant_(self.BN_3_4.bias, 0.0)
        nn.init.constant_(self.fc4.bias, 0.0)
        nn.init.constant_(self.fc5.bias, 0.0)
        nn.init.constant_(self.fc6.bias, 0.0)
        nn.init.constant_(self.fc7.bias, 0.0)
        nn.init.constant_(self.fc8.bias, 0.0)
        nn.init.constant_(self.fc9.bias, 0.0)
        nn.init.constant_(self.fc10.bias, 0.0)


    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = F.tanh(x)
        #x = self.BN(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        # x = self.BN_2_3(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x
        x = self.relu(x)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.fc6(x)
        x = self.relu(x)
        x = self.fc7(x)
        x = self.relu(x)
        x = self.fc8(x)
        x = self.relu(x)
        x = self.fc9(x)
        x = self.relu(x)
        x = self.fc10(x)
        return x

class self_mlp_torch(general_model):
    '''用于进行超参数搜索的模型'''
    def __init__(self) -> None:
        super().__init__(output_interval = 0.01)
        self.k_fold_data = None
        self.cv = 5             # 交叉验证折数
        self.calc_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 计算时，所用的设备，可能是cuda，也有可能是cpu
        self.host_device = torch.device('cpu')      # CPU设备
        self.using_gpu = True

    def _cross_valid_mtd(self, clf : nn.Module, x_data: np.ndarray, y_data: np.ndarray) -> float:
        '''pytorch交叉验证 clf:Model: 自行定义的Pytorch模型, 类型为自定义类型Model'''
        # 训练数据在先前已经送入GPU
        avg_test_accuracy = []      # K轮交叉验证准确率（K个值）
        avg_test_r2 = []            # K轮交叉验证R2（K个值）
        fold_cnt = 1                # K折计数
        for x_train, y_train, x_test, y_test in self.k_fold_data:
            # clf 清除历史参数
            # train_avg_loss = 0.0          # 训练集loss，用于判断是否还在下降
            best_accuracy = -sys.maxsize     # 测试集分类准确率，用于评价泛化性能           
            best_r2 = -sys.maxsize
            best_epoch = 0
            # 模型反复重置参数进行交叉验证
            for epoch in range(clf.epoch_max):
                clf.optimizer.zero_grad()
                outputs = clf.forward(x_train)
                loss = clf.criterion(outputs, y_train)
                loss.backward()
                # 梯度裁剪
                nn.utils.clip_grad_norm_(clf.optimizer_params(), 1, norm_type=2)
                clf.optimizer.step()
                clf.optimizer.zero_grad()
                # 使用验证集评估模型 在最后一轮强制进行比较
                if epoch % clf.test_check_epoch == 0 and epoch != 0 or epoch == clf.epoch_max - 1:
                    with torch.no_grad():
                        outputs = clf.forward(x_test)
                        _, predictions = torch.max(outputs, 1) # 获得outputs每行最大值的索引
                        # 准确率
                        test_accuracy = ((predictions == y_test).float().mean()).item() # (predictions == labels):输出值为bool的Tensor
                        # r2
                        test_r2 = r2_score(predictions, y_test)
                        # 如果此时r2 大于最优r2，则记录此r2
                        if(test_r2 > best_r2):
                            best_r2 = test_r2.item()
                            best_epoch = epoch
                        if(test_accuracy > best_accuracy):
                            best_accuracy = test_accuracy
                            best_epoch = epoch
                            # 记录模型 
                        # print("Round : %d, best_accu: %.2f %%, best_R2: %0.2f, best_epoch: %d" % (epoch+1, best_accuracy * 100, best_r2, best_epoch))
                        # print('Fold :', fold_cnt,'/', self.cv)        # 打印折数
            avg_test_accuracy.append(best_accuracy)
            avg_test_r2.append(best_r2)
            # 模型参数归零，防止数据信息泄漏
            clf.reset_parameters()
            fold_cnt += 1
        # 模型重新送回cpu，防止继续占用GPU资源
        clf = clf.to(self.host_device)
        # 返回r2
        return np.mean(avg_test_r2).item()

    def _build_clf(self, param: np.ndarray) -> nn.Module:
        '''创建用于超参数搜索的模型'''
        param = convert_to_numbers(param)        
        def build_fit_params(params_code:list, clf:pytorch_model_sklearn)->list:
            params_list = []
            if(params_code[0] == 0):
                params_list.append(nn.CrossEntropyLoss())
            if(params_code[1] == 0):
                params_list.append(torch.optim.Adam(clf.optimizer_params(),lr=0.005, weight_decay=0))
            elif(params_code[1] == 1):
                params_list.append(torch.optim.SGD(clf.optimizer_params(), lr=0.01, momentum=0.9))
            else:
                raise ValueError("Invalid params")
            return params_list
        # 1-10 层数
        # print(param[:11])
        input_params = param[:11]
        fit_params = param[11:]
        if(isinstance(input_params, np.ndarray)):
            input_params = input_params.tolist()
            fit_params = fit_params.tolist()
        clf = pytorch_model_sklearn(input_params)
        # 11-12 参数
        input_params = build_fit_params(fit_params, clf)
        clf.set_fitting_params(criterion=input_params[0], optimizer=input_params[1], using_gpu=self.using_gpu)
        if(self.using_gpu):
            clf.to(clf.calc_device)
        return clf
        
    def _param_filter(self, param: np.ndarray) -> np.ndarray:
        return param
    
    def fit(self, x_data: np.ndarray, y_data: np.ndarray, param_list: np.ndarray) -> None:
        # 进行数据格式统一，交叉验证数据拆分，传入GPU设备
        self.k_fold_data = utils.build_k_fold_data(self.cv,torch.from_numpy(x_data).float(), torch.from_numpy(y_data).float())
        for i in range(len(self.k_fold_data)):
            for j in range(len(self.k_fold_data[i])):
                if j % 2 == 1:
                    self.k_fold_data[i][j] = self.k_fold_data[i][j].long().squeeze(-1)
                self.k_fold_data[i][j] = self.k_fold_data[i][j].to(self.calc_device)
        super().fit(x_data, y_data, param_list)
        # 数据传回CPU
        for i in range(len(self.k_fold_data)):
            for j in range(len(self.k_fold_data[i])):
                self.k_fold_data[i][j] = self.k_fold_data[i][j].to(self.host_device)
    
def main():
    data_src_dir = sys.path[0]+'/../data/cell_input_data_new_1'
    data_out_dir = sys.path[0]+'/../data/cell_input_data_new_1/out'
    x_data, y_data = utils.load_data_from_path(2, data_out_dir, data_out_dir, data_src_dir)
    data = np.concatenate((x_data, y_data), axis=1)
    np.random.shuffle(data)
    x_data = data[:,:-8]
    y_data = data[:,-8:]
    Voc_data = y_data[:,0]
    x_data, y_data = load_iris_shuffle()
    mlp_torch_param[0] = [x_data.shape[1]]
    mlp_torch_param[6] = [len(set(y_data))]
    params = grid_param_builder(mlp_torch_param)
    clf = self_mlp_torch()
    clf.fit(x_data, y_data, params)
    clf.save_residual_param(sys.path[0]+'/tmp.txt')
    # 优化方法

def demo():
    '''示例方法2'''
    param_list = [
        [1],        # input_dim         
        [8192],
        [4096],
        [1024],
        [1],        # output_dim 4
        [512],
        [512],
        [512],
        [512],
        [512], 
        [1],        # output_dim 10
        [0],        # criterion 0 : CrossEntropyLoss
        [0,1]       # optimizer 0 : Adam 1.SGD(默认使用Adam)
    ]
    x_data, y_data = load_iris_shuffle()
    param_list[0] = [x_data.shape[1]]
    param_list[4] = [len(set(y_data))]
    params = grid_param_builder(param_list)
    clf = self_mlp_torch()
    clf.fit(x_data, y_data, params)
    # 优化方法

if __name__ == "__main__":
    demo()
    # main()
