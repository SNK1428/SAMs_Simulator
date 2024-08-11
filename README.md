**Not yet finished**

# 项目说明

## 项目结构

```
Project_root
  |-Data_Generator                  // 编码软件 预测数据生成软件
  |   |-config.txt                  // 配置文件(可通过执行执行文件时指定位置)
  |-Fitting_model                   // 存储的网格+交叉验证搜索的模型
  |-Raw_Mole_Encode_Generator       // 将分子结构（SMILES，CML，MOL）文件转为分子编码
  |-Data                            // 原始数据存储位置
  |-README.md                       // 本文件
```

## 操作系统及配置

Linux(Recommend: Debian12), 16G RAM + 256G swap, CUDA11.8

## 语言版本与主要包依赖

### C++17

CMake 3.25
Clang 14.0.6
GCC 11.3

### Python3.9

PyTorch 2.2.2
torcheval 0.0.7
NumPy 1.26.4
Scikit-learn 1.4.1
Pandas 2.2.1
Matplotlib 3.7.3
shap 0.42.1

## 使用方法

1. 数据预处理+独热编码
  Data_Generator/build/Onehot_Generator（编码后）

1. 特征过滤，模型筛选，集成模型训练
  Fitting_model/essemble_model.py

1. 数据生成与预测
  数据生成
  Data_Generator/build/Predictor 
  数据预测
  Fitting_model/essemble_model.py
