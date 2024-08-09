# 弱分类器 DecisionTreeClassifier 和 MLP
ada_param = [
    [0],                      # 弱分类器种类：决策树/mlp(不支持sampleWeight，后期再写)
    [50],                       # 估计器数量
    # ada参数
    ['SAMME'],                  # algorithm SAMME.R为回归模型参数，未来将被舍弃
    [0.1,0.3,0.5,1,1.2,1.5],    # leanring rate
    # DecisionTreeClassifier超参数
    ['gini'],                       # criterion (4)
    ['random', 'best'],             # splitter
    [2, 5, 10, 15],                   # min_samples_split
    [1, 5, 10],                      #
    ['balanced', 'None'],           # class_weight
    #mlp 超参数（参考MLP一章）
    [0, 50, 100],
    [0, 50, 100],                         # layers
    ['identity', 'logistic', 'tanh','relu'],
    ['lbfgs','sgd','adam'],
    [0,0.0001,0.0002,0.0004,0.0008],    # alpha（先用0）
    ['adaptive'],                       # learning rate
    [0.7, 0.9, 0.95, 0.97, 0.98, 0.99],            # monument（先用默认0.9）
]

# Gaussian 无超参数
# Multi 无超参数
# Cata 有一个bool型
# comple 有两个
BO_param = [
        [0,1,2,3],                  # BO type
        # Comple
        [0.001,0.01,0.05, 0.1,0.3, 0.5, 0.7, 1],       # alpha
        ['True', 'False'],        # norm
        ]

# KNN网格参数采样范围
'''https://blog.csdn.net/weixin_42182448/article/details/88639391'''
knn_param = [
    [3,4,5,6,7],                                            # n_neighbors 近邻邻居
    ['uniform','distance'],                                 # weights 距离权重
    ['ball_tree', 'kd_tree'],                               # algorithm 距离权重
    [30,50,70,100],                                         # leaf_size kd/ball树才有
    ['euclidean', 'minkowski','manhattan','chebyshev'],     # metric https://blog.csdn.net/weixin_44607126/article/details/102598096
    [1,2]                                                   # p(用于minkowski) 1 曼哈顿距离 2 欧
]

# 构建网格筛选器
lr_param = [
    # 公共参数
    [0,1,2],    # 0 elastic 1 lasso 2 ridge
    [0.01,0.05,0.1,0.3,0.5,1,1.5,2,5,10],       # alpha
    # elastic 和 lasso
    ['cyclic', 'random'],                       # selection
    # elastic
    [0.2,0.3,0.4,0.5,0.6,0.7,0.8],              # l1_ratio(3)
    # ridge
    ['balanced', 'None'],                       # class_weight(4)
    ['auto']                                    # solver
]

# 作为弱分类器的参数
mlp_sk_weak_list = [
    [0,50,100],
    [0,50,100],                         # layers
    ['identity', 'logistic', 'tanh','relu'],
    ['lbfgs','sgd','adam'],
    [0,0.0001,0.0002,0.0004,0.0008],    # alpha（先用0）
    ['adaptive'],                       # learning rate
    [0.7, 0.9, 0.95, 0.97, 0.98, 0.99],            # monument（先用默认0.9）
]

# 作为普通（强）分类器的参数 最高六层
mlp_param_list = [
    [0,100,300,500],
    [0,100,300,500],
    [0,100,300,500],
    [0,100,300,500],
    [0,100,300,500],     
    [0,100,300,500],# hidden_layer_size（最大六层）
    ['identity', 'logistic', 'tanh','relu'],         #activation
    ['libfgs', 'sgd', 'adam'],  # solver
    [0,0.0001,0.001,0.01,0.1],  # l2 punished
    ['auto'], # batch_size(使用默认)
    ['adaptive'], # learn_rate（）
    [0.001], # learn_rate_init 使用默认值
    [0.0001,0.0005,0.001], # power_t
    [0.7, 0.8, 0.9, 0.95], #mometum
    [], # nestervous_momentum
    [0.1], # validation_fraction
    [],  #beta 1
    [], #beta 2
]

# https://blog.csdn.net/VariableX/article/details/107190275
# https://blog.csdn.net/yingfengfeixiang/article/details/79369059
# 默认尺寸：50
rf_param = [
    ['gini', 'entropy', 'log_loss'],
    [2,6,12,24],                    # max_depth
    [1,2,5,10],                     # min_samples_leaf 
    [2,5,10,15,100],              # min_samples_split
    ['log2','sqrt'],                # max_features
    ['None', 'balanced']
]

# 仅使用linearSVC
# https://zhuanlan.zhihu.com/p/341554415
# https://blog.csdn.net/weixin_41990278/article/details/93165330
# https://blog.csdn.net/qq_41264055/article/details/133121451
# default '0'
svm_param = [
    # Public
    ['linear', 'poly', 'rbf', 'sigmoid'],
    #------------------------------
    # LinearSVC(linear)
    ['l1','l2'],                                    # penalty(1)
    [0],                                            # C（见7）
    ['hinge', 'squared_hinge'], # loss
    [True, False],                          # dual（优先 false）
    ['ovr', 'crammer_singer'],                      # multi_class 
    ['None'],                                       # class_weight
    #-------------------------------
    # SVC
    [0.1, 0.5, 1,2,3,4,5,7,9,10,12,15,20,30],       # C(7)
    [2,5,10],                                       # degree(for poly)
    ['scale', 'auto'],                              # gamma
    [0,1,2,3,4, 5,7,9,11,15, 20],                   # coef0 https://zhuanlan.zhihu.com/p/672562238
    [True, False],                                  # shrinking
    [True, False],                                  # probability
    ['balanced', 'None'],                           # class_weight（public）
    ['ovo', 'ovr'],                                 # decision function shape
    [True, False]                                   # break_ties
]

# XGBoost 在 SKlearn中的接口
xgb_sk_params = [
    # 模型/弱分类器选择：XGB XGBRF 
    [0, 1],                                                     # 0 : XGBClassifier 1 : XGBRFClassifier
    ['gbtree', 'gblinear','dart'],                              # booster
    #-----------------------
    [10],                                                       # n_estimator
    # public
    [0, 0.5, 1],                                                # reg_alpha (3)
    [0, 0.5, 1],                                                # reg_lambda
    ['multi:softmax', 'binary:logistic','reg:squarederror'],                       # objective 损失函数
    [0.3, 0.5, 0.7],                                                            # base_score
    [0],                                                        # max_delta_step    0为无约束 https://www.cnblogs.com/TimVerion/p/11436001.html#:~:text=max_delta_step%EF%BC%9A%E9%BB%98%E8%AE%A40%2C%E6%88%91%E4%BB%AC%E5%B8%B8%E7%94%A80.,%E8%BF%99%E4%B8%AA%E5%8F%82%E6%95%B0%E9%99%90%E5%88%B6%E4%BA%86%E6%AF%8F%E6%A3%B5%E6%A0%91%E6%9D%83%E9%87%8D%E6%94%B9%E5%8F%98%E7%9A%84%E6%9C%80%E5%A4%A7%E6%AD%A5%E9%95%BF%EF%BC%8C%E5%A6%82%E6%9E%9C%E8%BF%99%E4%B8%AA%E5%8F%82%E6%95%B0%E7%9A%84%E5%80%BC%E4%B8%BA0%2C%E5%88%99%E6%84%8F%E5%91%B3%E7%9D%80%E6%B2%A1%E6%9C%89%E7%BA%A6%E6%9D%9F%E3%80%82
    # ['gain','weight','cover','total_gain','total_cover'],     # importance_type
    # ['logloss', 'mlogloss'],                                  # eval_metric
    # gbtree dart
    [3, 6, 12],                                                 # max_depth (8)
    [0.01, 0.1,0.3],                                            # learning_rate
    [0, 1],                                                     # gamma
    [1, 5, 10],                                                 # min_child_weight
    [1],                                                        # subsample 采样比例
    [1],                                                        # colsample_bytree colsample_bylevel colsample_bynode # 使用1
    ['auto'],                                                   # tree_method
    # dart
    ['tree', 'forest'],                                         # normalize_type(15)
    # gblinear
    ['shotgun'],                                                # updater (16) 下降算法 （默认值）    https://zhuanlan.zhihu.com/p/687910054
    ['cyclic','random','greedy'],                               # feature_selector
    # nthread默认为-1（使用所有线程）
]

# 单独手动调参(基于pytorch的强分类MLP)
mlp_torch_param =[
        [4],        # input_dim         
        [50],
        [100],
        [50],
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

