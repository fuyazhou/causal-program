### 一、特征工程

代码：  feature_selection.py

##### 1、缺失值的处理

对是缺失值进行预测填充。使用其他特征作为feature，缺失值的特征作为target，使用xgboost进行训练预测，对缺失值进行填充。



##### 2、特征选择

使用 lingam 工具，对特征进行选择。

首先分别计算出所有feature 对 treatment和outcome 的影响，筛选特征的规则：

（1）筛选出对outcome 影响大，但是对treatment影响小的特征。

（2）筛选出对treatment影响大，但是对outcome 影响小的特征。

（2）筛选出的特征对treatment和outcome 具有正向的影响。



### 二、模型训练预测

代码：  main.py

##### 1、模型的选择

模型选择的是xgboost和CatBoost进行集成学习。使用single learner的方式，使用筛选出的特征+treatment 作为最后的特征，  outcome作为label。



##### 2、模型的优化

使用 hyperopt 对模型进行调参，使用最优的参数进行训练和预测。



##### 3、模型的评估

综合使用auuc 、qini 进行评估。



