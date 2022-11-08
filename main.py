#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import math
from catboost import CatBoostRegressor
from sklearn import ensemble
import ylearn
from ylearn.causal_discovery import CausalDiscovery
from xgboost.sklearn import XGBClassifier, XGBRegressor

from arguments import *

np.random.seed(2022)


# 数据的处理
def data_process(train_file, test_file):
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    final_train_cols = [x for x in train.columns if x not in set(['treatment', 'outcome'])]
    final_cols = final_train_cols

    # object feature encode
    for col in train.columns:
        if train[col].dtype == 'object':
            label_encoder = LabelEncoder().fit(train[col])
            train[col] = label_encoder.transform(train[col])
            test[col] = label_encoder.transform(test[col])
    train_na_dic = {}
    test_na_dic = {}
    for col in train.columns:
        train_na_dic[col] = train[col].isna().sum() / train.shape[0]
    for col in test.columns:
        test_na_dic[col] = test[col].isna().sum() / test.shape[0]

    na_fill_need_col = [x[0] for x in train_na_dic.items() if x[1] > 0]

    # 缺失值的填充
    training_col = [x for x in train.columns if x not in ['treatment', 'outcome']]
    train_cols = ['V_8', 'V_37', 'V_25', 'V_7', 'V_6', 'V_30', 'V_14', 'V_31', 'V_12', 'V_39', 'V_32', 'V_15', 'V_16',
                  'V_2', 'V_24']
    for col in na_fill_need_col:
        train_cols = [x for x in training_col if x != col and 'predict' not in x]
        train_na_df = train[~train[col].isna()]
        xgb = XGBRegressor(learning_rate=0.3, max_depth=5, min_child_weight=5,
                           colsample_bytree=0.7,
                           subsample=0.91,
                           n_estimators=100)
        cab = CatBoostRegressor()
        xgb.fit(train_na_df[train_cols], train_na_df[col])
        cab.fit(train_na_df[train_cols], train_na_df[col])
        train[col + '_predict'] = (xgb.predict(train[train_cols]) + cab.predict(train[train_cols])) / 2
        test[col + '_predict'] = (xgb.predict(test[train_cols]) + cab.predict(test[train_cols])) / 2

    def fillna(x1, x2):
        if math.isnan(x1):
            return x2
        else:
            return x1

    for col in na_fill_need_col:
        train[col] = train.apply(lambda x: fillna(x[col], x[col + '_predict']), axis=1)
        test[col] = test.apply(lambda x: fillna(x[col], x[col + '_predict']), axis=1)

    return train, test


def train_predict(train, test, final_cols):
    # 特征的确定
    scaler_cols = ['V_9', 'V_28', 'V_31', 'V_36']
    t = StandardScaler().fit(train[scaler_cols])
    training = t.transform(train[scaler_cols])
    testing = t.transform(test[scaler_cols])
    train[scaler_cols] = training
    test[scaler_cols] = testing

    xgb = XGBRegressor(learning_rate=0.3, max_depth=5, min_child_weight=5,
                       colsample_bytree=0.7,
                       subsample=0.91,
                       n_estimators=100)

    train_treat_ori = train['treatment']
    outcome_train = train['outcome']
    train = train[final_cols + ['treatment']]

    cab = CatBoostRegressor()
    xgb.fit(train, outcome_train)
    cab.fit(train, outcome_train)

    train['treatment'] = 1
    train_treatment1_xgb = xgb.predict(train)
    train_treatment1_cab = cab.predict(train)
    train_ensemble_treat1 = (train_treatment1_xgb + train_treatment1_cab) / 2

    train['treatment'] = 2
    train_treatment2_xgb = xgb.predict(train)
    train_treatment2_cab = cab.predict(train)
    train_ensemble_treat2 = (train_treatment2_xgb + train_treatment2_cab) / 2

    train['treatment'] = 0
    train_treatment0_xgb = xgb.predict(train)
    train_treatment0_cab = cab.predict(train)
    train_ensemble_treat0 = (train_treatment1_xgb + train_treatment0_cab) / 2
    train['t1_pred'] = train_ensemble_treat1
    train['t2_pred'] = train_ensemble_treat2
    train['t0_pred'] = train_ensemble_treat0
    train['t_ori'] = train_treat_ori
    train['outcome'] = outcome_train

    test = test[final_cols]
    test['treatment'] = 1
    test_treatment1_xgb = xgb.predict(test)
    test_treatment1_cab = cab.predict(test)
    test_ensemble_treat1 = (test_treatment1_xgb + test_treatment1_cab) / 2

    test['treatment'] = 2
    test_treatment2_xgb = xgb.predict(test)
    test_treatment2_cab = cab.predict(test)
    test_ensemble_treat2 = (test_treatment2_xgb + test_treatment2_cab) / 2

    test['treatment'] = 0
    test_treatment0_xgb = xgb.predict(test)
    test_treatment0_cab = cab.predict(test)
    test_ensemble_treat0 = (test_treatment0_xgb + test_treatment0_cab) / 2

    test['t1_pred'] = test_ensemble_treat1
    test['t2_pred'] = test_ensemble_treat2
    test['t0_pred'] = test_ensemble_treat0

    # outcome重置到原来的值，表现更差，目前没有用到，“1”不起作用。
    def convert(pred, t_ori, outcome, flag):
        if (flag == 'treat1' and t_ori == '1') or (flag == 'treat2' and t_ori == '2') or (
                flag == 'treat0' and t_ori == '0'):
            return outcome
        else:
            return pred

    train['t1_final'] = train.apply(lambda x: convert(x['t1_pred'], x['t_ori'], x['outcome'], 'treat1'), axis=1)
    train['t2_final'] = train.apply(lambda x: convert(x['t2_pred'], x['t_ori'], x['outcome'], 'treat2'), axis=1)
    train['t0_final'] = train.apply(lambda x: convert(x['t0_pred'], x['t_ori'], x['outcome'], 'treat0'), axis=1)

    train['ce_1'] = train['t1_final'] - train['t0_final']
    train['ce_2'] = train['t2_final'] - train['t0_final']
    test['ce_1'] = test['t1_pred'] - test['t0_pred']
    test['ce_2'] = test['t2_pred'] - test['t0_pred']
    train[['ce_1', 'ce_2']].append(test[['ce_1', 'ce_2']]).to_csv('result.csv', index=False)
    return "every thing is done"


train, test = data_process(arg_dic["train_file"], arg_dic["test_file"])
final_cols = ['V_2', 'V_9', 'V_15', 'V_28', 'V_8', 'V_31', 'V_36', 'V_12']
train_predict(train, test, final_cols)
