#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import lingam
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import math
import ylearn
from ylearn.causal_discovery import CausalDiscovery
from lingam.utils import make_dot
from xgboost.sklearn import XGBClassifier, XGBRegressor

model = lingam.DirectLiNGAM()

train = pd.read_csv('train.csv')
for col in train.columns:
    if train[col].dtype == 'object':
        label_encoder = LabelEncoder().fit(train[col])
        train[col] = label_encoder.transform(train[col])

train_na_dic = {}
for col in train.columns:
    train_na_dic[col] = train[col].isna().sum() / train.shape[0]

na_fill_need_col = [x[0] for x in train_na_dic.items() if x[1] > 0]

training_col = [x for x in train.columns if x not in ['treatment', 'outcome']]

for col in na_fill_need_col:
    train_cols = [x for x in training_col if x != col and 'predict' not in x]
    train_na_df = train[~train[col].isna()]
    xgb = XGBRegressor()
    xgb.fit(train_na_df[train_cols], train_na_df[col])
    train[col + '_predict'] = xgb.predict(train[train_cols])
    # test[col + '_predict'] = xgb.predict(test[train_cols])


def fillna(x1, x2):
    if math.isnan(x1):
        return x2
    else:
        return x1


for col in na_fill_need_col:
    train[col] = train.apply(lambda x: fillna(x[col], x[col + '_predict']), axis=1)
    # test[col] = test.apply(lambda x:fillna(x[col],x[col+'_predict']),axis=1)

cols = ['V_0', 'V_1', 'V_2', 'V_3', 'V_4', 'V_5', 'V_6', 'V_7', 'V_8', 'V_9',
        'V_10', 'V_11', 'V_12', 'V_13', 'V_14', 'V_15', 'V_16', 'V_17', 'V_18',
        'V_19', 'V_20', 'V_21', 'V_22', 'V_23', 'V_24', 'V_25', 'V_26', 'V_27',
        'V_28', 'V_29', 'V_30', 'V_31', 'V_32', 'V_33', 'V_34', 'V_35', 'V_36',
        'V_37', 'V_38', 'V_39', 'treatment', 'outcome']

train[cols].to_csv("train_done.csv", index=False)

model = lingam.DirectLiNGAM()
model.fit(train[cols])

d = {}
for i in range(39):
    d[cols[i]] = model.estimate_total_effect(train[cols], i, 41)

get_ipython().run_line_magic('pinfo2', 'model.estimate_total_effect')

d2 = {}
for i in range(39):
    d2[cols[i]] = model.estimate_total_effect(train[cols], i, 40)

model.estimate_total_effect(train[cols], 9, 40)

train[cols]

cols = ['V_0', 'V_1', 'V_2', 'V_3', 'V_4', 'V_5', 'V_6', 'V_7', 'V_8', 'V_9',
        'V_10', 'V_11', 'V_12', 'V_13', 'V_14', 'V_15', 'V_16', 'V_17', 'V_18',
        'V_19', 'V_20', 'V_21', 'V_22', 'V_23', 'V_24', 'V_25', 'V_26', 'V_27',
        'V_28', 'V_29', 'V_30', 'V_31', 'V_32', 'V_33', 'V_34', 'V_35', 'V_36',
        'V_37', 'V_38', 'V_39', 'treatment', 'outcome']

edges = []
for i, index in enumerate(cols):
    adj_matrix = model.adjacency_matrix_[i]
    for j, val in enumerate(adj_matrix):
        if val > 0.01:
            edges.append((cols[i], cols[j], val))

edges

for key in sorted(d.items(), key=lambda x: x[1], reverse=True):
    print(key[0], d[key[0]], d2[key[0]])

sorted(d2.items(), key=lambda x: x[1], reverse=True)

for col in ['V_2', 'V_15', 'V_28', 'V_34', 'V_33', 'V_31']:  # 28
    print(d[col], d2[col])

# V_2 -> treatment [label=0.63]
# 	V_24 -> treatment [label=0.21]
# 	V_2 -> outcome [label=-0.59]
# 	V_3 -> outcome [label=0.69]
# 	V_8 -> outcome [label=0.56]
# 	V_10 -> outcome [label=-0.23]
# 	V_11 -> outcome [label=0.34]
# 	V_13 -> outcome [label=0.46]
# 	V_14 -> outcome [label=-0.39]
# 	V_15 -> outcome [label=0.25]
# 	V_16 -> outcome [label=-0.64]
# 	V_33 -> outcome [label=0.48]
# 	V_39 -> outcome [label=0.31] 


# In[ ]:


# V_2 -> treatment [label=0.63]
# 	V_16 -> treatment [label=0.15]
# 	V_24 -> treatment [label=0.21]
# 	V_33 -> treatment [label=0.12]
# 	V_35 -> treatment [label=0.11]


# 	V_0 -> outcome [label=-0.14]
# 	V_2 -> outcome [label=-0.59]
# 	V_3 -> outcome [label=0.69]
# 	V_8 -> outcome [label=0.56]
# 	V_10 -> outcome [label=-0.23]
# 	V_11 -> outcome [label=0.34]
# 	V_13 -> outcome [label=0.46]
# 	V_14 -> outcome [label=-0.39]
# 	V_15 -> outcome [label=0.25]
# 	V_16 -> outcome [label=-0.64]
# 	V_18 -> outcome [label=0.11]
# 	V_31 -> outcome [label=0.14]
# 	V_32 -> outcome [label=0.15]
# 	V_33 -> outcome [label=0.48]
# 	V_39 -> outcome [label=0.31]


# In[ ]:


# 	V_2 -> treatment [label=0.63]
# 	V_24 -> treatment [label=0.21]
# 	V_2 -> outcome [label=-0.59]
# 	V_3 -> outcome [label=0.69]
# 	V_8 -> outcome [label=0.56]
# 	V_10 -> outcome [label=-0.23]
# 	V_11 -> outcome [label=0.34]
# 	V_13 -> outcome [label=0.46]
# 	V_14 -> outcome [label=-0.39]
# 	V_15 -> outcome [label=0.25]
# 	V_16 -> outcome [label=-0.64]
# 	V_33 -> outcome [label=0.48]
# 	V_39 -> outcome [label=0.31]


dot = make_dot(model.adjacency_matrix_, labels=cols)
