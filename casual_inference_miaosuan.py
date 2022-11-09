import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import math
from catboost import CatBoostRegressor
from sklearn import ensemble
from sklearn.preprocessing import RobustScaler, StandardScaler
from xgboost.sklearn import XGBClassifier, XGBRegressor
from causalml.metrics.visualize import plot_lift, plot_qini, plot_gain, plot_tmlegain, auuc_score, qini_score, \
    get_tmlegain
from causalml.propensity import ElasticNetPropensityModel
import pickle
import argparse
import logging

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)


def data_process(train_data, inference_data, feature_columns, treatment_columns_category, outcome_column):
    # 众数 补充缺失值
    train = pd.concat([train_data, inference_data])
    train = train[feature_columns]
    for col in feature_columns:
        d = dict(train[col].value_counts())
        train[col] = train[col].fillna(sorted(d.items(), key=lambda x: x[1], reverse=True)[0][0])

        train_data[col] = train_data[col].fillna(sorted(d.items(), key=lambda x: x[1], reverse=True)[0][0])
        inference_data[col] = inference_data[col].fillna(sorted(d.items(), key=lambda x: x[1], reverse=True)[0][0])

    treatment_columns_category_dict = dict()
    for col in treatment_columns_category:
        label_encoder = LabelEncoder().fit(train[col])
        temp_dict = dict(
            zip(range(0, len(label_encoder.classes_)), label_encoder.classes_))
        treatment_columns_category_dict[col] = temp_dict
        train[col] = label_encoder.transform(train[col])

        train_data[col] = label_encoder.transform(train_data[col])
        inference_data[col] = label_encoder.transform(inference_data[col])

    for col in feature_columns:
        if train[col].dtype == 'object':
            label_encoder = LabelEncoder().fit(train[col])
            train[col] = label_encoder.transform(train[col])

            train_data[col] = label_encoder.transform(train_data[col])
            inference_data[col] = label_encoder.transform(inference_data[col])

    xgb = XGBRegressor()
    xgb.fit(train_data[feature_columns], train_data[outcome_column])

    inference_data["outcome1"] = xgb.predict(inference_data[feature_columns])

    return treatment_columns_category_dict, inference_data, xgb


def inference(train, treatment_columns_category_dict, xgb, feature_columns,
              treatment_columns_category, treatment_columns_common, treatment_change_value,
              treatment_columns_continuous,
              userid_column):
    outcome_pd = pd.DataFrame()
    outcome_pd['outcome'] = xgb.predict(train[feature_columns])
    result = dict()

    # 基于连续型变量的处理
    treatment_cont_common = []
    for col, limit_value in treatment_columns_continuous.items():
        if col in treatment_columns_common:
            treatment_cont_common.append(col)
        else:
            train_temp = train.copy(deep=True)
            if limit_value[2] == 0:
                if limit_value[0] != None:
                    train_temp[col] = train_temp[col].apply(
                        lambda x: x * (1 - treatment_change_value) if x * (1 - treatment_change_value) > limit_value[
                            0] else x)
                if limit_value[1] != None:
                    train_temp[col] = train_temp[col].apply(
                        lambda x: x * (1 - treatment_change_value) if x * (1 - treatment_change_value) < limit_value[
                            1] else x)
            if limit_value[2] == 1:
                if limit_value[0] != None:
                    train_temp[col] = train_temp[col].apply(
                        lambda x: x * (1 + treatment_change_value) if x * (1 + treatment_change_value) > limit_value[
                            0] else x)
                if limit_value[1] != None:
                    train_temp[col] = train_temp[col].apply(
                        lambda x: x * (1 + treatment_change_value) if x * (1 + treatment_change_value) < limit_value[
                            1] else x)
            outcome_pd[col] = xgb.predict(train_temp[feature_columns])
    result["treatment_cont_common"] = treatment_cont_common

    # 基于类别型变量的处理
    treatment_cate_common = dict()
    for col in treatment_columns_category:
        train_temp = train.copy(deep=True)
        if col in treatment_columns_common:
            pd_category = pd.DataFrame()
            logging.info("{} is category 变量，在common中 ".format(col))
            treat_enum = treatment_columns_category_dict[col]
            logging.info(treat_enum)
            for enum_value in treat_enum.keys():
                logging.info(enum_value)
                train_temp[col] = enum_value
                pd_category[enum_value] = xgb.predict(
                    train_temp[feature_columns])
            d = dict(pd_category.mean())
            d = sorted(d.items(), key=lambda x: x[1], reverse=True)[0][0]
            treatment_cate_common[col] = treat_enum[d]
            logging.info("\n\n")
        else:
            pd_category = pd.DataFrame()
            logging.info("{} is category 变量，不在common中 ".format(col))
            treat_enum = treatment_columns_category_dict[col]
            logging.info(treat_enum)
            for enum_value in treat_enum.keys():
                logging.info(enum_value)
                train_temp[col] = enum_value
                pd_category[enum_value] = xgb.predict(
                    train_temp[feature_columns])
            logging.info("\n\n")
            cate_max = pd_category.max(axis=1)
            cate_idxmax = pd_category.idxmax(axis=1)
            pd_category[col] = cate_max
            pd_category[col + "_idx"] = cate_idxmax

            outcome_pd[col] = cate_max
            outcome_pd[col + "_idx"] = cate_idxmax

    result["treatment_cate_common"] = treatment_cate_common

    temp_pd = outcome_pd[[i for i in treatment_columns_category + list(
        treatment_columns_continuous.keys()) if i not in treatment_columns_common]]
    outcome_pd['max_idx'] = temp_pd.idxmax(axis=1)
    outcome_pd["user_id"] = train[userid_column].values

    user_res = dict()
    outcome_pd_cols = outcome_pd.columns.to_list()
    for key, value in outcome_pd.iterrows():
        max_idx = value["max_idx"]
        if max_idx + "_idx" in outcome_pd_cols:
            temp = dict()
            temp[max_idx] = value[max_idx + "_idx"]
            user_res[value["user_id"]] = temp
        else:
            user_res[value["user_id"]] = max_idx
    #     logging.info(max_idx)
    result["user_treatment"] = user_res

    # 保存文件
    with open("causal_inference.pkl", "wb") as tf:
        pickle.dump(result, tf)

    return result, outcome_pd


def get_auuc(pred_df, y_col, treatment_col):
    auuc_df = pd.DataFrame(auuc_score(pred_df, outcome_col=y_col,
                                      treatment_col=treatment_col, normalize=True)).reset_index()
    auuc_df.columns = ['Learner', 'auuc']
    auuc_df['Lift'] = (auuc_df['auuc'] /
                       auuc_df[auuc_df.Learner == 'Random'].auuc.values) - 1
    auuc_df = auuc_df.sort_values('auuc', ascending=False)
    logging.info(auuc_df)
    return auuc_df


# mock data
# train = pd.read_csv('train.csv')
# train = train[['V_0', 'V_1', 'V_2', 'V_3', 'V_4', 'V_5', 'V_6', 'V_7', 'V_8', 'V_9',
#                'V_10', 'treatment', 'outcome']]
# train['user_id'] = train.index
#
# feature_columns = ['V_0', 'V_1', 'V_2', 'V_3', 'V_4', 'V_5', 'V_6', 'V_7', 'V_8', 'V_9',
#                    'V_10', 'treatment']
# # treatment_columns_continuous=['V_6', 'V_7']
# treatment_columns_category = ['V_10', 'treatment']
# treatment_columns_common = ['V_6', 'treatment']
# treatment_change_value = 0.2
# outcome_column = ["outcome"]
# user_id_column = ["user_id"]
# treatment_columns_continuous = {
#     "V_6": [None, 10, 1],
#     "V_7": [0.3, None, 0]
# }
# train_data = train[0:30000]
# inference_data = train[30000:]
# treatment_columns_category_dict, inference_data, xgb = data_process(train_data, inference_data, feature_columns,
#                                                                     treatment_columns_category,
#                                                                     treatment_columns_common,
#                                                                     outcome_column, treatment_columns_continuous)
# m = inference(inference_data, treatment_columns_category_dict, xgb, feature_columns,
#               treatment_columns_category, treatment_columns_common, outcome_column, treatment_columns_continuous)
#


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str, default="train_data.csv", help="训练文件路径")
    parser.add_argument('--inference_data_path', type=str, default="inference_data.csv", help="推理文件路径")
    parser.add_argument('--feature_columns', type=str, default="", help="所有的特征list")
    parser.add_argument('--treatment_columns_category', type=str, default="", help="枚举值的treatment，list")
    parser.add_argument('--treatment_columns_continuous', type=str, default="", help="连续值的treatment，dict")
    parser.add_argument('--treatment_columns_common', type=str, default="", help="预先定义的treatment，list")
    parser.add_argument('--outcome_column', type=str, default="", help="标签列名，list")
    parser.add_argument('--userid_column', type=str, default="user_id", help="userid列名，list")
    args = parser.parse_args()

    try:
        train_data_path = args.train_data_path
        inference_data_path = args.inference_data_path
        feature_columns = eval(args.feature_columns)
        treatment_columns_category = eval(args.treatment_columns_category)
        treatment_columns_continuous = eval(args.treatment_columns_continuous)
        treatment_columns_common = eval(args.treatment_columns_common)
        outcome_column = eval(args.outcome_column)
        userid_column = eval(args.userid_column)
        treatment_change_value = 0.2

        train_data = pd.read_csv(train_data_path)
        inference_data = pd.read_csv(inference_data_path)
        treatment_columns_category_dict, inference_data, xgb = data_process(train_data, inference_data, feature_columns,
                                                                            treatment_columns_category,
                                                                            outcome_column)
        result, outcome_dp = inference(inference_data, treatment_columns_category_dict, xgb, feature_columns,
                                       treatment_columns_category, treatment_columns_common, treatment_change_value,
                                       treatment_columns_continuous,
                                       userid_column)
        logging.info("\n\n**********")
        logging.info("final result file is : causal_inference.pkl")

    except:
        logging.warning("\n\n****something wrong****")

## example
#  python casual_inference_miaosuan.py --feature_columns "['V_0', 'V_1', 'V_2', 'V_3', 'V_4', 'V_5', 'V_6', 'V_7', 'V_8', 'V_9','V_10', 'treatment']" --treatment_columns_category "['V_10', 'treatment']" --treatment_columns_continuous "{'V_6': [None, 10, 1],'V_7': [0.3, None, 0]}" --treatment_columns_common "['V_6','treatment']" --outcome_column "['outcome']" --userid_column "['user_id']"
