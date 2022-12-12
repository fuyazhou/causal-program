import pandas as pd
import numpy as np
import traceback, sys
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from xgboost.sklearn import XGBClassifier, XGBRegressor
from causalml.metrics.visualize import plot_lift, plot_qini, plot_gain, plot_tmlegain, auuc_score, qini_score, \
    get_tmlegain
import argparse
import logging
import json

logging.basicConfig(filename='casual_inference_miaosuan.log.txt',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

logging.info("Running Urban Planning")
logger = logging.getLogger('urbanGUI')


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

    return treatment_columns_category_dict, inference_data, xgb, train_data


def inference(train, treatment_columns_category_dict, xgb, feature_columns,
              treatment_columns_category, treatment_columns_common, treatment_change_value,
              treatment_columns_continuous,
              userid_column, target_data_path):
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
            _a = limit_value[0] if limit_value[0] is not None else -np.inf
            _b = limit_value[1] if limit_value[1] is not None else np.inf
            if limit_value[2] == 0:
                train_temp[col] = train_temp[col].apply(
                    lambda x: x * (1 - treatment_change_value) if _a < x * (1 - treatment_change_value) < _b else x)
            if limit_value[2] == 1:
                train_temp[col] = train_temp[col].apply(
                    lambda x: x * (1 + treatment_change_value) if _a < x * (1 + treatment_change_value) < _b else x)

            outcome_pd[col] = xgb.predict(train_temp[feature_columns])
    result["treatment_cont_common"] = treatment_cont_common


    treatment_cate_common = []
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
                pd_category[enum_value] = xgb.predict(train_temp[feature_columns])
            d = dict(pd_category.mean())
            d = sorted(d.items(), key=lambda x: x[1], reverse=True)[0][0]
            _temp_dict = dict()
            _temp_dict["treatment"] = col
            _temp_dict["value"] = treat_enum[d]
            treatment_cate_common.append(_temp_dict)
            logging.info("\n\n")

        else:
            pd_category = pd.DataFrame()
            logging.info("{} is category 变量，不在common中 ".format(col))
            treat_enum = treatment_columns_category_dict[col]
            logging.info(treat_enum)
            for enum_value in treat_enum.keys():
                logging.info(enum_value)
                train_temp[col] = enum_value
                pd_category[enum_value] = xgb.predict(train_temp[feature_columns])
            logging.info("\n\n")
            cate_max = pd_category.max(axis=1)
            cate_idxmax = pd_category.idxmax(axis=1)
            pd_category[col] = cate_max
            pd_category[col + "_idx"] = cate_idxmax

            outcome_pd[col] = cate_max
            outcome_pd[col + "_idx"] = cate_idxmax

    result["treatment_cate_common"] = treatment_cate_common

    treat_cols = [i for i in treatment_columns_category + list(
        treatment_columns_continuous.keys()) if i not in treatment_columns_common]

    if len(treat_cols) == 0:
        logging.info("None feature need treatment!!!!")
        return None, None

    logging.info("feature need treatment ： {}".format(str(treat_cols)))

    temp_pd = outcome_pd[treat_cols]
    outcome_pd['max_idx'] = temp_pd.idxmax(axis=1)
    outcome_pd["user_id"] = train[userid_column].values

    user_res = []
    outcome_pd_cols = outcome_pd.columns.to_list()
    for key, value in outcome_pd.iterrows():
        _temp_dict = dict()
        max_idx = value["max_idx"]
        if max_idx + "_idx" in outcome_pd_cols:
            _temp_dict["user_id"] = value["user_id"]
            _temp_dict["treatment"] = max_idx
            _temp_dict["value"] = value[max_idx + "_idx"]

        else:
            _temp_dict["user_id"] = value["user_id"]
            _temp_dict["treatment"] = max_idx
        user_res.append(_temp_dict)
    result["user_treatment"] = user_res

    with open(target_data_path, "w", encoding="utf8") as tf:
        json.dump(result, tf, ensure_ascii=False, indent=2, cls=NpEncoder)

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


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def auuc_output(train_data, feature_columns, outcome_column, treatment_columns_category_dict):
    try:
        auuc_pd = pd.DataFrame()
        auuc_pd = train_data[outcome_column]
        for i in treatment_columns_category_dict.keys():
            auuc_pd[i] = train_data[i]
            i_keys = list(treatment_columns_category_dict[i].keys())
            i_values = list(treatment_columns_category_dict[i].values())
            logging.info(str(i_values) + " " + str(i_keys))
            for j in i_keys:
                temp_data = train_data.copy(deep=True)
                temp_data[i] = j
                auuc_pd[str(i) + "_" + str(j)
                        ] = xgb.predict(temp_data[feature_columns])
                if j != 0:
                    auuc_pd[str(i_values[j]) + "--" + str(i_values[0])
                            ] = auuc_pd[str(i) + "_" + str(j)] - auuc_pd[str(i) + "_" + str(0)]
                    lalla = auuc_pd[outcome_column + [i] +
                                    [str(i_values[j]) + "--" + str(i_values[0])]]

                    logging.info(str(i) + '  ' + str(j) + '  ' + str(i_values[j]))
                    lalla = lalla[lalla[i].isin([0, j])]
                    lalla[i].replace(j, 1, inplace=True)
                    logging.info(get_auuc(lalla, outcome_column[0], i))
                    logging.info("\n\n")
    except:
        logging.info("\n\n****something wrong about auuc_output****")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--parameters_data_path', type=str, default="parameters.json", help="参数文件路径")
    args = parser.parse_args()

    try:
        logging.info("********start casual inference ************")
        parameters_path = args.parameters_data_path
        logging.info("input path is : " + parameters_path)
        with open(parameters_path, 'r', encoding='utf8') as fp:
            parameters = json.load(fp)
        logging.info("input parameters data  is : " + str(parameters))

        target_data_path = parameters["target_data_path"]
        train_data_path = parameters["train_data_path"]
        inference_data_path = parameters["inference_data_path"]
        feature_columns = parameters["feature_columns"]
        treatment_columns_category = parameters["treatment_columns_category"]
        treatment_columns_continuous_temp = parameters["treatment_columns_continuous"]
        treatment_columns_common = parameters["treatment_columns_common"]
        outcome_column = parameters["outcome_column"]
        userid_column = parameters["userid_column"]

        treatment_columns_continuous = {}

        for continuous_temp in treatment_columns_continuous_temp:
            temp_list = [continuous_temp["min_value"], continuous_temp["max_value"],
                         continuous_temp["neg_correlation_boolean"]]
            treatment_columns_continuous[continuous_temp["field_name"]] = temp_list

        treatment_change_value = 0.2

        if train_data_path.endswith("csv"):
            train_data = pd.read_csv(train_data_path, error_bad_lines=False)
            inference_data = pd.read_csv(inference_data_path, error_bad_lines=False)
        else:
            train_data = pd.read_excel(train_data_path)
            inference_data = pd.read_excel(inference_data_path)
        treatment_columns_category_dict, inference_data, xgb, train_data = data_process(train_data, inference_data,
                                                                                        feature_columns,
                                                                                        treatment_columns_category,
                                                                                        outcome_column)

        auuc_output(train_data, feature_columns, outcome_column, treatment_columns_category_dict)

        result, outcome_dp = inference(inference_data, treatment_columns_category_dict, xgb, feature_columns,
                                       treatment_columns_category, treatment_columns_common, treatment_change_value,
                                       treatment_columns_continuous,
                                       userid_column, target_data_path)
        logging.info("\n\n")
        logging.info(f"final result file is : {target_data_path}")
        logging.info("**********\n\n\n\n")


    except Exception as e:
        logging.error("error detail is :{}, {} ".format(e.__class__.__name__, e))
        # exc_type, exc_value, exc_traceback = sys.exc_info()
        logging.error(str(traceback.print_exc()))
        logging.error("****something wrong****\n\n\n")


