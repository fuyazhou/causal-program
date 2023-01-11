# coding=utf8
import argparse
import json
import pandas as pd
import numpy as np
from ortools.sat.python import cp_model
import traceback, sys
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from xgboost.sklearn import XGBClassifier, XGBRegressor
from causalml.metrics.visualize import plot_lift, plot_qini, plot_gain, plot_tmlegain, auuc_score, qini_score, \
    get_tmlegain
import copy
from util import linear_dml
import logging

logging.basicConfig(filename='optimion_uplift.log',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%D -- %H:%M:%S',
                    level=logging.DEBUG)
logging.info("\n\n\n")
logging.info("Running Urban Planning")
logger = logging.getLogger('urbanGUI : ')


# mock data
def generate_data():
    a = np.random.rand(50, 3).astype('float16')
    data = pd.DataFrame(a, columns=["outcome0", "outcome1", "outcome2"])
    # 模拟红包的金额
    red_packet = [0, 2, 5]
    data["uplift0"] = 0
    data["uplift1"] = (data["outcome1"] - data["outcome0"]) / (red_packet[1] - red_packet[0])  # 使用斜率代替uplift
    data["uplift2"] = (data["outcome2"] - data["outcome1"]) / (red_packet[2] - red_packet[1])  # 使用斜率代替uplift

    # 理论上预测每个红包的概率
    outcome = data[["outcome0", "outcome1", "outcome2"]].values
    outcome = outcome.tolist()

    ## uplift
    costs = data[["uplift0", "uplift1", "uplift2"]].values
    costs = costs.tolist()
    return outcome, costs


def dml_inference(train_data, inference_data, feature_columns, treatment_columns_category, outcome_column,
                  userid_column):
    train_data.dropna(subset=[outcome_column[0]], inplace=True)
    # 众数 补充缺失值
    train = pd.concat([train_data, inference_data])
    train = train[feature_columns]
    for col in feature_columns:
        # if all is null， fill 0;
        if train[col].isnull().all():
            train[col].fillna(0)

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

    dml_estimators = dict()
    for _t in treatment_columns_category:
        temp_feature_columns = copy.deepcopy(feature_columns)
        temp_feature_columns.remove(_t)
        x = train_data[temp_feature_columns].values
        t = train_data[_t].values
        y = train_data[outcome_column[0]].values
        logger.info(
            "start train category dml models: x columns is :{}, treatment columns is :{},outcome column is :{}".format(
                str(temp_feature_columns), _t, str(outcome_column)))
        est = linear_dml(x=x, t=t, y=y, discrete=True, grid_search=False)
        dml_estimators[_t] = est

    for col in treatment_columns_category:
        train_temp = inference_data.copy(deep=True)
        temp_feature_columns = copy.deepcopy(feature_columns)
        temp_feature_columns.remove(col)
        est = dml_estimators[col]
        x = train_temp[temp_feature_columns].values

        pd_category = pd.DataFrame()
        pd_category[userid_column[0]] = inference_data[userid_column[0]].values

        logger.info("treatment is：{}".format(col))
        treat_enum = treatment_columns_category_dict[col]
        logger.info(treat_enum)
        for enum_value in treat_enum.keys():
            logger.info('treatment value is : {}'.format(enum_value))
            effect = est.effect(X=x, T0=0, T1=enum_value)
            pd_category[treat_enum[enum_value]] = effect
            logger.info("{} treatment之后, enum_value : {}， ate为：{} ".format(col, enum_value, np.mean(effect)))
        logger.info("\n\n")

    pd_uplift = pd_category
    # treatment的概率是加上  后续加上，目前默认是1
    # treatment_probability = pd_category[list(treat_enum.values())]
    # treatment_probability.iloc[:, :] = 1
    treatment_probability = (est.models_t[0][0].predict_proba(x) + est.models_t[0][1].predict_proba(x)) / 2
    coupon_list = list(treat_enum.values())
    logger.info("coupon list : ".format(coupon_list))
    return pd_uplift, coupon_list, treatment_probability


def optimization_define(outcome, costs, red_limit, red_packet, userid_column):
    # outcome : 发红包的概率 treatment_probability
    # costs : uplift值  pd_uplift
    # red_limit : 红包限制的总额
    # red_packet ：红包枚举值 coupon_list
    index2id = dict(zip(range(len(costs)), list(costs[userid_column[0]])))
    costs = costs[list(red_packet)].values
    num_workers = len(costs)
    num_tasks = len(costs[0])

    model = cp_model.CpModel()

    x = {}
    for worker in range(num_workers):
        for task in range(num_tasks):
            x[worker, task] = model.NewBoolVar(f'x[{worker},{task}]')

    # Each worker is assigned to at most one task.
    # 每个用户只能发一个红包
    for worker in range(num_workers):
        model.AddAtMostOne(x[worker, task] for task in range(num_tasks))

    all_red = []
    for worker in range(num_workers):
        for task in range(num_tasks):
            all_red.append(int(outcome[worker][task] * 100) * red_packet[task] * x[worker, task])  # *100是因为该求解器值支持整数
            # all_red.append(red_packet[task] * x[worker, task])  # *100是因为该求解器值支持整数
    model.Add(sum(all_red) <= int(red_limit * 100))  # 限制条件也乘以100
    # model.Add(sum(all_red) <= int(red_limit))  # 限制条件也乘以100  only support integer

    objective_terms = []
    for worker in range(num_workers):
        for task in range(num_tasks):
            objective_terms.append(costs[worker][task] * x[worker, task])
    model.Maximize(sum(objective_terms))

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    result = []
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        logger.info(f'Total cost = {solver.ObjectiveValue()}\n')
        for worker in range(num_workers):
            for task in range(num_tasks):
                if solver.BooleanValue(x[worker, task]):
                    # result[index2id[worker]] = red_packet[task]
                    result_temp = dict()
                    result_temp["user_id"] = index2id[worker]
                    result_temp["treatment"] = red_packet[task]
                    result.append(result_temp)
                    if worker < 20:
                        logger.info(f'Worker {index2id[worker]} assigned to task {task}.' +
                                    f' Cost = {costs[worker][task]}')
    else:
        logger.warning('No solution found.')
    logger.info("\n\n")
    logger.info("last result  ***** userid -- coupon")
    logger.info(str(result[0:5]))

    return result


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


# if __name__ == "__main__":
# red_limit = 100
# red_packet = [0, 6, 10]
# outcome, costs = generate_data()
# optimization_define(outcome, costs, red_limit, red_packet)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--parameters_data_path', type=str, default="uplift_parameters.json", help="参数文件路径")
    args = parser.parse_args()

    try:
        logger.info("********start uplift optimization ************")
        parameters_path = args.parameters_data_path
        logger.info("input path is : " + parameters_path)
        with open(parameters_path, 'r', encoding='utf8') as fp:
            parameters = json.load(fp)
        logger.info("input parameters data  is : " + str(parameters))

        target_data_path = parameters["target_data_path"]
        train_data_path = parameters["train_data_path"]
        inference_data_path = parameters["inference_data_path"]
        feature_columns = parameters["feature_columns"]
        treatment_columns_category = parameters["treatment_columns_category"]
        outcome_column = parameters["outcome_column"]
        userid_column = parameters["userid_column"]
        coupon_limit = parameters["coupon_limit"]

        if train_data_path.endswith("csv"):
            train_data = pd.read_csv(train_data_path, error_bad_lines=False)
            inference_data = pd.read_csv(inference_data_path, error_bad_lines=False)
        else:
            train_data = pd.read_excel(train_data_path)
            inference_data = pd.read_excel(inference_data_path)

        pd_uplift, coupon_list, treatment_probability = dml_inference(train_data, inference_data, feature_columns,
                                                                      treatment_columns_category, outcome_column,
                                                                      userid_column)

        # auuc_output(train_data, feature_columns, outcome_column, treatment_columns_category_dict)

        # 分块计算，运筹优化的时间复杂度是指数上升的
        result = []
        bath_size = 19998  # 分块的大小，随机拍的
        # bath_size = 99  # 分块的大小，随机拍的
        epoch = int(len(pd_uplift) / bath_size) + 1
        logger.info(f"optimization bath_size = {bath_size}, epoch = {epoch}")
        for i in range(0, epoch):
            m = i * bath_size
            n = (i + 1) * bath_size
            if i == epoch - 1:
                n = len(pd_uplift)
                if (n - m) == 0:
                    continue
            logger.info(f"start sub chunk {m}  --  {n}")
            sub_result = optimization_define(treatment_probability[m:n], pd_uplift[m:n], coupon_limit / epoch,
                                             coupon_list, userid_column)
            result.extend(sub_result)

        # 不分块计算
        # result = optimization_define(treatment_probability, pd_uplift, coupon_limit, coupon_list, userid_column)

        logger.info(f"totally hava {len(result)} workers was treatmented")

        with open(target_data_path, "w", encoding="utf8") as tf:
            json.dump(result, tf, ensure_ascii=False, indent=2, cls=NpEncoder)
        logger.info(f"final result file is : {target_data_path}")
        logger.info("**********\n\n\n\n")


    except Exception as e:
        logger.error("error detail is :{}, {} ".format(e.__class__.__name__, e))
        # exc_type, exc_value, exc_traceback = sys.exc_info()
        logger.error(str(traceback.print_exc()))
        logger.error("****something wrong****\n\n\n")
