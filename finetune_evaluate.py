# 模型的调参 和 uplift模型的评估

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from catboost import CatBoostRegressor
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import math
from hyperopt import hp
import numpy as np
from sklearn.metrics import mean_squared_error
from causalml.metrics.visualize import plot_lift, plot_qini, plot_gain, plot_tmlegain, auuc_score, qini_score, \
    get_tmlegain
from causalml.propensity import ElasticNetPropensityModel
from sklearn.metrics import r2_score, mean_squared_error
import logging
from collections import defaultdict

# 不输出warn消息
import warnings

warnings.filterwarnings('ignore')

# 输出info消息
import logging

log = logging.getLogger()
log.setLevel(logging.INFO)


# def parameter_tuning_hyperopt():
def params_setting():
    # XGB parameters
    xgb_reg_params = {
        'learning_rate': hp.choice('learning_rate', np.arange(0.05, 0.31, 0.05)),
        'max_depth': hp.choice('max_depth', np.arange(5, 16, 1, dtype=int)),
        'min_child_weight': hp.choice('min_child_weight', np.arange(1, 8, 1, dtype=int)),
        'colsample_bytree': hp.choice('colsample_bytree', np.arange(0.3, 0.8, 0.1)),
        'subsample': hp.uniform('subsample', 0.8, 1),
        'n_estimators': 100,
    }
    xgb_fit_params = {
        'eval_metric': 'rmse',
        'early_stopping_rounds': 10,
        'verbose': False
    }
    xgb_para = dict()
    xgb_para['reg_params'] = xgb_reg_params
    xgb_para['fit_params'] = xgb_fit_params
    xgb_para['loss_func'] = lambda y, pred: np.sqrt(mean_squared_error(y, pred))

    # LightGBM parameters
    lgb_reg_params = {
        'learning_rate': hp.choice('learning_rate', np.arange(0.05, 0.31, 0.05)),
        'max_depth': hp.choice('max_depth', np.arange(5, 16, 1, dtype=int)),
        'min_child_weight': hp.choice('min_child_weight', np.arange(1, 8, 1, dtype=int)),
        'colsample_bytree': hp.choice('colsample_bytree', np.arange(0.3, 0.8, 0.1)),
        'subsample': hp.uniform('subsample', 0.8, 1),
        'n_estimators': 100,
    }
    lgb_fit_params = {
        'eval_metric': 'l2',
        'early_stopping_rounds': 10,
        'verbose': False
    }
    lgb_para = dict()
    lgb_para['reg_params'] = lgb_reg_params
    lgb_para['fit_params'] = lgb_fit_params
    lgb_para['loss_func'] = lambda y, pred: np.sqrt(mean_squared_error(y, pred))

    # CatBoost parameters
    ctb_reg_params = {
        'learning_rate': hp.choice('learning_rate', np.arange(0.05, 0.31, 0.05)),
        'max_depth': hp.choice('max_depth', np.arange(5, 16, 1, dtype=int)),
        'colsample_bylevel': hp.choice('colsample_bylevel', np.arange(0.3, 0.8, 0.1)),
        'n_estimators': 100,
        'eval_metric': 'RMSE',
    }
    ctb_fit_params = {
        'early_stopping_rounds': 10,
        'verbose': False
    }
    ctb_para = dict()
    ctb_para['reg_params'] = ctb_reg_params
    ctb_para['fit_params'] = ctb_fit_params
    ctb_para['loss_func'] = lambda y, pred: np.sqrt(mean_squared_error(y, pred))
    return xgb_para, lgb_para, ctb_para


import lightgbm as lgb
import xgboost as xgb
import catboost as ctb
from hyperopt import fmin, tpe, STATUS_OK, STATUS_FAIL, Trials


class HPOpt(object):
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def process(self, fn_name, space, trials, algo, max_evals):
        fn = getattr(self, fn_name)
        try:
            result = fmin(fn=fn, space=space, algo=algo, max_evals=max_evals, trials=trials)
        except Exception as e:
            return {'status': STATUS_FAIL,
                    'exception': str(e)}
        return result, trials

    def xgb_reg(self, para):
        reg = xgb.XGBRegressor(**para['reg_params'])
        return self.train_reg(reg, para)

    def lgb_reg(self, para):
        reg = lgb.LGBMRegressor(**para['reg_params'])
        return self.train_reg(reg, para)

    def ctb_reg(self, para):
        reg = ctb.CatBoostRegressor(**para['reg_params'])
        return self.train_reg(reg, para)

    def train_reg(self, reg, para):
        reg.fit(self.x_train, self.y_train,
                eval_set=[(self.x_train, self.y_train), (self.x_test, self.y_test)],
                **para['fit_params'])
        pred = reg.predict(self.x_test)
        loss = para['loss_func'](self.y_test, pred)
        return {'loss': loss, 'status': STATUS_OK, 'model': reg}


# 调参，返回最优的参数
def parameter_tuning_hyperopt(x_train, x_test, y_train, y_test):
    obj = HPOpt(x_train, x_test, y_train[["outcome"]], y_test[["outcome"]])
    xgb_para, lgb_para, ctb_para = params_setting()
    xgb_opt = obj.process(fn_name='xgb_reg', space=xgb_para, trials=Trials(), algo=tpe.suggest, max_evals=100)
    lgb_opt = obj.process(fn_name='lgb_reg', space=lgb_para, trials=Trials(), algo=tpe.suggest, max_evals=50)
    ctb_opt = obj.process(fn_name='ctb_reg', space=ctb_para, trials=Trials(), algo=tpe.suggest, max_evals=50)
    return xgb_opt, lgb_opt, ctb_opt


import logging

logging.basicConfig()
logger = logging.getLogger("NBOE")
logger.setLevel(logging.INFO)


def plot_lift_curve(pred_df, y_col, treatment_col):
    plot_lift(pred_df, outcome_col=y_col,
              treatment_col=treatment_col, figsize=(8, 8))


def plot_uplift_curve(pred_df, y_col, treatment_col):
    plot_gain(pred_df, outcome_col=y_col,
              treatment_col=treatment_col, figsize=(8, 8), normalize=True)


def plot_qini_curve(pred_df, y_col, treatment_col):
    plot_qini(pred_df, outcome_col=y_col,
              treatment_col=treatment_col, figsize=(8, 8), normalize=True)


def plot_tmle_curve(pred_df, inference_col, y_col, treatment_col, p_col):
    pred_df[p_col] = 0.5
    plot_tmlegain(pred_df, inference_col, outcome_col=y_col,
                  treatment_col=treatment_col, p_col=p_col)


def get_auuc(pred_df, y_col, treatment_col):
    auuc_df = pd.DataFrame(auuc_score(pred_df, outcome_col=y_col,
                                      treatment_col=treatment_col, normalize=True)).reset_index()
    auuc_df.columns = ['Learner', 'auuc']
    auuc_df['Lift'] = (auuc_df['auuc'] /
                       auuc_df[auuc_df.Learner == 'Random'].auuc.values) - 1
    auuc_df = auuc_df.sort_values('auuc', ascending=False)
    logger.info(auuc_df)
    return auuc_df


def get_qini(pred_df, y_col, treatment_col):
    qini_df = pd.DataFrame(qini_score(pred_df, outcome_col=y_col,
                                      treatment_col=treatment_col, normalize=True)).reset_index()
    qini_df.columns = ['Learner', 'qini']
    qini_df = qini_df.sort_values('qini', ascending=False)
    logger.info(qini_df)
    return qini_df
