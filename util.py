import os
import pandas as pd
import numpy as np
from econml.dml import LinearDML, CausalForestDML, NonParamDML
from econml.dr import LinearDRLearner
from econml.cate_interpreter import SingleTreeCateInterpreter
from sklearn.linear_model import LogisticRegressionCV, LinearRegression, LogisticRegression, Lasso, LassoCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
import lightgbm as lgb
from xgboost.sklearn import XGBClassifier, XGBRegressor
from econml.sklearn_extensions.model_selection import GridSearchCVList
import logging

logging.basicConfig(filename='casual_inference_miaosuan.log.txt',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%D:%H:%M:%S',
                    level=logging.DEBUG)

logging.info("Running Urban Planning")
logger = logging.getLogger('urbanGUI')


def grid_search_reg():
    return GridSearchCVList([LassoCV(),
                             RandomForestRegressor(),
                             lgb.LGBMRegressor()],
                            param_grid_list=[{},
                                             {'n_estimators': [50, 100],
                                              'max_depth': [3, None]},
                                             {'n_estimators': [50, 100],
                                              'max_depth': [3, None]}],
                            cv=2,
                            scoring='neg_mean_squared_error',
                            n_jobs=-1)


def grid_search_clf():
    return GridSearchCVList([LogisticRegressionCV(),
                             RandomForestClassifier(),
                             lgb.LGBMClassifier()],
                            param_grid_list=[{},
                                             {'n_estimators': [50, 100],
                                              'max_depth': [3, None]},
                                             {'n_estimators': [50, 100],
                                              'max_depth': [3, None]}],
                            cv=2,
                            scoring='neg_mean_squared_error',
                            n_jobs=-1)


def linear_dml(x, t, y, discrete, grid_search):
    if discrete:
        if grid_search:
            model_t = grid_search_clf().fit(x, t).best_estimator_
            model_y = grid_search_reg().fit(x, y).best_estimator_
        if not grid_search:
            model_t = XGBClassifier()
            model_y = XGBRegressor()

    if not discrete:
        if grid_search:
            model_t = grid_search_reg().fit(x, t).best_estimator_
            model_y = grid_search_reg().fit(x, y).best_estimator_
        if not grid_search:
            model_t = XGBRegressor()
            model_y = XGBRegressor()

    est = LinearDML(model_y=model_y, model_t=model_t, discrete_treatment=discrete)
    est.fit(Y=y, T=t, X=x)
    return est
