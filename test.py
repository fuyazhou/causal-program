import econml
## Ignore warnings
import warnings

warnings.filterwarnings('ignore')
# Main imports
from econml.dml import DML, LinearDML, SparseLinearDML, NonParamDML
from econml.metalearners import XLearner, TLearner, SLearner, DomainAdaptationLearner
from econml.dr import DRLearner

import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Treatment effect function
def exp_te(x):
    return x[:, 0] > 0.5


np.random.seed(123)
n = 5000
support_size = 5
n_x = 10
# Outcome support
support_Y = np.random.choice(range(n_x), size=support_size, replace=False)
coefs_Y = np.random.uniform(0, 1, size=support_size)
epsilon_sample = lambda n: np.random.uniform(-1, 1, size=n)
# Treatment support
support_T = support_Y
coefs_T = np.random.uniform(0, 1, size=support_size)
eta_sample = lambda n: np.random.uniform(-1, 1, size=n)

# Generate controls, covariates, treatments and outcomes
X = np.random.uniform(0, 1, size=(n, n_x))
# Heterogeneous treatment effects
TE = exp_te(X)
# Define treatment
log_odds = np.dot(X[:, support_T], coefs_T) + eta_sample(n)
T_sigmoid = 1 / (1 + np.exp(-log_odds))
T = np.array([np.random.binomial(1, p) for p in T_sigmoid])
# Define the outcome
Y = TE * T + np.dot(X[:, support_Y], coefs_Y) + epsilon_sample(n)

# get testing data
X_test = np.random.uniform(0, 1, size=(n, n_x))
X_test[:, 0] = np.linspace(0, 1, n)
expected_te_test = exp_te(X_test)

reg = lambda: RandomForestRegressor(min_samples_leaf=10)
clf = lambda: RandomForestClassifier(min_samples_leaf=10)

X_train, X_val, T_train, T_val, Y_train, Y_val = train_test_split(X, T, Y, test_size=.4)

Y_train1 = np.random.randint(0, 3, size=Y_train.shape)

m1 = LinearDML(model_y=clf(), model_t=clf(), discrete_treatment=True,
               linear_first_stages=False, cv=1)

est = m1.fit(Y_train1, T_train, X=X_train)

m2 = LinearDML(model_y=reg(), model_t=clf(), discrete_treatment=True,
               linear_first_stages=False, cv=1)
est2 = m2.fit(Y_train1, T_train, X=X_train)

q = est.effect(X_train)
m = est.models_y[0][0].predict_proba(X_train[0:3])
print(q)
print(m)
