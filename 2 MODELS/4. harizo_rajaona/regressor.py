import pandas as pd
from sklearn.ensemble import *
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import Imputer, MinMaxScaler, PolynomialFeatures
from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline
from sklearn import linear_model
from sklearn.preprocessing import MaxAbsScaler
 
 
class Regressor(BaseEstimator):
    def __init__(self):
        self.clf = make_pipeline(
            Imputer(),
            MaxAbsScaler(),
            ExtraTreesRegressor(n_estimators=200, criterion='mse', max_depth=None, min_samples_split=10, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=False, oob_score=False, n_jobs=1, random_state=42, verbose=0, warm_start=False)
            )
 
 
    def fit(self, X, y):
        return self.clf.fit(X, y)
 
    def predict(self, X):
        return self.clf.predict(X)