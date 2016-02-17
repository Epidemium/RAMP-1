import pandas as pd
from sklearn.ensemble import *
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import Imputer, MinMaxScaler, PolynomialFeatures
from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline
 
 
class Regressor(BaseEstimator):
    def __init__(self):
        self.clf = make_pipeline(
            Imputer(),
            MinMaxScaler(),
            GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.7))
 
 
    def fit(self, X, y):
        return self.clf.fit(X, y)
 
    def predict(self, X):
        return self.clf.predict(X)