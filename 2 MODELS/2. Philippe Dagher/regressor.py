import pandas as pd
from sklearn.ensemble import *
from sklearn.preprocessing import *
from sklearn import linear_model
from sklearn import svm
from sklearn.preprocessing import Imputer
from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline
import numpy as np
 
 
class Regressor(BaseEstimator):
    def __init__(self):
        self.clf1 = make_pipeline(Imputer(),
                                  GradientBoostingRegressor(n_estimators=5000, max_depth=8))
        self.clf2 = make_pipeline(Imputer(),
                                  MaxAbsScaler(),
                                  ExtraTreesRegressor(n_estimators=5000, criterion='mse', max_depth=8,
                                                      min_samples_split=10, min_samples_leaf=1,
                                                      min_weight_fraction_leaf=0.0,
                                                      max_features='auto', max_leaf_nodes=None, bootstrap=False,
                                                      oob_score=False,
                                                      n_jobs=1, random_state=42, verbose=0, warm_start=True))
        self.clf3 = make_pipeline(Imputer(),
                                  svm.LinearSVR())
        self.clf = linear_model.LinearRegression()
 
    def fit(self, X_t, y_t):
        self.X_t = X_t
        self.y_t = y_t
        self.clf1.fit(X_t, y_t)
        self.clf2.fit(X_t, y_t)
        self.clf3.fit(X_t, y_t)
 
        y1 = self.clf1.predict(self.X_t)
        y2 = self.clf2.predict(self.X_t)
        y3 = self.clf3.predict(self.X_t)
 
        d = pd.DataFrame({'y1': y1, 'y2': y2, 'y3': y3}).values
        self.clf.fit(d, self.y_t)
 
    def predict(self, X_cv):
        r1 = self.clf1.predict(X_cv)
        r2 = self.clf2.predict(X_cv)
        r3 = self.clf3.predict(X_cv)
 
        r = pd.DataFrame({'y1': r1, 'y2': r2, 'y3': r3}).values
        return self.clf.predict(r)