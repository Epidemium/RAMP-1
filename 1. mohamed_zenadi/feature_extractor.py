import pandas as pd
from numpy import NaN
 
 
class FeatureExtractor(object):
    core_cols = ['Year']
    region_cols = ['RegionType', 'Part of', 'Region']
    categ_cols = ['Gender', 'MainOrigin', 'cancer_type'] + region_cols
    additional_cols = ['HIV_15_49', 'income', 'alcool_consumption']
 
    def __init__(self):
        self.more_cols = set()
        pass
 
    def fit(self, X_df, y_array):
        pass
 
    def transform(self, X_df):
        ret = X_df[self.core_cols].copy()
        # dummify the categorical variables
 
        train_data = True
        if self.more_cols:
            train_data = False
        extra_cols = set()
 
        for col in self.categ_cols:
            c = pd.get_dummies(X_df[col], prefix=col[:3])
            if train_data:
                self.more_cols.update({co for co in c.columns.values})
            else:
                extra_cols.update({co for co in c.columns.values})
            ret = ret.join(c)
 
        if not train_data:
            for c in self.more_cols:
                if c not in extra_cols:
                    ret[c] = NaN
            for c in extra_cols:
                if c not in self.more_cols:
                    ret = ret.drop(c, 1)
 
 
        print ret.shape
 
        # add extra information
        for col in self.additional_cols:
            ret[col] = X_df[col]
 
        return ret.values