import pandas as pd
from sklearn import preprocessing
 
 
class FeatureExtractor(object):
    core_cols = ['Year', 'income', 'fastfood_spending', 'alcool_consumption', 'nickel_emission', 'chromium_emission']
    core_cols += ['measles_vacc_1', 'polio_vacc', 'tetanus_vacc', 'diphteria_vacc']
    core_cols += ['hepb_vacc', 'shale_oil', 'transplants_prevalence', 'cadmium_export', 'companies_indus']
    region_cols = ['RegionType', 'Part of', 'Region']
    categ_cols = ['Gender', 'Age', 'MainOrigin', 'cancer_type']
    additional_cols = ['HIV_15_49']
 
    def __init__(self):
        pass
 
    def fit(self, X_df, y_array):
        pass
 
    def transform(self, X_df):
        ret = X_df[self.core_cols].copy()
        # dummify the categorical variables
        for col in self.categ_cols:
            ret = ret.join(pd.get_dummies(X_df[col], prefix=col[:3]))
        # add extra information
        for col in self.additional_cols:
            ret[col] = X_df[col]
 
        #ret = ret.dropna()
        return ret.values