import pandas as pd
 
 
class FeatureExtractor(object):    
    core_cols = ['Year']    
    region_cols = []    
    categ_cols = ['Gender', 'Age', 'MainOrigin','cancer_type','RegionType', 'Part of', 'Region']    
    additional_cols = ['HIV_15_49','g_HBV' ,'alcool_consumption','diphteria_vacc',
                      'tetanus_vacc','pertussis_vacc','polio_vacc','hepb_vacc','measles_vacc_1']       
    
    def __init__(self):
        pass
 
    def fit(self, X_df, y_array):
        pass
 
    def transform(self, X_df):
        ret = X_df[['Year']].copy()
        # dummify the categorical variables
        for col in self.categ_cols:
            ret = ret.join(pd.get_dummies(X_df[col], prefix=col[:3]))
        # add extra information
        for col in self.additional_cols:
            ret[col] = X_df[col]
        return ret.values