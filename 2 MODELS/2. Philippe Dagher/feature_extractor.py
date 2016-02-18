import pandas as pd
import numpy as np
 
 
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
 
def data2percentiles(data, pattern=None):
    percentiles = (rankdata(data, method='min')-1)/len(data)
    if pattern:
        return [pattern[int(x*len(pattern))] for x in percentiles]
    else:
        return percentiles
 
class FeatureExtractor(object):
    
    def __init__(self):
        self.vectorizer = CountVectorizer()
 
    def fit(self, X_df, y_array):
        self.X = X_df.copy()
        self.y = y_array
        self.co_num = [c for c, npt in zip(X_df.columns, X_df.dtypes) if npt == np.dtype('float64') or npt == np.dtype('int64')]
        self.co_obj = [c for c, npt in zip(X_df.columns, X_df.dtypes) if npt == np.dtype('object')]
        for c in self.co_obj:
            self.X[c] = (c+self.X[c]).apply(lambda x: '' if x!=x else str(hash(x)).replace('-','_'))
        self.texts = self.X[self.co_obj].apply(lambda x: ' '.join(x), axis=1)
        self.vectorizer = self.vectorizer.fit(self.texts)
 
    def transform(self, X_df):
        ret = X_df.copy()
        ret_con = [c for c, npt in zip(X_df.columns, X_df.dtypes) if npt == np.dtype('float64') or npt == np.dtype('int64')]
        ret_coo = [c for c, npt in zip(X_df.columns, X_df.dtypes) if npt == np.dtype('object')]
        for c in ret_coo:
            ret[c] = (c+ret[c]).apply(lambda x: '' if x!=x else str(hash(x)).replace('-','_'))
        ret_texts = ret[ret_coo].apply(lambda x: ' '.join(x), axis=1)
        
        vectors = self.vectorizer.transform(ret_texts)
 
        return np.hstack((ret[ret_con].values, vectors.toarray()))