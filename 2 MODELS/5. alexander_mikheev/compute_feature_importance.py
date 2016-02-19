import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


pd.set_option('display.max_columns', None)


filename = 'data/public/train.csv'

df = pd.read_csv(filename)

def meta_dataframe(df, uniq_examples=7):
    from collections import defaultdict
    res = defaultdict(list)
    for i in range(df.shape[1]):
        res['col_name'].append(df.columns[i])
        uniques = df.iloc[:,i].unique()
        notnull_rate = df.iloc[:,i].dropna().size / df.iloc[:,i].size
        res['n_uniques'].append(uniques.size)
        res['n_notnull'].append(notnull_rate)
        res['dtype'].append(df.iloc[:,i].dtype)
        for j in range(1, uniq_examples + 1):
            v = uniques[j-1] if j <= uniques.size else ''
            res['value_' + str(j)].append(v)
    return pd.DataFrame(res, columns=sorted(res.keys())).set_index('col_name')

meta_df = meta_dataframe(df.dropna(how='all', axis=[0, 1]))

meta_df.sort_values('n_notnull', ascending=False)

# ## Prediction model

from sklearn.cross_validation import train_test_split

from sklearn.metrics import mean_squared_error



df_tmp = df[df['Part of'] == 'France']

df = df.drop(df_tmp.index)

from regressor import Regressor
from feature_extractor import FeatureExtractor

df_features = df.drop('target', axis=1)
y = df.target.values

df_train, df_test, y_train, y_test = train_test_split(df_features, y, test_size=0.5, random_state=42)


feature_extractor = FeatureExtractor()
model = Regressor()


X_train = feature_extractor.transform(df_train)
model.fit(X_train, y_train)

X_test = feature_extractor.transform(df_test)
y_pred = model.predict(X_test)
print('RMSE = ', np.sqrt(mean_squared_error(y_test, y_pred)))


imputer = model.clf.named_steps['imputer']

valid_idx = imputer.transform(np.arange(df_train.shape[1])).astype(np.int)
et = model.clf.named_steps['extratreesregressor']

feature_importances = pd.DataFrame(data=et.feature_importances_,
                                   index=df_train.columns[valid_idx][0])
feature_importances['counts'] = df_train.count()[valid_idx][0]
feature_importances.to_csv('feature_importance.csv')


