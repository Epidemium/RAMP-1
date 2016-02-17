import numpy as np
import pandas as pd
from importlib import import_module
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import mean_squared_error

#train_filename = '../data/public/public_train.csv'
train_filename = '../data/public/train.csv'
n_cv_iter = 3
score = mean_squared_error
target_col = 'target'


def train_submission(X_df, y_array, train_is):
    # Feature extraction
    feature_extractor = import_module('feature_extractor')
    fe = feature_extractor.FeatureExtractor()
    fe.fit(X_df.iloc[train_is], y_array[train_is])
    X_array = fe.transform(X_df.iloc[train_is])
    # Regression
    regressor = import_module('regressor')
    reg = regressor.Regressor()
    reg.fit(X_array, y_array[train_is])
    return fe, reg


def test_submission(trained_model, X_df, test_is):
    fe, reg = trained_model
    # Feature extraction
    X_array = fe.transform(X_df.iloc[test_is])
    # Regression
    y_pred = reg.predict(X_array)
    return y_pred


if __name__ == '__main__':
    print("Reading file ...")
    data = pd.read_csv(train_filename)

    X_df = data.drop(target_col, axis=1)
    y_array = data[target_col].values.astype(np.float32)

    skf = ShuffleSplit(
        len(y_array), n_iter=n_cv_iter, test_size=0.5, random_state=42)

    print("Training model ...")
    for train_is, test_is in skf:
        model = train_submission(X_df, y_array, train_is)
        y_pred = test_submission(model, X_df, test_is)
        print('RMSE = ', np.sqrt(score(y_array[test_is], y_pred)))
