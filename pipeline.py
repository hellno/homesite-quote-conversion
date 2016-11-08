import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline, make_union, Pipeline, FeatureUnion
from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score
from sklearn.cross_validation import train_test_split


class DateSplitter(BaseEstimator, TransformerMixin):

    def __init__(self, attributes=['day', 'month', 'year']):
        self.attributes = attributes

    def fit(self, x, y=None):
        return self

        for attr in self.attributes:
            df[attr] = getattr(dt.dt, attr)

        return df

    def transform(self, dates):
        new_df = pd.DataFrame()
        df = pd.to_datetime(dates.ix[:, 0])

        for attr in self.attributes:
            new_df[attr] = getattr(df.dt, attr)

        return new_df


class DictMaker(BaseEstimator, TransformerMixin):

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return list(x.T.to_dict().values())


class MissingValuesFiller(BaseEstimator, TransformerMixin):

    def __init__(self, default=-1000):
        self.default = default

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return x.fillna(self.default)


class PandasSelector(BaseEstimator, TransformerMixin):

    def __init__(self, dtype=None, columns=None, inverse=False,
                 return_vector=True):
        self.dtype = dtype
        self.columns = columns
        self.inverse = inverse
        self.return_vector = return_vector

    def check_condition(self, x, col):
        cond = (self.dtype is not None and x[col].dtype == self.dtype) or \
               (self.columns is not None and col in self.columns)
        return self.inverse ^ cond

    def fit(self, x, y=None):
        return self

    def _check_if_all_columns_present(self, x):
        if not self.inverse and self.columns is not None:
            missing_columns = set(self.columns) - set(x.columns)
            if len(missing_columns) > 0:
                missing_columns_ = ','.join(col for col in missing_columns)
                raise KeyError('Keys are missing in the record: %s' %
                               missing_columns_)

    def transform(self, x):
        # check if x is a pandas DataFrame
        if not isinstance(x, pd.DataFrame):
            raise KeyError('Input is not a pandas DataFrame')

        selected_cols = []
        for col in x.columns:
            if self.check_condition(x, col):
                selected_cols.append(col)

        # if the column was selected and inversed = False make sure the column
        # is in the DataFrame
        self._check_if_all_columns_present(x)

        # if only 1 column is returned return a vector instead of a dataframe
        if len(selected_cols) == 1:
            if self.return_vector:
                return x[selected_cols[0]]
            else:
                return pd.DataFrame({selected_cols[0]: x[selected_cols[0]]})
        else:
            return x.ix[:, selected_cols]

print('script start')

processing_pipeline = Pipeline([
    ('feature_selector', PandasSelector(inverse=True,
                                        columns=['QuoteNumber', 'QuoteConversion_Flag'])),
    ('features', FeatureUnion([
        ('Imputer', Pipeline([
            ('selector', PandasSelector(dtype="O", inverse=True)),
            ('transformer', Imputer(strategy='mean', axis=0))
        ])),
        ('Dates', Pipeline([
            ('selector', PandasSelector(columns=[
                'Original_Quote_Date'], return_vector=False)),
            ('transformer', DateSplitter())
        ])),
        ('Categorical', Pipeline([
            ('selector', PandasSelector(dtype="O")),
            ('transformer', MissingValuesFiller()),
            ('dict_maker', DictMaker()),
            ('OneHotEncoder', DictVectorizer(sparse=False))
        ]))
    ]))
])

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

rf = RandomForestClassifier(n_estimators=500,
                            n_jobs=-1,
                            max_depth=5,
                            )

print('fitting rf pipeline')

pipeline = make_pipeline(processing_pipeline, rf)

X_train, X_test, y_train, y_test = train_test_split(
    train, train.QuoteConversion_Flag, test_size=0.2)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict_proba(X_test)[:, 1]
print('rf roc auc score: {}'.format(roc_auc_score(y_test, y_pred)))

# XGBoost Classifier

from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    learning_rate=0.023,
    nthread=4,
    objective="binary:logistic",
    max_depth=6,
    subsample=0.83,
    colsample_bytree=0.77
)

print('fitting xgb pipeline')
boost_pipeline = make_pipeline(processing_pipeline, xgb_model)

boost_pipeline.fit(X_train, y_train)
y_pred = boost_pipeline.predict_proba(X_test)[:, 1]
print('xgb roc auc score: {}'.format(roc_auc_score(y_test, y_pred)))

# output to CSV for Kaggle
# uses all of training data

xgb_model = XGBClassifier(
    learning_rate=0.023,
    nthread=4,
    objective="binary:logistic",
    max_depth=6,
    subsample=0.83,
    colsample_bytree=0.77
)

print('training on train data')

boost_pipeline = make_pipeline(processing_pipeline, xgb_model)
boost_pipeline.fit(train, train.QuoteConversion_Flag)

print('predicting on test data')
preds = boost_pipeline.predict_proba(test)[:, 1]

test['QuoteConversion_Flag'] = preds

test[['QuoteNumber', 'QuoteConversion_Flag']].to_csv(
    'xgb_kaggle.csv', index=False)

print('done')
