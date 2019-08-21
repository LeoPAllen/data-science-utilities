"""
Collection of transformation objects that clean categorical data
"""

import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator

# I set these arbitrarily, use at your peril
DEFAULT_UNIQUENESS_THESHOLD = .1
DEFAULT_FREQUENCY_THRESHOLD = .01

class FlatCase(TransformerMixin, BaseEstimator):
    """Flattens categorical data to all lowercase. Removes training
    whitespaces. Wraps pandas functionality for pipeline insertion
    """
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(X, y=None, **transform_params):
        df = X.copy()
        for col in self.column
            df[col] = df[col].str.strip()
            df[col] = df[col].str.lower()

        return df

class FrequencyFlatten(TransformerMixin, BaseEstimator):
    """Flattens observations that fall below a frequency threshold within one
    feature space to a single filler variable. The idea motivating this
    """


class UniquenessDrop(TransformerMixin, BaseEstimator):
    """Drops columns where the number of unique entries in the column exceeds
    a passed in uniqueness threshold.
    """
    def __init__(self, columns, unique_theshold=DEFAULT_UNIQUENESS_THESHOLD):
        self.columns = columns
        self.u_t = unique_theshold

    def fit(self, X, y=None, **fit_params):
        """Creates a list of columns that have too many unique categorical
        variable values to one hot encode
        """
        df = X.copy()
        n_entries = len(df)
        keep_columns = []
        for col in self.columns:
            assert col is in df.columns,'check column list'
            unique_count = df[col].nunique()
            if unique_count / n_entries >= self.u_t:
                keep_columns.append(col)

        print(f'The following list of columns have too many unique entries to be meaningfully one-hot-encoded: {keep_columns}')
        self.keep_columns_ = keep_columns

        return self

    def transform(self, X, y=None, **transform_params):
        """Drops columns if they have too many unique values
        """
        return X[self.keep_columns_]


