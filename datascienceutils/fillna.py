import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator


class FillNA(TransformerMixin, BaseEstimator):
    """Transformation object that fills empty cells in a pandas DataFrame with
    an intelligently picked value.

    Parameters
    ----------
    column_types : dictionary
        A dictionary where column value types key to a names of columns of
        an appropraite data type.

    Example
    -------
    column_types = {
        'continuous_num_cols': [num_col_name_0, num_col_name_1],
        'onehot_categorical_cols': [cat_col_name_0, cat_col_name_1],
        'ordinal_cols': [ord_col_name_0, ord_col_name_1],
    }
    """
    def __init__(self, column_types):
        self.column_types = column_types

    def fit(self, X, y=None, **fit_params):
        """
        Sets the fillna values for each column type. Creates a dictionary where
        dict[column_type][column_name] keys to a fill value.

        Parameters
        ----------
        X : pandas.DataFrame
            The data to fit the transformer with.
        y : array-like, default None
            The target values.

        Returns
        -------
        self
        """
        df = X.copy()
        self.fill_vals_ = {'continuous_num_cols': {},
                           'onehot_categorical_cols': {},
                           'ordinal_cols': {}}

        for col in self.column_types['continuous_num_cols']:
            # fill NAs for continuous numerical_columns with the mean
            if col not in df.columns:
                continue
            self.fill_vals_['continuous_num_cols'][col] = df[col].mean()

        for col in self.column_types['onehot_categorical_cols']:
            # fill NAs for binarized columns with 'UNKNOWN'
            if col not in df.columns:
                continue
            self.fill_vals_['onehot_categorical_cols'][col] = 'UNKNOWN'

        for col in self.column_types['ordinal_cols']:
            # fill ordinal columns with the most frequently occuring value
            if col not in df.columns:
                continue
            series = df[col].copy()
            series = series.dropna()
            self.fill_vals_['ordinal_cols'][col] = series.mode()[0]

        return self

    def transform(self, X, y=None, **transform_params):
        """Fills columns in X with values computed in .fit()

        Parameters
        ----------
        X : pandas.DataFrame
            Data to transform.
        y : array-like, default None
            Target values.

        Returns
        -------
        pandas.DataFrame
            The input DataFrame with the NA values filled.
        """
        X = X.copy()
        for column_type in self.fill_vals_.keys():
            for col in self.column_types[column_type]:
                if col not in X:
                    continue
                X[col] = X[col].fillna(self.fill_vals_[column_type][col])

        return X

