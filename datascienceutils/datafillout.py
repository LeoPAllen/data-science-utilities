import pandas as pd
import numpy as np

from sklearn.base import TransformerMixin, BaseEstimator


# Huristic Fillout Rate Thresholds
RECORD_FILLOUT_P = 0.5

COL_FILLOUT_P = {
    'continuous_num_cols': 0.5,
    'onehot_categorical_cols': 0.5,
    'ordinal_cols': 0.5
}

class RecordDrop(TransformerMixin, BaseEstimator):
    """Transformation object that drops records that have a fillout rate below
    some threshold fill_p

    Parameters
    ----------
    fill_p : float
        minimum fillout proportion
    """
    def __init__(self, fill_p=RECORD_FILLOUT_P):
        self.fill_proportion = fill_p

    def fit(self, X, y=None, **fit_params):
        """Generates a list of record indexes to keep in the training DataFrame
        based on the fillout rate of each row. Rows need to exceed a fillout
        rate fill_p to be included in training set.

        Parameters
        ----------
        X : pandas.DataFrame
            Data to train a model
        y : pandas.DataFrame
        """
        df = X.copy()
        n_cols = len(df.columns)
        fill_rate_s = (df.notnull().astype(int).sum(axis=1)) / n_cols
        fill_rate_s = fill_rate_s[fill_rate_s > self.fill_proportion]
        self.records_to_keep_ = list(fill_rate_s.index.values)

        return self

    def transform(self, X, y=None, **transform_params):
        """Drops all records but ones with a fillout rate above fill_p

        Parameters
        ----------
        X : pandas.DataFrame
            Data to train a model
        y : pandas.DataFrame
            Target values.

        Returns
        -------
        X : pandas.DataFrame
            Data to train a model, with acceptable fillout rate

        """
        return X.loc[self.records_to_keep_]

class ColumnDrop(TransformerMixin, BaseEstimator):
    """Transformation object that drops columns that have a fillout rate below
    a set of threshold values specified in fill_p. Different variable types
    require different fillout proportions for inclusion

    Parameters
    ----------
    fill_p : dictionary
        column types key to minimum fillout proportions
    """
    def __init__(self, column_types, fill_p=COL_FILLOUT_P):
        self.column_types = column_types
        self.fill_proportions = fill_p
        self.column_p_map = self._map_column_to_fill_p()

    def _map_column_to_fill_p(self):
        """Utility method that defines a map (dictionary) where column
        names key to required fillout proportions. Executes on object initialization.
        """
        col_p_dict = {}
        for dtype in self.column_types.keys():
            for col in self.column_types[dtype]:
                assert col not in col_p_dict.keys(),"Duplicate in column_types"
                col_p_dict[col] = self.fill_proportions[dtype]

        return col_p_dict


    def fit(self, X, y=None, **fit_params):
        """Generates a list of columns to include in a dataset prior to further
        munging and processing.
        """
        df = X.copy()
        n_rows = len(df)
        fill_rate_s = (df.notnull().astype(int).sum(axis=0)) / n_rows
        fill_rate_s = fill_rate_s.rename('fillout_rate')
        fr_df = fill_rate_s.to_frame()
        fr_df['theshold'] = fr_df.index.map(self.column_p_map)
        fr_df['include'] = np.where(fr_df['fillout_rate'] > fr_df['theshold'],
                                    1,
                                    0)
        include_fr_df = fr_df[fr_df['include'] == 1]
        self.cols_to_keep_ = list(include_fr_df.index.values)
        return self


    def transform(self, X, y=None, **transform_params):
        """Transforms a training or test dataset by reducing the column space
        based on the fit.
        """
        return X[self.cols_to_keep_]
