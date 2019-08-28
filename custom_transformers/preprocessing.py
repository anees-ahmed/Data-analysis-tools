import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from scipy import stats
from sklearn import preprocessing as skl_preproc


class LowFreqCombiner(BaseEstimator, TransformerMixin):
    """
    Renames classes which occur too infrequently to 'Other'.
    Searches through all columns of dtype 'O'.
    Returns dataframe, including columns with dtype other than 'O'.
    
    -----Parameters---------------------------------------------------------------------------------------------------
        tol        : ('auto' or float [0.0, 100.0])
                    If a category in a columns has less than tol instances (in %), then it will be renamed.
                    If float, tol is static. If 'auto', tol for each column is calculated as
                    expected_frequency/tol_ratio, where expected_frequency = 100.0/<number of categories in column> %.
        tol_ratio  : (positive float)
                    Only used if tol='auto'
    -----Methods------------------------------------------------------------------------------------------------------
        too_few_   : nested list of categories with too few instances
        
    -----Issues-------------------------------------------------------------------------------------------------------
    1. If there is a single low frequency category in a given column, it gets renamed to Other as well, which is
        completely unnecessary
"""
    
    def __init__(self, tol='auto', tol_ratio=2.5):
        self.tol = tol
        self.tol_ratio = tol_ratio
        
        
    def fit(self, df):
        df = df.copy()
        
        # list of classes with too few instances
        self.too_few_ = []
        
        # extract categorical column names
        self.cat_cols = df.dtypes[df.dtypes=='O'].index.tolist()
        
        # make a list of tolerances
        self.n_features = len(self.cat_cols)
        self.n_cats = [len(df[col].unique()) for col in self.cat_cols]

        if self.tol is 'auto':
            self.tol_list = [100.0/(n*self.tol_ratio) for n in self.n_cats]
        elif type(self.tol) != 'float':
            raise ValueError('tol must be either \'auto\' or a float between 0.0 and 100.0')
        else:
            self.tol_list = [self.tol]*self.n_features
            
        # populate self.too_few
        for col, tolerance in zip(self.cat_cols, self.tol_list):
            vc = df[col].value_counts()*100.0/df.shape[0]
            vc = vc[vc < tolerance]
            to_append = vc.index.tolist()
            self.too_few_.append(to_append)

        return self
    
    
    def transform(self, df):
        df = df.copy()

        # add any categories present only in transform dataframe to self.to_transform,
        # else there will be a dimension (size) mismatch.
        # this assumes that the transform dataframe isn't so wildly different from the fit dataframe,
        # that an otherwise common in transform dataframe is entirely missing from the fit dataframe.
        for col, tolerance in zip(self.cat_cols, self.tol_list):
            vc = df[col].value_counts()*100/df.shape[0]
            vc = vc[vc < tolerance] 
            
            for cat in vc.index:
                if cat not in self.too_few_[self.cat_cols.index(col)]:
                    self.too_few_[self.cat_cols.index(col)].append(cat)
        
        # prepare mapping dict
        for cats, col in zip(self.too_few_, self.cat_cols):
            cat_map = {cat: 'Other' for cat in cats}
    
            if cat_map: # to avoid mapping from empty dictionaries
                df[col] =  df[col].map(cat_map).fillna(df[col])
                # map() return NaN values unless the mapping dictionary is exhaustive
                # the alternative replace() is slow, but doesnt have this issue

        return df



class OHE(BaseEstimator, TransformerMixin):
    """
    Transformer version of pandas.get_dummies - one hot encodes all categorical columns (=dtype 'O')
    
    -----Parameters---------------------------------------------------------------------------------------------------
        drop_first: (boolean)
                    If true, drop first dummy column
    """
    
    def __init__(self, drop_first=True):
        self.drop_first = drop_first
        
        
    def fit(self, df):
        df = df.copy()
        
        # columns to encode
        self.to_encode = df.dtypes[df.dtypes=='O'].index.values
            
        # store all (including dummy) columns of the one-hot-encoded dataframe in self.all_cols
        # make sure self.to_encode is not empty
        # if it is, then self.all_cols should contain the original columns
        if self.to_encode.size == 0:
            self.all_cols = df.columns.values
        else:
            self.all_cols = pd.get_dummies(df,
                                           columns=self.to_encode,
                                           drop_first=self.drop_first,
                                           dtype='float32').columns.values
        return self
    
    
    def transform(self, df):
        df = df.copy()
        
        # check if self.to_encode is empty, in which case return the original unmodified dataframe
        if self.to_encode.size == 0:
            return df
        else:
            df_ohe = pd.get_dummies(df,
                                    columns=self.to_encode,
                                    drop_first=self.drop_first,
                                    dtype='float32')
            
            # add dummy features that were present in the fit dataframe, but missing from the transformed dataframe
            for col in self.all_cols:
                if col not in df_ohe.columns:
                    df_ohe[col] = 0.0
                
            return df_ohe[self.all_cols]



class Scaler(BaseEstimator, TransformerMixin):
    """
    Rescales dataframe, and outputs dataframe
    
    -----Parameters---------------------------------------------------------------------------------------------------
        scaling          : (string: 'Standard', 'MinMax', 'Robust')
                            Standard sklearn scalers
        cols_to_rescale  : (tuple of strings, or None)
                            If tuple, it must contain columns in the dataframe to be scaled. Only these columns will be
                            rescaled. If None, all features will be rescaled.
        ignore_binary    : (boolean)
                            Only used if cols_to_rescale is None. If True, binary columns (number of unique
                            categories <= 2) will not be scaled
        binary_correction   : (boolean)
                            Only used if ignore_binary is True (which requires cols_to_rescale=None) and scaling is
                            'Standard'. Rescales non-binary columns a second time by a factor of 0.5, and de-means
                            binary variable so that means and stds of all variables are approx 0 and 0.5 resp.
                        
    """
    
    def __init__(self, scaling='Standard', cols_to_rescale=None, ignore_binary=True, binary_correction=True):
        self.cols_to_rescale = cols_to_rescale
        self.scaling = scaling
        self.ignore_binary = ignore_binary
        self.binary_correction = binary_correction
        
        
    def fit(self, df):
        df = df.copy()
        
        self.to_scale = np.array(self.cols_to_rescale)
        if self.cols_to_rescale is None:
            if self.ignore_binary:
                # dont scale binary columns
                self.to_scale  = np.array([col for col in df.columns
                                           if (df[col].dtype!='O') and (len(df[col].unique())>2)])
            else:
                self.to_scale  = np.array([col for col in df.columns if df[col].dtype!='O'])

        # if self.to_scale is empty, do nothing
        # otherwise fit the transformation
        if self.to_scale.size!=0:
            # if empty, return scaling is identity
            if self.scaling == 'Standard':
                self.scaler = StandardScaler()
            elif self.scaling == 'MinMax':
                self.scaler = MinMaxScaler()
            elif self.scaling == 'Robust':
                self.scaler = RobustScaler()
            else:
                raise ValueError('scaling takes on of these values: \'Standard\',\'MinMax\' or \'Robust\'')

            X = df[self.to_scale]
            self.scaler.fit(X)

        return self
    
    
    def transform(self, df):
        df_transf = df.copy()
        
        if self.to_scale.size==0:
            return df_transf
        else:
            X = df[self.to_scale]
            X_transf = pd.DataFrame(self.scaler.transform(X),
                                    columns=self.to_scale,
                                    index=df.index)
            for col in self.to_scale:
                df_transf[col] = X_transf[col]
                
            if self.binary_correction:
                if (self.scaling=='Standard') and (self.cols_to_rescale is None) and (self.ignore_binary==True):
                    # correction
                    df_transf[self.to_scale] = df_transf[self.to_scale]/2.0
                     # factor of 2 makes sure binary and non binary columns have std roughly 0.5
                    for col in df_transf.columns:
                        if (df_transf[col].dtype!='O') and (len(df_transf[col].unique())<=2):
                            # this makes sure binary variables also have 0 mean
                            df_transf[col] = df_transf[col] - df_transf[col].mean()
            
            return df_transf


class KNNImputer(BaseEstimator, TransformerMixin):
    """
    Imputes a single column using K nearest neighbors algorithm.
    Any columns with null values other than the target column will be ignored.
    Any non-numeric columns will also be ignored.
    
    -----Parameters---------------------------------------------------------------------------------------------------
        column_name  : (string)
                        Column to impute
        k            : (int, positive)
                        Number of neighbors used in KNN algorith
    """
    
    
    def __init__(self, column_name, k=3):
        self.column_name = column_name
        self.k = k
        
        
    def fit(self, df):
        df = df.copy()
        
        # make a list of columns to exclude from the predictor set (X)
        # 1. columns with missing values
        # 2. non-numeric columns
        
        self.to_drop = np.array([col for col in df.columns
                                 if ((df[col].isnull().sum()>0) or (df[col].dtype=='O'))])
        
        # separate target and predictor columns
        Y = df[self.column_name]
        X = df.drop(self.to_drop, axis=1)
        
        # drop rows with null target values
        X_sans_null = X[~Y.isnull()]
        Y_sans_null = Y[~Y.isnull()]
        
        # use KNeighborsClassifier if target column is categorical, otherwise use KNeighborsRegressor
        if Y.dtype == 'O':
            self.estimator =  KNeighborsClassifier(n_neighbors=self.k)
        else:
            self.estimator =  KNeighborsRegressor(n_neighbors=self.k)
            
        self.estimator.fit(X_sans_null, Y_sans_null)
        
        return self
    
    
    def transform(self, df):
        df_transf = df.copy()
        Y = df_transf[self.column_name]
        X = df_transf.drop(self.to_drop, axis=1)
        
        # predict target values
        Y_pred = self.estimator.predict(X)
        
        # fill null values using predicted values
        df_transf[self.column_name] =  df_transf[self.column_name].fillna(pd.Series(Y_pred, index=df.index))

        return df_transf[df.columns]



class PowerTransformer(BaseEstimator, TransformerMixin):
    """
    Power transforms supplied columns of a dataframe.
    Returns all columns of dataframe, including those which were not transformed.
    Columns whose skewness is below supplied (percent) threshold are not transformed.
    
    -----Parameters---------------------------------------------------------------------------------------------------
    columns_to_transform : (tuple of strings, None)
                            If None, all columns will be considered for transformation, otherwise only the provided
                            columns
    tol                  : (non-negative  float)
                            Skewness threshold. A column is transformed only if its skewness is above tol
    how                  : ('log', 'boxcox', 'boxcox1p', 'yj')
                            'log' for for log transform (x -> log(1+x)),
                            'boxcox' for Box-Cox transformation of x,
                            'boxcox1p' for Box-Cox transformation of 1+x,
                            'yj' for Yeo-Johnson transformation of x
    ignore_binary        : (boolean)
                            If True, binary columns are not power transformed
    standardize          : (boolean)
                            Whether to standardize to zero mean unit variance after transforming, or not

    -----Issues--------------------------------------------------------------------------------------------------------
    1. Won't work if any to-be-transformed columns have null values
    """
    
    
    def __init__(self, columns_to_transform=None, how='yj', tol=1.0, ignore_binary=True, standardize=False):
        self.columns_to_transform = columns_to_transform
        self.how = how
        self.tol = tol
        self.ignore_binary = ignore_binary
        self.standardize = standardize
    

    def fit(self, df):
        df = df.copy()
        
        # columns which will be considered for transformation 
        self.search_from = np.array(self.columns_to_transform)
        if self.columns_to_transform is None:
            # all numeric columns
            self.search_from = np.array([col for col in df.columns if df[col].dtype!='O'])

        # identify columns with high skewness
        self.to_transform = [] # will hold highly skewed columns
        for col in self.search_from:
            # ignore (or not) binary variables
            if self.ignore_binary:
                n_unique_vals = len(df[col].unique())
                if n_unique_vals > 2:
                    skewness = stats.skew(df[col])
                    if skewness > self.tol:
                        self.to_transform.append(col)
            else:
                skewness = stats.skew(df[col])
                if skewness > self.tol:
                    self.to_transform.append(col)

        return self
    
    
    def transform(self, df):
        df = df.copy()
        
        for col in self.to_transform:
            if self.how=='log':
                df[col] = np.log(1+df[col])
            elif self.how=='yj':
                df[col] = skl_preproc.power_transform(df[col].values.reshape(-1, 1),
                                                      method='yeo-johnson',
                                                      standardize=self.standardize)
            elif self.how=='boxcox':
                df[col] = skl_preproc.power_transform(df[col].values.reshape(-1, 1),
                                                      method='box-cox',
                                                      standardize=self.standardize)
            elif self.how=='boxcox1p':
                df[col] = skl_preproc.power_transform(1 + df[col].values.reshape(-1, 1),
                                                      method='box-cox',
                                                      standardize=self.standardize)

        return df