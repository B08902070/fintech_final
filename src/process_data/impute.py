import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer, KNNImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import ExtraTreesRegressor
import xgboost


from process_data.data_config import FeatureType, CONFIG_MAP, DataSource



"""Impute missing data: Numerical"""

def fill_zero(col, ref_data):
    ref_data[col].fillna(0, inplace=True)
    return ref_data[col]

def fill_mean(col, ref_data):
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    ref_data.values[:] = imputer.fit_transform(ref_data)
    return ref_data[col]

def fill_median(col, ref_data):
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    ref_data.values[:] = imputer.fit_transform(ref_data)
    return ref_data[col]

def bayesian(col, ref_data):
    imputer = IterativeImputer(estimator=BayesianRidge())
    ref_data.values[:] = imputer.fit_transform(ref_data)

    return ref_data[col]

# may have error
def extra_tree(col, ref_data):
    imputer = IterativeImputer(estimator=ExtraTreesRegressor(n_estimators=50, random_state=0, min_samples_split=1))
    ref_data.values[:] = imputer.fit_transform(ref_data)

    return ref_data[col]

def xgboost_imp(col, ref_data):
    imputer = IterativeImputer(
        estimator=xgboost.XGBRegressor(
            n_estimators=5,
            random_state=1,
            tree_method='gpu_hist',
        ),
        missing_values=np.nan,
        max_iter=5,
        initial_strategy='mean',
        imputation_order='ascending',
        verbose=2,
        random_state=1
    )
    ref_data.values[:] = imputer.fit_transform(ref_data)

    return ref_data[col]



"""Impute  missing data : Categorical"""
def fill_null(col, ref_data):
    ref_data[col].fillna('NULL', inplace=True)
    return ref_data[col]

def most_freq(col, ref_data):
    imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    ref_data.values[:] = imputer.fit_transform(ref_data)
    return ref_data[col]

# run loo long
def knnc(col, ref_data):
    imputer = KNNImputer(n_neighbors=5, weights='uniform')
    ref_data.values[:] = imputer.fit_transform(ref_data)

    return ref_data[col]



IMPUTE_NUMERICAL={
    'fill_zero': fill_zero,
    'fill_mean': fill_mean,
    'fill_median': fill_median,
    'bayesian': bayesian,
    'extra_tree': extra_tree,
    'xgboost': xgboost_imp
}
IMPUTE_CATEGORICAL={
    'fill_null': fill_null,
    'most_freq': most_freq,
    'knnc': knnc
}

def impute(impute_num, impute_cat, data, data_source):
    if data.isna().sum().sum  == 0:
        return data

    if data_source == DataSource.DP:
        data['debit_credit'].values[:] = pd.Categorical(data['debit_credit']).codes

    impute_num_fn = IMPUTE_NUMERICAL[impute_num]
    impute_cat_fn = IMPUTE_CATEGORICAL[impute_cat]
    config = CONFIG_MAP[data_source]
    cols = data.columns
    num_list, cat_list=[], []
    """make refernce data"""
    for col in cols:
        feature_type = getattr(config, col)
        if feature_type in [FeatureType.NUMERICAL, FeatureType.DATE]:
            num_list += [data[col].copy()]
        elif col != 'sar_flag' and feature_type == FeatureType.CATEGORICAL:
            cat_list += [data[col].copy()]

    num_data, cat_data = pd.DataFrame(num_list).T, pd.DataFrame(cat_list).T
    ref_data = num_data.join(cat_data)

    """Impute data"""
    for col in cols:
        feature_type = getattr(config, col)
        if col == 'sar_flag' or data[col].isna().sum() == 0:
            continue

        if feature_type == FeatureType.NUMERICAL:
            data[col].values[:] = impute_num_fn(col, ref_data).astype('int32').values[:]
        elif feature_type == FeatureType.CATEGORICAL:
            data[col].astype(str)
            data[col].values[:] = impute_cat_fn(col, ref_data).values[:]
        
    return data
