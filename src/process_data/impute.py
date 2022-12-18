import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer, KNNImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import ExtraTreesRegressor
import xgboost


from process_data.data_config import FeatureType,CONFIG_MAP



"""Impute missing data: Numerical"""

def fill_zero(data):
    data.fillna(0, inplace=True)
    return data

def fill_mean(data):
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    ret = imputer.fit_transform(data)
    return ret

def fill_median(data):
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    ret = imputer.fit_transform(data)
    return ret

def bayesian(data):
    imputer = IterativeImputer(estimator=BayesianRidge(), random_state=0)
    ret = imputer.fit_transform(data)
    return ret

def extra_tree(data):
    imputer = IterativeImputer(estimator=ExtraTreesRegressor(n_estimators=10, random_state=0), random_state=0)
    ret = imputer.fit_transform(data)
    return ret

def xgboost_imp(data):
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

    ret = imputer.fit_transform(data)
    return ret



"""Impute  missing data : Categorical"""
def fill_null(data):
    data.fillna('NULL', inplace=True)
    return data

def most_freq(data):
    imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    ret = imputer.fit_transform(data)
    return ret

def knnc(data):
    imputer = KNNImputer(n_neighbors=10, weights='uniform')
    ret = imputer.fit_transform(data)
    return ret





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
    'knnc':knnc
}

def impute(impute_num, impute_cat, datas):
    impute_num_fn = IMPUTE_NUMERICAL[impute_num]
    impute_cat_fn = IMPUTE_CATEGORICAL[impute_cat]

    for i in range(len(datas)):
        data, data_source = datas[i]
        config = CONFIG_MAP[data_source]
        cols = data.columns
        num_list, cat_list, others=[], [], []
        sar_flag_col=None
        for col in cols:
            feature_type = getattr(config, col)
            if col == 'sar_flag':
                sar_flag_col = data[col].copy()
            elif feature_type == FeatureType.NUMERICAL:
                num_list += [data[col].copy()]
            elif feature_type == FeatureType.CATEGORICAL:
                cat_list += [data[col].copy()]
            else:
                others += [data[col].copy()]
        if not num_data.empty:
            num_data = impute_num_fn(pd.DataFrame(num_list).T)
        if not cat_data.empty
            cat_data = impute_cat_fn(pd.DataFrame(cat_list).T)
        others = pd.DataFrame(others).T
        new_data = others.join(num_data)
        new_data = new_data.join(cat_data)
        if sar_flag_col is not None:
            new_data = new_data.join(sar_flag_col)
        datas[i][0] = new_data
        

    for d in datas:
        print(d[0].columns)
    return datas


