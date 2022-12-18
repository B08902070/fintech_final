import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer, KNNImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import ExtraTreesRegressor
import xgboost


from process_data.data_config import FeatureType, CONFIG_MAP



"""Impute missing data: Numerical"""

def fill_zero(col, target_data, refer_data):
    target_data.fillna(0, inplace=True)
    return target_data

def fill_mean(col, target_data, refer_data):
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    target_data.values[:] = imputer.fit_transform(target_data)
    return target_data

def fill_median(col, target_data, refer_data):
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    target_data.values[:] = imputer.fit_transform(target_data)
    return target_data

def bayesian(col, target_data, refer_data):
    imputer = IterativeImputer(estimator=BayesianRidge(), random_state=0)
    imputer.fit(refer_data)
    target_data.values[:] = imputer.transform(target_data)
    return target_data

def extra_tree(col, target_data, refer_data):
    imputer = IterativeImputer(estimator=ExtraTreesRegressor(n_estimators=10, random_state=0), random_state=0)
    imputer.fit(refer_data)
    target_data.values[:] = imputer.transform(target_data)
    return target_data

def xgboost_imp(col, target_data, refer_data):
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
    imputer.fit(refer_data)
    target_data.values[:] = imputer.transform(target_data)
    return target_data



"""Impute  missing data : Categorical"""
def fill_null(col, target_data, refer_data):
    target_data.fillna('NULL', inplace=True)
    return target_data

def most_freq(col, target_data, refer_data):
    imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    target_data.values[:] = imputer.fit_transform(target_data)
    return target_data

def knnc(col, target_data, refer_data):
    imputer = KNNImputer(n_neighbors=10, weights='uniform')
    imputer.fit(refer_data)
    target_data.values[:] = imputer.transform(target_data)
    return target_data





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

def impute(impute_num, impute_cat, datas):
    impute_num_fn = IMPUTE_NUMERICAL[impute_num]
    impute_cat_fn = IMPUTE_CATEGORICAL[impute_cat]

    for i in range(len(datas)):
        data, data_source = datas[i]
        if data.isna().sum().sum  == 0:
            continue
        config = CONFIG_MAP[data_source]
        cols = data.columns
        num_list, cat_list, others=[], [], []
        sar_flag_col=None
        """make refernce data"""
        for col in cols:
            feature_type = getattr(config, col)
            if col != 'sar_flag' and feature_type in [FeatureType.NUMERICAL, FeatureType.DATE]:
                num_list += [data[col].copy()]
            elif feature_type == FeatureType.CATEGORICAL:
                cat_list += [data[col].copy()]
        num_data, cat_data = pd.DataFrame(num_list).T, pd.DataFrame(cat_list).T
        if not cat_data.empty
            cat_data = pd.get_dummies(cat_data)
        refer_data = num_data.join(cat_data)

        """Impute data"""
        for col in cols:
            feature_type = getattr(config, col)
            if col == 'sar_flag' or data[col].isna().sum() == 0:
                continue

            if feature_type == FeatureType.NUMERICAL:
                datas[i][0][col] = impute_num_fn(col, data[col].copy(), refer_data).astype('int32')
            elif feature_type == FeatureType.CATEGORICAL:
                datas[i][0][col] = impute_cat_fn(col, data[col].copy(), refer_data).astype('category')
        
    return datas
