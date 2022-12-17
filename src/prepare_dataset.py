import argparse 

from sklearn.preprocessing import QuantileTransformer
import numpy as np
from easydict import EasyDict as edict
import pandas as pd
from tqdm import tqdm

from process_data.data_config import (DataSource, FeatureType, CONFIG_MAP)
from process_data.utils import save_pickle
from process_data.impute import impute

TRAIN_DIR = 'train_first'

CCBA_PATH = '../raw_dataset/public_train_x_ccba_full_hashed.csv'
CDTX_PATH = '../raw_dataset/public_train_x_cdtx0001_full_hashed.csv'
CUSTINFO_PATH = '../raw_dataset/public_train_x_custinfo_full_hashed.csv'
DP_PATH = '../raw_dataset/public_train_x_dp_full_hashed.csv'
REMIT_PATH = '../raw_dataset/public_train_x_remit1_full_hashed.csv'
PDATE_PATH = '../raw_dataset/public_x_alert_date.csv'
TDATE_PATH = '../raw_dataset/train_x_alert_date.csv'
ANSWER_PATH = '../raw_dataset/train_y_answer.csv'
SAMPLE_PATH = '../raw_dataset/sample.csv'

def prepare_data(args):
    ccba = pd.read_csv(CCBA_PATH)
    cdtx = pd.read_csv(CDTX_PATH)
    cinfo = pd.read_csv(CUSTINFO_PATH)
    dp = pd.read_csv(DP_PATH)
    remit = pd.read_csv(REMIT_PATH)
    pdate = pd.read_csv(PDATE_PATH)
    tdate = pd.read_csv(TDATE_PATH)
    answer = pd.read_csv(ANSWER_PATH)
    sample = pd.read_csv(SAMPLE_PATH)

    date = pd.concat([pdate, tdate], axis=0)
    cinfo = cinfo.merge(date, on='alert_key', how='left')
    cinfo = cinfo.merge(answer, on='alert_key', how='left')

    def normalize(col):
        qt = QuantileTransformer(
            n_quantiles=10_000, 
            random_state=0, 
            subsample=min(5*10**5, len(col)),
            output_distribution='normal'
        )
        return qt.fit_transform(col)

    def process_numerical(col):
        col = normalize(col)
        return col

    def process_catgorical(col):
        map_dict = {v:i for i, v in enumerate(set(col.unique()))}
        col = col.map(map_dict)
        return col

    datas = [
        (ccba, DataSource.CCBA), 
        (cdtx, DataSource.CDTX),
        (dp, DataSource.DP),
        (remit, DataSource.REMIT),
        (cinfo, DataSource.CUSTINFO),
    ]

    impute(args.impute_num, args.impute_cat, datas)

    # process numerical and categorical and data_source
    for data, data_source in datas:
        config = CONFIG_MAP[data_source]
        cols = data.columns
        numericals = []
        for col in cols:
            feature_type = getattr(config, col)
            if feature_type == FeatureType.NUMERICAL and col != 'sar_flag':
                numericals.append(col)
            elif feature_type == FeatureType.CATEGORICAL:
                data[col] = process_catgorical(data[col].copy())

        if numericals:
            data[numericals] = process_numerical(data[numericals].copy())
        data['data_source'] = data_source


    datas = [d[0] for d in datas]

    datas_g = [d.groupby(by='cust_id') for d in datas]

    def get_date(d):
        ds = d.data_source
        
        if ds == DataSource.CCBA:
            date = d.byymm
        elif ds == DataSource.CDTX:
            date = d.date
        elif ds == DataSource.DP:
            date = d.tx_date
        elif ds == DataSource.REMIT:
            date = d.trans_date
        elif ds == DataSource.CUSTINFO:
            date = d.date
        return date, ds


    cust_ids = cinfo.cust_id.unique()
    ret_data = edict()
    for cust_id in tqdm(cust_ids):
        # get all data from each group
        cust_data = []
        for d in datas_g:
            if not cust_id in d.groups:
                continue
            cust_data += d.get_group(cust_id).to_dict('records')
        for i in range(len(cust_data)):
            cust_data[i] = edict(cust_data[i])
        
        # sort by date
        cust_data.sort(key=get_date)
        
        # generate source list and target_mask
        source_list = []
        train_mask = []
        test_mask = []
        for i, c in enumerate(cust_data):
            ds = c.data_source
            source_list.append(ds)
            if ds != DataSource.CUSTINFO:
                pass
            elif np.isnan(c.sar_flag):
                test_mask.append(i)
            else:
                train_mask.append(i)
        
        # save data
        ret_data[cust_id] = edict({
            'sources': source_list,
            'train_mask': train_mask,
            'test_mask': test_mask,
            'cust_data': cust_data,
        })
	
    save_pickle(ret_data, f'../data/cust_data_{args.impute_num}_{args.impute_cat}.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--impute_num", default='zero_fill', help='impute method for missing numerical data')
    parser.add_argument("--impute_cat", default='zero_null', help='impute method for missing categorical data')

    args = parser.parse_args()

    prepare_data(args)