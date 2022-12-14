import pickle

from sklearn.preprocessing import QuantileTransformer
import numpy as np
from easydict import EasyDict as edict
import pandas as pd
from tqdm import tqdm

from process_data.data_config import (DataSource, FeatureType, CCBAConfig, CDTXConfig, DPConfig, REMITConfig, CUSTINFOConfig, CONFIG_MAP)
from process_data.utils import load_yaml, save_yaml, save_pickle, load_pickle
from pandas_profiling import ProfileReport


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


ccba = pd.read_csv(CCBA_PATH)
cdtx = pd.read_csv(CDTX_PATH)
cinfo = pd.read_csv(CUSTINFO_PATH)
dp = pd.read_csv(DP_PATH)
remit = pd.read_csv(REMIT_PATH)
pdate = pd.read_csv(PDATE_PATH)
tdate = pd.read_csv(TDATE_PATH)
answer = pd.read_csv(ANSWER_PATH)
sample = pd.read_csv(SAMPLE_PATH)

names = ['ccba', 'cdtx', 'custinfo', 'dp', 'remit', 'pdate', 'tdate', 'answer', 'sample']
datas = [ccba, cdtx, cinfo, dp, remit, pdate, tdate, answer, sample]
num_files = len(datas)

for i in range(num_files):
    print(f'{names[i]}: {datas[i].shape}')
    profile = ProfileReport(datas[i], minimal=True, title=names[i])
    profile.to_file(f'../raw_dataset/data_report/{names[i]}.html', )
	
date = pd.concat([pdate, tdate], axis=0)
cinfo = cinfo.merge(date, on='alert_key', how='left')
cinfo = cinfo.merge(answer, on='alert_key', how='left')
cinfo

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
    col = np.nan_to_num(col, nan=0)
    return col


def process_catgorical(col):
    col.fillna('NULL', inplace=True)
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

num_cat_dict = {}

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
            num_cat = data[col].nunique()
            if data_source not in num_cat_dict:
                num_cat_dict[data_source] = {}
            num_cat_dict[data_source][col] = num_cat

    if numericals:
        data[numericals] = process_numerical(data[numericals].copy())
    data['data_source'] = data_source
	
save_yaml(num_cat_dict, 'num_cat_dict.yml')

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
save_data = edict()
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
    save_data[cust_id] = edict({
        'sources': source_list,
        'train_mask': train_mask,
        'test_mask': test_mask,
        'cust_data': cust_data,
    })
	
isna = cinfo.sar_flag.isna()
train_num = sum(~isna)
test_num = sum(isna)

train_num2 = 0
test_num2 = 0
for v in save_data.values():
    train_num2 += len(v.train_mask)
    test_num2 += len(v.test_mask)


print(train_num == train_num2)
print(test_num == test_num2)

save_pickle(save_data, 'cust_data.pkl')

# get len of cust_data of save_data
lens = []
for k, v in save_data.items():
    lens.append(len(v.sources))
pd.DataFrame(data=lens, columns=None).describe(percentiles=[.25, .5, .75, .9, .95, .99])

train_mask = []
test_mask = []
for v in save_data.values():
    train_mask += v.train_mask
    test_mask += v.test_mask

display(pd.DataFrame(data=train_mask, columns=['train']).describe(percentiles=np.linspace(0,1,11)))
display(pd.DataFrame(data=test_mask, columns=['test']).describe(percentiles=np.linspace(0,1,11)))

data = load_pickle('./cust_data.pkl')
print(len(data))

sars = set()
for cust_id, v in data.items():
    for idx in v.train_mask:
        if v.cust_data[idx].sar_flag == 1:
            sars.add(cust_id)
            break
len(sars)

num1 = []
for cust_id in sars:
    d = data[cust_id]
    tmp = 0
    for idx in d.train_mask:
        tmp += (d.cust_data[idx].sar_flag == 1)
    num1.append(tmp)
sum(num1)

# num_sar = []
num_len = []
num0 = []
num1 = []
for k, d in data.items():
    tmp0 = 0
    tmp1 = 0
    for idx in d.train_mask:
        tmp0 += d.cust_data[idx].sar_flag == 0
        tmp1 += d.cust_data[idx].sar_flag == 1
    num0.append(tmp0)
    num1.append(tmp1)
    num_len.append(len(d.cust_data))
	
df = pd.DataFrame({'num0': num0, 'num1':num1,'num_len': num_len})

df.describe()

df[df.num1>0].describe()

df[df.num_len == df.num_len.max()]

df[df.num1==0].describe()

mask_ids = []
for k, v in data.items():
    for i, idx in enumerate(v.train_mask):
        if i == 0:
            mask_ids.append(idx)
        else:
            mask_ids.append(idx-v.train_mask[i-1])
pd.DataFrame({'mask_ids': mask_ids}).describe(percentiles=np.arange(.9, 1.01, 0.01))


