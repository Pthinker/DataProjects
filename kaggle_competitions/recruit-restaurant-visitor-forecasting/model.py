import glob, re
import numpy as np
import pandas as pd
from datetime import datetime
import os
from sklearn import preprocessing
import xgboost as xgb


data_folder = "./data/"


def load_csv(f_name, dtype=None, parse_dates=[]):
    return pd.read_csv(os.path.join(data_folder, f_name), dtype=dtype, parse_dates=parse_dates)


air_reserve = load_csv("air_reserve.csv", 
        {'air_store_id': str, 'visit_datetime': str, 'reserve_datetime': str, 'reserve_visitors': int}, 
        ['visit_datetime', 'reserve_datetime'])

hpg_reserve = load_csv('hpg_reserve.csv', 
        {'hpg_store_id': str, 'visit_datetime': str, 'reserve_datetime': str, 'reserve_visitors': int},
        ['visit_datetime', 'reserve_datetime'])

air_store = load_csv('air_store_info.csv', 
        {'air_store_id': str, 'air_genre_name': str, 'air_area_name': str, 'latitude': float, 'longitude': float})

hpg_store_info = load_csv('hpg_store_info.csv', 
        {'hpg_store_id': str, 'hpg_genre_name': str, 'hpg_area_name': str, 'latitude': float, 'longitude': float})

store_id_relation = load_csv('store_id_relation.csv', 
        {'hpg_store_id': str, 'air_store_id': str})

air_visit = load_csv('air_visit_data.csv', 
        {'air_store_id': str, 'visit_date': str, 'visitors': int},
        ['visit_date'])

date_info = load_csv('date_info.csv', 
        {'calendar_date': str, 'day_of_week': str, 'holiday_flg': bool},
        ['calendar_date'])

submission = load_csv("sample_submission.csv")


air_reserve["visit_date"] = air_reserve['visit_datetime'].dt.date
air_reserve['reserve_visit_diff_days'] = air_reserve.apply(lambda r: (r['visit_datetime'] - r['reserve_datetime']).days, axis=1)
air_reserve_gp1 = air_reserve.groupby(['air_store_id', 'visit_date'], as_index=False)['reserve_visitors'].sum()
air_reserve_gp2 = air_reserve.groupby(['air_store_id', 'visit_date'], as_index=False)['reserve_visit_diff_days'].mean()
air_reserve_gp = pd.merge(air_reserve_gp1, air_reserve_gp2, how='inner', on=['air_store_id', 'visit_date'])

air_visit['visit_week'] = air_visit['visit_date'].dt.dayofweek
air_visit['visit_month'] = air_visit['visit_date'].dt.month
air_visit['visit_year'] = air_visit['visit_date'].dt.year
air_visit['visit_date'] = air_visit['visit_date'].dt.date
air_visit_reserve = pd.merge(air_visit ,air_reserve_gp, how='left', on=['air_store_id', 'visit_date'])

lbl = preprocessing.LabelEncoder()
air_store['air_genre_name'] = lbl.fit_transform(air_store['air_genre_name'])
air_store['air_area_name'] = lbl.fit_transform(air_store['air_area_name'])

day_mapping={"Monday":1, "Tuesday":2, "Wednesday":3, "Thursday":4, "Friday":5, "Saturday":6, "Sunday":0}
date_info['dow'] = date_info['day_of_week'].map(day_mapping)
date_info['calendar_date'] = date_info['calendar_date'].dt.date

air_visit_reserve_store = pd.merge(air_visit_reserve, air_store, how='left', on=['air_store_id'])
air_visit_reserve_store = pd.merge(air_visit_reserve_store, date_info, left_on='visit_date', right_on='calendar_date')

air_visit_reserve_store_gp = air_visit_reserve_store.groupby(['air_store_id', 'dow'], as_index=False)['reserve_visitors', 'reserve_visit_diff_days'].mean().rename(
        columns={'reserve_visitors': 'mean_reserve_visitors', 'reserve_visit_diff_days': 'mean_reserve_visit_diff_days'})

air_visit_reserve_store = pd.merge(air_visit_reserve_store, air_visit_reserve_store_gp, on=['air_store_id', 'dow'], how='left')
air_visit_reserve_store['reserve_visitors'] = air_visit_reserve_store['reserve_visitors'].fillna(air_visit_reserve_store['mean_reserve_visitors'])
air_visit_reserve_store['reserve_visit_diff_days'] = air_visit_reserve_store['reserve_visit_diff_days'].fillna(air_visit_reserve_store['mean_reserve_visit_diff_days'])
air_visit_reserve_store['reserve_visitors'] = air_visit_reserve_store['reserve_visitors'].fillna(0)
air_visit_reserve_store['reserve_visit_diff_days'] = air_visit_reserve_store['reserve_visit_diff_days'].fillna(0)

train = air_visit_reserve_store
train['air_store_id2'] = lbl.fit_transform(train['air_store_id'])
train['date_int'] = train['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
X_train = train.drop(['calendar_date', 'day_of_week', 'visitors', 'visit_date', 'air_store_id', 'mean_reserve_visitors', 'mean_reserve_visit_diff_days'], axis=1)
train_columns = X_train.columns
Y_train = train['visitors'].values


submission['air_store_id'] = submission.id.map(lambda x: '_'.join(x.split('_')[:-1]))
submission['visit_date'] = submission.id.map(lambda x: x.split('_')[-1])
submission['visit_date'] = pd.to_datetime(submission['visit_date'])
submission['visit_week'] = submission['visit_date'].dt.dayofweek
submission['visit_month'] = submission['visit_date'].dt.month
submission['visit_year'] = submission['visit_date'].dt.year
submission['visit_date'] = submission['visit_date'].dt.date

test = pd.merge(submission, date_info, how='left', left_on='visit_date', right_on='calendar_date')
test = pd.merge(test, air_store, how='left', on=['air_store_id'])
test = pd.merge(test, air_reserve_gp, how='left', on=['air_store_id', 'visit_date'])
test_gp = pd.merge(test, air_visit_reserve_store_gp, on=['air_store_id', 'dow'], how='left')
test_gp['reserve_visitors'] = test_gp['reserve_visitors'].fillna(air_visit_reserve_store['mean_reserve_visitors'])
test_gp['reserve_visit_diff_days'] = test_gp['reserve_visit_diff_days'].fillna(air_visit_reserve_store['mean_reserve_visit_diff_days'])
test_gp['reserve_visitors'] = test_gp['reserve_visitors'].fillna(0)
test_gp['reserve_visit_diff_days']=test_gp['reserve_visit_diff_days'].fillna(0)

test_gp['air_store_id2'] = lbl.fit_transform(test_gp['air_store_id'])
test_gp['date_int'] = test_gp['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)

X_test = test_gp[train_columns]

split = 200000
x_train, y_train, x_valid, y_valid = X_train[:split], Y_train[:split], X_train[split:], Y_train[split:]
d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)
d_test = xgb.DMatrix(X_test)
params = {}
params['eta'] = 0.02
params['objective'] = 'reg:linear'
#params['eval_metric'] = 'mae' #mean absolute error
params['max_depth'] = 4
params['silent'] = 1

watchlist = [(d_train, 'train'), (d_valid, 'valid')]
clf = xgb.train(params, d_train, 2000, watchlist, early_stopping_rounds=100, verbose_eval=10)
p_test = clf.predict(d_test)
print(p_test)

output = pd.read_csv('data/sample_submission.csv')
output['visitors'] = p_test
output.loc[output.visitors<0, "visitors"] = 0
output.to_csv('submission/submission.csv', index=False, float_format='%.4f')


