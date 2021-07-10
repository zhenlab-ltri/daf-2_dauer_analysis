import json
import csv
import re
import os
import time
import numpy as np
import pandas as pd
from functools import reduce
from numpy.ma import compress
from statsmodels.sandbox.stats.multicomp import fdrcorrection0

def get_job_id():
    today = time.strftime('%Y%m%d%H%M')
    return str(today)

def get_job_dir():
    d = f'./output/{get_job_id()}/'

    os.mkdir(d)
    return d

def open_json(fpath):
    data = None
    with open(fpath) as f:
        data = json.load(f)
    return data


def write_json(fpath, data):
    with open(fpath, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def tuple_key (pre, post):
    return (pre, post)

def merge(dict1, dict2, dict3, dict4):
    new = {**dict1, **dict2, **dict3, **dict4}
    return new

def get_key(pre, post):
    return pre + '$' + post

def remove_space(string):
    return string.replace(' ', '')

def comma_string_to_list(string):
    return np.array(string.split(','))

def make_connection_key(data):
    df = pd.read_csv(data)
    connection = df['Pre Class'].str.cat(df['Post Class'], sep = '$')
    df.insert(0, 'connection', connection)

    return df

def merge2(d1, d2, connections, name1, name2):
    dataset_only_1 = d1[d1['connection'].isin(connections)]
    dataset_only_2 = d2[d2['connection'].isin(connections)]
    df_dataset_only_1 = dataset_only_1.drop(['classification', 'nondauer contact'], axis = 1)
    df_dataset_only_2 = dataset_only_2.drop (['Pre Class', 'Post Class', 'Pre', 'Post' ], axis = 1)
    dfs = [df_dataset_only_1, df_dataset_only_2]
    df_dataset_only = reduce(lambda left, right: pd.merge(left, right,  on = 'connection', suffixes = (f'_{name1}', f'_{name2}')), dfs)

    return df_dataset_only

def merge3 (d1, d2, d3, connections, name1, name2):
    dataset_only_1 = d1[d1['connection'].isin(connections)]
    dataset_only_2 = d2[d2['connection'].isin(connections)]
    dataset_only_3 = d3[d3['connection'].isin(connections)]
    df_dataset_only_1 = dataset_only_1.drop(['classification', 'nondauer contact'], axis = 1)
    df_dataset_only_2 = dataset_only_2.drop(['Pre Class', 'Post Class', 'Pre', 'Post', 'classification', 'nondauer contact'], axis = 1)
    df_dataset_only_3 = dataset_only_3.drop (['Pre Class', 'Post Class', 'Pre', 'Post' ], axis = 1)
    dfs = [df_dataset_only_1, df_dataset_only_2, df_dataset_only_3]
    df_dataset_only = reduce(lambda left, right: pd.merge(left, right,  on = 'connection', suffixes = (f'_{name1}', f'_{name2}')), dfs)

    return df_dataset_only

def fdr_correction(pvalues):

    mask = np.isfinite(pvalues)
    corrected_pvalues = np.empty(pvalues.shape)
    corrected_pvalues.fill(np.nan) 
    corrected_pvalues[mask] = fdrcorrection0(pvalues[mask])[1]

    significance = np.empty(pvalues.shape)
    significance.fill(np.nan)
    significance[mask] = fdrcorrection0(pvalues[mask])[0]

    return corrected_pvalues,significance

def clean_data(data, range):
    filter = np.isfinite(data)
    plot = list(compress(filter, data))



    return plot