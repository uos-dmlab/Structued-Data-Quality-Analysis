import os
import sys
import json
import random

import pickle as pkl
import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings('ignore')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f)


def load_json(path):
    with open(path, 'r') as f:
        obj = json.load(f)
    return obj


def write_pickle(obj, path):
    with open(path, 'wb') as f:
        pkl.dump(obj, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        obj = pkl.load(f)
    return obj


def evaluate_mse(X_imputed, X, mask):
    return ((X_imputed[mask] - X[mask]) ** 2).mean()


def category_mapping(X, cat_vars):
    df = pd.DataFrame(X)
    for col in cat_vars:
        df[col] = pd.Categorical(df[col], categories=df[col].unique()).codes
    res = df.values
    return res


def get_data(data_name):
    
    if data_name == 'titanic':
        data = pd.read_csv('./data/titanic.csv')
        X_origin = data.iloc[:, [1, 2, 4, 5, 6, 7, 9, 11]].dropna(axis=0).values
        
        n_col = X_origin.shape[1]
        cat_vars = [0, 1, 2, 7]
        num_vars = list(set(range(n_col)) - set(cat_vars))
    
    elif data_name == 'breast':
        from sklearn.datasets import load_breast_cancer

        x, y = load_breast_cancer(return_X_y=True)
        X_origin = np.concatenate([x, y.reshape(-1, 1)], axis=1)

        n_col = X_origin.shape[1]
        cat_vars = [30]
        num_vars = list(set(range(n_col)) - set(cat_vars))

    elif data_name == 'adult':
        data = pd.read_csv('./data/adult.csv')
        num_idx = data.dtypes[data.dtypes != 'object'].index
        num_vars = [data.columns.get_loc(idx) for idx in num_idx]
        
        X_origin = data.values
        na_idx = np.array([ any(i=='?') for i in X_origin])
        X_origin = X_origin[~na_idx, :]

        n_col = X_origin.shape[1]
        cat_vars = list(set(range(n_col)) - set(num_vars))

    elif data_name == 'bank':
        data = pd.read_csv('./data/bank.csv')
        num_idx = data.dtypes[data.dtypes != 'object'].index
        num_vars = [data.columns.get_loc(idx) for idx in num_idx]
        
        X_origin = data.values

        n_col = X_origin.shape[1]
        cat_vars = list(set(range(n_col)) - set(num_vars))
    
    elif data_name == 'crime':
        data = pd.read_csv('./data/crime.csv', error_bad_lines=False)
        data.dropna(axis=0, inplace=True)
        num_vars = [16, 17, 20, 21]
        cat_vars = [5, 6, 9, 10, 12, 13, 14, 15]
        vars_lst = cat_vars + num_vars

        data.iloc[:, num_vars] = data.iloc[:, num_vars].astype('float64')
        data.iloc[:, cat_vars] = data.iloc[:, cat_vars].astype('object')
        data = data.iloc[:, vars_lst]

        num_idx = data.dtypes[data.dtypes != 'object'].index
        num_vars = [data.columns.get_loc(idx) for idx in num_idx]
        
        X_origin = data.values

        n_col = X_origin.shape[1]
        cat_vars = list(set(range(n_col)) - set(num_vars))

    else:
        raise ValueError('Choose titanic/breast/adult/bank')
    return X_origin, n_col, num_vars, cat_vars


def ex_classify(data_name, train_array, test_array, num_vars, cat_vars):
    
    label_dict = {
            'titanic': 0,
            'breast': 30,
            'adult': 14,
            'bank': 16,
            'crime': 2
            }

    label_idx = label_dict[data_name]

    n_tr = train_array.shape[0]
    n_te = test_array.shape[0]

    all_features = np.concatenate([train_array, test_array], axis=0)
    
    le = LabelEncoder()
    labels = le.fit_transform(all_features[:, label_idx])

    y_train = labels[:n_tr]
    y_test = labels[n_tr:]

    cat_vars = list(set(cat_vars) - {label_idx})
    X = list()
    X.append(StandardScaler().fit_transform(
        all_features[:, num_vars].astype('float')))
    if cat_vars:
        for idx in cat_vars:
            enc = OneHotEncoder()
            X.append(enc.fit_transform(all_features[:, idx].reshape(-1, 1)).toarray())
    X = np.concatenate(X, axis=1)

    X_train = X[:n_tr, :]
    X_test = X[n_tr:, :]

    clf = LogisticRegression(random_state=0).fit(X_train, y_train)
    score = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    
    return score


def evaluate_mse(X_imputed, X, mask):
    return ((X_imputed[mask] - X[mask]) ** 2).mean()


def generate_missing_mask(X, percent_missing=10, missingness='MCAR'):
    if missingness=='MCAR':
        # missing completely at random
        mask = np.random.rand(*X.shape) < percent_missing / 100.
    elif missingness=='MAR':
        # missing at random, missingness is conditioned on a random other column
        # this case could contain MNAR cases, when the percentile in the other column is 
        # computed including values that are to be masked
        mask = np.zeros(X.shape)
        n_values_to_discard = int((percent_missing / 100) * X.shape[0])
        # for each affected column
        for col_affected in range(X.shape[1]):
            # select a random other column for missingness to depend on
            depends_on_col = np.random.choice([c for c in range(X.shape[1]) if c != col_affected])
            # pick a random percentile of values in other column
            if n_values_to_discard < X.shape[0]:
                discard_lower_start = np.random.randint(0, X.shape[0]-n_values_to_discard)
            else:
                discard_lower_start = 0
            discard_idx = range(discard_lower_start, discard_lower_start + n_values_to_discard)
            values_to_discard = X[:,depends_on_col].argsort()[discard_idx]
            mask[values_to_discard, col_affected] = 1
    elif missingness == 'MNAR':
        # missing not at random, missingness of one column depends on unobserved values in this column
        mask = np.zeros(X.shape)
        n_values_to_discard = int((percent_missing / 100) * X.shape[0])
        # for each affected column
        for col_affected in range(X.shape[1]):
            # pick a random percentile of values in other column
            if n_values_to_discard < X.shape[0]:
                discard_lower_start = np.random.randint(0, X.shape[0]-n_values_to_discard)
            else:
                discard_lower_start = 0
            discard_idx = range(discard_lower_start, discard_lower_start + n_values_to_discard)
            values_to_discard = X[:,col_affected].argsort()[discard_idx]
            mask[values_to_discard, col_affected] = 1
    return mask > 0


def cal_metric_numpy(X_res, X, missing_mask, num_vars, cat_vars):
    task_result = {}
    num_result = (X_res[:, num_vars] - X[:, num_vars]).astype('float')
    mse = np.power(num_result, 2).sum() / missing_mask[:, num_vars].sum()
    nmse = mse / np.var(X[:, num_vars])
    nrmse = nmse ** 0.5
    task_result['nrmse'] = round(nrmse, 4)
    tot_error = 0.
    tot_num = 0.
    for idx in cat_vars:
        mask = missing_mask[:, idx]
        cat_result = X_res[:, idx] != X[:, idx]
        error_rate = cat_result[mask].sum() / mask.sum()
        tot_error += cat_result[mask].sum()
        tot_num += mask.sum()
        task_result['col_{}_error_rate'.format(idx)] = round(error_rate, 4)
    task_result['total_error_rate'] = round(tot_error / tot_num, 4)
    return task_result
