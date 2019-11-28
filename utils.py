import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder


cat_features = ['tipodepropiedad','ciudad','provincia']


#Solo anda target encoding, el resto no los probe
def target_encoding(df_norm, mode, enc=None):
    '''enc debe ser None si se llama con mode="train", Target Encoder si se llama con mode="test"'''
    if mode=='train':
        enc = ce.TargetEncoder(cols=cat_features).fit(df_norm.drop('precio', axis=1), df_norm['precio'])
        df_final = enc.transform(df_norm.drop('precio', axis=1), df_norm['precio'])
    
    if mode=='test':
        df_final = enc.transform(df_norm)
        
    return df_final, enc


def label_encoding(df_norm, mode, enc=None):
    if mode=='train':
        encoder = LabelEncoder()
        encoded = df_norm[cat_features].apply(encoder.fit_transform)
        df_2 = df_norm.drop(['tipodepropiedad','provincia','ciudad'], axis=1)
        data_cols = list(df_2.columns)
        baseline_data = df_2[data_cols].join(encoded)
    if mode=='test':
        new_cats = enc.transform(df_norm[cat_features])
        baseline_data = df_norm.drop(['tipodepropiedad','ciudad','provincia'], axis=1).join(new_cats)
        
    return baseline_data, encoder

def one_hot_enc(df_norm):
    one_hot_enc = ce.OneHotEncoder()
    one_hot_encoded = one_hot_enc.fit_transform(df_norm[cat_features])
    data = df_norm.join(one_hot_encoded.add_suffix("_oh"))
    return data, one_hot_enc

def binary_enc(df_norm):
    binary_enc = ce.BinaryEncoder()
    binary_encoded = binary_enc.fit_transform(df_norm[cat_features])
    data = df_norm.join(binary_encoded.add_suffix("_binary"))
    data = data.drop(['tipodepropiedad','provincia','ciudad'], axis=1)
    return data, binary_enc

def count_enc(df_norm, mode, count_enc=None):
    if mode=='train':
        count_enc = ce.CountEncoder()
        count_encoded = count_enc.fit_transform(df_norm[cat_features])
        data = df_norm.join(count_encoded.add_suffix("_count"))
        data = data.drop(['tipodepropiedad','provincia','ciudad'], axis=1)
    if mode=='test':
        new_cats = count_enc.transform(df_norm[cat_features])
        baseline_data = df_norm.drop(['tipodepropiedad','ciudad','provincia'], axis=1).join(new_cats)
    return data, count_enc


def catboost_enc(df_norm):
    catboost_enc = ce.CatBoostEncoder(cols=cat_features)
    catboost_enc.fit(df_norm[cat_features], df_norm['precio'])
    data = df_norm.join(catboost_enc.transform(df_norm[cat_features]).add_suffix('_cb'))
    data = data.drop(['tipodepropiedad','provincia','ciudad'], axis=1)
    return data, catboost_enc