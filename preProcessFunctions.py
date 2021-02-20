# -*- coding: utf-8 -*-
"""
Created on Feb 20 2021
@author: gari.ciodaro.guerra
Auxiliar functions to explore DataFrames.
"""
import pandas as pd

from scipy import stats
from pyod.models.iforest import IForest
from sklearn.preprocessing import OneHotEncoder

def print_unique_categorical_values(df,cols):
    """ prints unique categorical values
    ----------
    df :  pandas.dataframe.
    cols : list(String)
        categorical cols to ana
    """
    for each_col in cols:
        
        unique_values=df[each_col].unique()
        unique_values_len=len(unique_values)
        print(" {} has {} total values".format(each_col, unique_values_len))
        if unique_values_len<10:
            print("---values are: ",unique_values)

def check_nulls(df,mode='show'):
    """ print total register with null dataframe
    Parameters
    ----------
    df :  pandas.dataframe.
    """
    null_count=pd.DataFrame(df.isna().sum(),columns=['total nulls'])
    if mode=='show':
        print(null_count)
    else:
        return null_count

def remove_outlier_zscore(df,cols,threshold=2):
    """Detect and remove outliers based
    on the zscore. This assummes that your Random Variable
    populations is normally distributed.
    Zscore=X-mean/Std
    Parameters
    ----------
    df : pandas.dataframe
    cols : list
    threshold : float
    """
    df_in=df.copy()
    for each_col in cols:
        df_in[each_col+'_zscore']=stats.zscore(df_in[each_col])
        df=df[df_in[each_col+'_zscore']<=threshold]
    return df

def remove_outlier_IsolationForest(df):
    """Remove outliers using isolation forest
    """
    df_out       = df.copy()
    outlier_algo = IForest()
    outlier_algo.fit(df)
    prediction   =pd.DataFrame(
        outlier_algo.predict(df),
        columns=['outlier_pred'],index=df.index)
    df_out=df_out[prediction['outlier_pred']==0]
    return df_out



#def hot_encoder_df()

def hot_enconder_dict_generator(df,cols):
    """Hotencode columns cols of dataframe df.
    returns enconders as dictionary and out dataframe
    """
    out_cols=[each for each in df.columns if each not in cols]
    if len(out_cols)>0:
        df_out=df[out_cols].copy()
    else:
        df_out=df.copy()
    dict_enconders={each+'_encoder':OneHotEncoder(
        sparse=False,
        handle_unknown ='ignore') for each in cols}
    dict_df={each+'_df':df[[each]] for each in cols}
    for each_col in cols:
        enconded_data=dict_enconders.get(
            each_col+'_encoder').fit_transform(
            dict_df.get(each_col+'_df'))
        names_cols_encoded=list(dict_enconders.get(each_col+'_encoder').categories_[0])
        dict_df[each_col+'_df']=pd.DataFrame(enconded_data,columns=names_cols_encoded,index=df.index)
    for key,value in dict_df.items():
        df_out=df_out.join(value)
    return dict_enconders,df_out

