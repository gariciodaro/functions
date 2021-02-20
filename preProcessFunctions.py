# -*- coding: utf-8 -*-
"""
Created on Feb 20 2021
@author: gari.ciodaro.guerra
Auxiliar functions to explore DataFrames.
"""
import pandas as pd

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
