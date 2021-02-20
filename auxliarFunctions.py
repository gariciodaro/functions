# -*- coding: utf-8 -*-
"""
Created on Feb 2020
@author: gari.ciodaro.guerra
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.feature_selection import f_classif,chi2


def detect_relevant_categories(df,categorical_col_list,target_col,p=0.05):
    """ Apply chi2 statistical test on all categorical_col_list
    against binary target target_col.
    
    Parameters
    ----------
    df :  pandas.dataframe.
    categorical_col_list : list[string]
        Contains columns of the DataFrame to check chi2 dependency against 
        target_col.
    target_col: string
        name of the binary target columns.
    
    Returns
    -------
    relevant_categories : dictionary
        key -> name of feature, values -> statistically significant
        categories.
    p_values_sorted : dictionary
        key -> name of feature, values -> p_values dataframe
    pertange_relevant_cat : dictionary
        key -> name of feature, values -> (pertange_relevant_cat, total_cat)
    """
    
    relevant_categories={}
    p_values_sorted={}
    pertange_relevant_cat={}
    for each_cat in categorical_col_list:
        ## Hot encode categories
        hot_enconder = OneHotEncoder(sparse=False,handle_unknown ='ignore')
        
        # enconded data
        enconded_data=hot_enconder.fit_transform(df[[each_cat]])
        
        ## Use chi2 to determine importance
        # Perform stastistical test of dependency
        F_vales_cat,p_values_cat = chi2(enconded_data,df[target_col])

        # Get dataframe with proper categories in columns and hot encoded.
        categories_encoded=hot_enconder.categories_
        list_enconded,names_cat=flatten_cat_encon(categories_encoded,[each_cat])
        names_df=pd.DataFrame(names_cat,columns=['names'],index=list_enconded)
        
        # Create dataframe of p-values results
        pvalues_cat=pd.DataFrame(p_values_cat.reshape(len(p_values_cat),1),
                                index=list_enconded,
                                columns=['p_value'])
        pvalues_cat=pvalues_cat.join(names_df)
        
        fail_p_values=pvalues_cat[pvalues_cat['p_value']>p]

        per_fail_p_values=len(fail_p_values)/len(pvalues_cat['p_value'])
        sorted_pvalues_df=pvalues_cat.sort_values(by='p_value',ascending=True)
        sorted_pvalues_df['Evidence']=(1-pvalues_cat.p_value)
        relevant_categories[each_cat]=list(
                        sorted_pvalues_df[pvalues_cat['p_value']<p].index)
        p_values_sorted[each_cat]=sorted_pvalues_df
        pertange_relevant_cat[each_cat]=((1-per_fail_p_values),len(sorted_pvalues_df))
    return relevant_categories,p_values_sorted,pertange_relevant_cat


def flatten_cat_encon(cat_enco,cols):
    hold=[]
    names=[]
    for index,each in enumerate(cat_enco):
        for each_in in each:
            hold.append(each_in)
            names.append(cols[index])
    return hold,names