# -*- coding: utf-8 -*-
"""
Created on Oct 28 2020
@author: gari.ciodaro.guerra
@company: sharcx
Auxiliar plotting functions.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

def plot_pie_box_categorical_continuos(
                        df,
                        col,
                        y,
                        colors_list,
                        labels=None,
                        explode_list=None,
                        loc='upper left',
                        title=None,
                        startangle=90,
                        skip_one_cat=True):
    """ plot categorical feature vs continous one.
    Using pie(x col value counts) and box plot (x axis category)

    Parameters
    ----------
        df :  pandas.dataframe
        col : string
            categorical column
        y : string
            continous column
        colors_list : list
        labels : list
        explode_list: list
        loc : string.
            possible values. upper left, upper right etc.
        title : string
        startangle : int
        skip_one_cat : boolean
            skip first category for box plot
    """
    fig = plt.figure(figsize=(15,7))
    ax1=fig.add_subplot(1,2,1)
    ax2=fig.add_subplot(1,2,2)

    if explode_list:
        pass
    else:
        explode_list = [0 for each in colors_list]

    df[col].value_counts().plot(kind='pie',
                                ax=ax1,
                                autopct='%1.1f%%',
                                startangle=startangle,
                                shadow=True,
                                labels=None, 
                                colors=colors_list,
                                explode=explode_list)
    if labels:
        ax1.legend(labels=labels, loc=loc)

    if skip_one_cat:
        df_box=df[df[y]>skip_one_cat]
        colors_list=colors_list[1:]
    else:
        df_box=df
    sns.boxplot(x=col,
                y=y,
                data=df_box,
                ax=ax2,
                palette=colors_list)
    ax2.grid(True)
    if title:
        plt.suptitle(title)

def plot_scatter_box(df, x, y, 
                    palette=None, 
                    target=None, xlim=None, ylim=None, height=13):
    """ Scartter plot of columns x vs y. Also adds box plots to each axis

    Parameters
    ----------
        df : pandas.dataframe
        x  : string
            Column of df to the x axis.
        y  : string
            Column of df to the y axis.
        target : string
            Optional hue. Extra category
        palette :  List
            Optional colors to be used.
        xlim : float
        ylim : float
        height : float
            size of the plot in inches.
    """

    g = sns.JointGrid(data=df, x=x,
                  y=y, 
                  hue=target,height=height,xlim=xlim, ylim=ylim)
    g.plot(sns.scatterplot, sns.boxplot,palette=palette)

def plot_pie_bar_of_category(df,col,title=None):
    """ plot categorical distributions of col
    as pie and bar.

    Parameters
    ----------
        df : pandas.dataframe
        col : string
    """
    fig = plt.figure(figsize=(15,7))
    ax1=fig.add_subplot(1,2,1)
    ax2=fig.add_subplot(1,2,2)
    df[col].value_counts().plot(kind='pie',
                                ax=ax1,
                                autopct='%1.1f%%')
    df[col].value_counts().plot(kind='bar',
                                ax=ax2)
    plt.suptitle(title, y=1, fontsize=15) 
    plt.show()


