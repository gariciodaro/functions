import matplotlib.pyplot as plt 
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import StratifiedKFold
import numpy as np


def plot_auc_cv(X,y,clf,n_splits=5):
    stratified_cv=StratifiedKFold(n_splits=n_splits)
    # True positve rate sequence
    tprs=[]
    # Auc sequence
    aucs=[]
    
    #false posive rate axis
    x_axis=np.linspace(0,1,100)

    figure = plt.figure(figsize=(10,10))
    ax =  figure.add_subplot(1,1,1)

    for i,(train_split_index,test_split_index) in enumerate(stratified_cv.split(X,y)):
        clf.fit(X.iloc[train_split_index],
                y.iloc[train_split_index].to_numpy().ravel())
        viz =  plot_roc_curve(clf,
                             X.iloc[test_split_index],
                             y.iloc[test_split_index].to_numpy().ravel(),
                             name='ROC Fold {}'.format(i),
                             alpha=0.3, lw=1, ax=ax)
        interpolate_line=np.interp(x_axis,viz.fpr, viz.tpr)
        interpolate_line[0]=0.0
        tprs.append(interpolate_line)
        aucs.append(viz.roc_auc)
    ax.plot([0,1],[0,1],linestyle='--', 
                        lw=2,
                        color='r',
                        label='unskilled',
                        alpha=0.8)
    mean_tpr=np.mean(tprs,axis=0)
    x_axis[-1]=1.0

    #compute area under de curve of mean curve
    mean_auc = auc(x_axis,mean_tpr)
    std_auc = np.std(aucs)

    ax.plot(x_axis, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
        lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(x_axis, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title="Receiver operating characteristic")
    ax.legend(loc="lower right")
    plt.show()
