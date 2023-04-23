import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from utils.configs import get_training_config

def plot_confusion_matrix(output, y_true, save_path):
    classes = ['Negativo','Positivo']
    
    y_pred = np.round(output)

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix, index = [i for i in classes],
                        columns = [i for i in classes])
    plt.figure(figsize = (6,4))
    sns.heatmap(df_cm, annot=True, fmt='d')

    plt.savefig(save_path)
    
    
def plot_curves(output, y_true, save_path):
    '''
    Plots Receiver Operating Characteristic and Precision-Recall curves
    '''
    fpr, tpr, _ = metrics.roc_curve(y_true, output)
    roc_auc = metrics.auc(fpr, tpr)
    precision, recall, _ = metrics.precision_recall_curve(y_true, output)
    pr_auc = metrics.auc(recall, precision)
        
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,4))
    
    ax1.title.set_text('Receiver Operating Characteristic')
    ax1.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    ax1.legend(loc = 'lower right')
    ax1.plot([0, 1], [0, 1],'r--')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_xlabel('False Positive Rate')
    
    ax2.title.set_text('Precision vs Recall')
    ax2.plot(recall, precision, label = 'AUC = %0.2f' % pr_auc)
    ax2.legend(loc = 'lower left')
    ax2.set_ylabel('Precision')
    ax2.set_xlabel('Recall')
    
    plt.savefig(save_path)