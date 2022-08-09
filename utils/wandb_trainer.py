import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix, log_loss,
matthews_corrcoef, average_precision_score)

def get_metrics(probability, label_list):
    metrics_dictionary = {}
    # predictions = np.argmax(probability.detach().cpu().numpy(), axis=0)
    predictions =  [int(i > .5) for i in probability]

    accuracy = accuracy_score(label_list, predictions)
    precision = precision_score(label_list, predictions, zero_division=0)
    recall = recall_score(label_list, predictions, zero_division=0)
    f1 = f1_score(label_list, predictions, zero_division=0)
    log_loss = metrics.log_loss(label_list, predictions)
    mcc = matthews_corrcoef(label_list, predictions)
    auc = roc_auc_score(label_list, probability)
    
    mAP = average_precision_score(label_list, probability)

    metric_df = pd.DataFrame(np.empty(0)) #, dtype=metric_types
    metric_df.loc[0] = [mAP, auc, accuracy, precision, recall, f1, mcc, log_loss, 0]

    return metric_df

class Wandb_Trainer():
    def __init__(self, wandb_config, class_number=1):
        self