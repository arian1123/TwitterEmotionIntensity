# coding=utf-8
"""
Created on November 5 2017

@author: Jingshi & Arian
"""
import numpy as np
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix
import pickle

import scipy.stats

def measure_oc(y, z):
    '''
    evaluation metrics for classification
    '''
    accuracy = np.mean(y == z)
    micro_recall = recall_score(y, z, average='micro')
    macro_recall = recall_score(y, z, average='macro')
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y, z)
    pears_corr = scipy.stats.pearsonr(y, z)[0]
    return pears_corr, accuracy, micro_recall, macro_recall, cm

def measure_reg(y, z):
    '''
    evaluation metrics for regression
    '''
    pears_corr = scipy.stats.pearsonr(y, z)[0]
    spear_corr = scipy.stats.spearmanr(y, z)[0]
    print_output = "Pearson correlation: " + str(pears_corr) + "; Spearman correlation: " + str(spear_corr)
    return pears_corr, spear_corr
