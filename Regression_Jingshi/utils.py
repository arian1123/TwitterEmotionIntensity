import numpy as np
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix
import pickle

import scipy.stats

def measure_clf(y, z):
	#print ('accuracy = ', np.mean(y == z))
	print ('accuracy = ', accuracy_score(y, z))
	print ('micro-recall = ', recall_score(y, z, average='micro'))	
	print ('macro-recall = ', recall_score(y, z, average='macro'))

	accuracy = np.mean(y == z)
	micro_recall = recall_score(y, z, average='micro')
	macro_recall = recall_score(y, z, average='macro')
	from sklearn.metrics import confusion_matrix
	cm = confusion_matrix(y, z)
	print('confusion matrix: ', '\n',  cm)
	return accuracy, micro_recall, macro_recall, cm

def measure_reg(y, z):
	
	#return np.corrcoef(y, z)[0,1]
	pears_corr = scipy.stats.pearsonr(y, z)[0]
	spear_corr = scipy.stats.spearmanr(y, z)[0]
	print_output = "Pearson correlation: " + str(pears_corr) + "; Spearman correlation: " + str(spear_corr)
	return pears_corr, spear_corr
