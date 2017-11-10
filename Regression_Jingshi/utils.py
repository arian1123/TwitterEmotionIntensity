import numpy as np
from sklearn.metrics import recall_score, confusion_matrix

import scipy.stats

def measure_clf(y, z):
	print ('accuracy = ', np.mean(y == z))	
	print ('micro-recall = ', recall_score(y, z, average='micro'))	
	print ('macro-recall = ', recall_score(y, z, average='macro'))
	print ('confusion matrix = ', confusion_matrix(y, z))

def measure_reg(y, z):
	
	#return np.corrcoef(y, z)[0,1]
	pears_corr = scipy.stats.pearsonr(y, z)[0]
	spear_corr = scipy.stats.spearmanr(y, z)[0]
	output = "Pearson correlation: " + str(pears_corr) + "; Spearman correlation: " + str(spear_corr)
	return output
