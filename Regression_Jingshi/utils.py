import numpy as np
from sklearn.metrics import recall_score, confusion_matrix

def measure_clf(y, z):
	print ('accuracy = ', np.mean(y == z))	
	print ('micro-recall = ', recall_score(y, z, average='micro'))	
	print ('macro-recall = ', recall_score(y, z, average='macro'))
	print ('confusion matrix = ', confusion_matrix(y, z))

def measure_reg(y, z):
	return np.corrcoef(y, z)[0,1]

