#! usr/bin/python
# coding=utf-8

from utils import *
from preprocess import *
from feature_factory import *
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
#from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
import numpy as np


SVM = SVR()
XGboost = GradientBoostingRegressor(n_estimators=200)
MLP = MLPRegressor(hidden_layer_sizes=[100, 50], activation='logistic')

svm_coef, boost_coef, mlp_coef = [], [], []

for _emotion in ['anger', 'fear', 'joy', 'sadness']:
	print ('')
	print ('Emotion:', _emotion)
	train_x, train_y = load_original_reg(emotion=_emotion)
	dev_x, dev_y = load_original_reg(path='./data/2018-EI-reg-En-dev', emotion=_emotion)
	for i in range(len(train_x)):
		train_x[i] = regular_tweet(train_x[i])
	for i in range(len(dev_x)):
		dev_x[i] = regular_tweet(dev_x[i])



	cdict = build_dict_from_corpus(train_x, min_freq=5)
	train_x = lexicon_feature(train_x, cdict)
	dev_x = lexicon_feature(dev_x, cdict)

	print ('training data has', len(train_x), 'samples')
	print ('SVM regressor')
	SVM.fit(train_x, train_y)
	output = measure_reg(dev_y, SVM.predict(dev_x))
	#print('                    test on dev: ', 'Pearson correlation:', output[0], ';Spearman correlation: ', output[1])
	
	kf = KFold(n_splits=10, random_state=2, shuffle=True)
	folds = kf.split(train_x, train_y)
	kfold = 0
	Pearson_correlation = []
	Spearman_correlation = []
	for idx1, idx2 in folds:
		#print (idx1, idx2)
		#training data
		training_input = [train_x[i] for i in idx1]
		training_output = [train_y[i] for i in idx1]
		#test data
		test_input = [train_x[i] for i in idx2]
		test_output = [train_y[i] for i in idx2]

		kfold += 1
		SVM.fit(training_input, training_output)
		output = measure_reg(test_output, SVM.predict(test_input))
		Pearson_correlation.append(output[0])
		Spearman_correlation.append(output[1])
	#print('10-fold cross validation average: Pearson correlation:', np.average(Pearson_correlation), ';Spearman correlation: ',np.average(Spearman_correlation))
	svm_coef.append(Pearson_correlation)	
	
	print ('XGBoost regressor')
	XGboost.fit(train_x, train_y)
	output = measure_reg(dev_y, XGboost.predict(dev_x))
	#print('                    test on dev: ', 'Pearson correlation:', output[0], ';Spearman correlation: ', output[1])

	
	kf = KFold(n_splits=10, random_state=2 , shuffle=True)
	folds = kf.split(train_x, train_y)
	kfold = 0
	Pearson_correlation = []
	Spearman_correlation = []
	for idx1, idx2 in folds:
		#print (idx1, idx2)
		#training data
		training_input = [train_x[i] for i in idx1]
		training_output = [train_y[i] for i in idx1]
		#test data
		test_input = [train_x[i] for i in idx2]
		test_output = [train_y[i] for i in idx2]

		kfold += 1
		XGboost.fit(training_input, training_output)
		output = measure_reg(test_output, XGboost.predict(test_input))
		Pearson_correlation.append(output[0])
		Spearman_correlation.append(output[1])
	#print('10-fold cross validation average: Pearson correlation:', np.average(Pearson_correlation), ';Spearman correlation: ',np.average(Spearman_correlation))
	boost_coef.append(Pearson_correlation)
	
	print ('MLP regressor')
	MLP.fit(train_x, train_y)
	output = measure_reg(dev_y, MLP.predict(dev_x))
	#print('                    test on dev: ', 'Pearson correlation:', output[0], ';Spearman correlation: ', output[1])

	
	kf = KFold(n_splits=10, random_state=2, shuffle=True)
	folds = kf.split(train_x, train_y)
	kfold = 0
	Pearson_correlation = []
	Spearman_correlation = []
	for idx1, idx2 in folds:
		#print (idx1, idx2)
		#training data
		training_input = [train_x[i] for i in idx1]
		training_output = [train_y[i] for i in idx1]
		#test data
		test_input = [train_x[i] for i in idx2]
		test_output = [train_y[i] for i in idx2]

		kfold += 1
		MLP.fit(training_input, training_output)
		output = measure_reg(test_output, MLP.predict(test_input))
		Pearson_correlation.append(output[0])
		Spearman_correlation.append(output[1])
	#print('10-fold cross validation average: Pearson correlation:', np.average(Pearson_correlation), ';Spearman correlation: ',np.average(Spearman_correlation))
	mlp_coef.append(Pearson_correlation)

print (' '*10,'   SVM  ', '    XGBoost', '  MLP')
for i, _emotion in enumerate(['anger', 'fear', 'joy', 'sadness']):
    print('%10s%10.4f%10.4f%10.4f'%(_emotion, np.mean(svm_coef[i]), np.mean(boost_coef[i]), np.mean(mlp_coef[i])))
 