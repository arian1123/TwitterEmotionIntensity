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

SVM = SVR()
XGboost = GradientBoostingRegressor(n_estimators=200)
MLP = MLPRegressor(hidden_layer_sizes=[100, 50], activation='logistic')

svm_coef, boost_coef, mlp_coef = [], [], []
for _emotion in ['anger', 'fear', 'joy', 'sadness']:
    
	print ('')
	print ('Emotion:', _emotion)
	train_x, train_y = load_original_reg(emotion=_emotion)
	for i in range(len(train_x)):
		train_x[i] = regular_tweet(train_x[i])
	#dev_x, dev_y = load_reg(path='./data/2018-EI-reg-En-dev', emotion=_emotion)

	cdict = build_dict_from_corpus(train_x, min_freq=5)
	train_x1 = lexicon_feature(train_x, cdict)
	train_x2 = sum_of_word_embedding(train_x)
	#print(train_x1[0], ' and ', train_x2[0])
	#train_x = train_x2
	train_x = np.concatenate((train_x1, train_x2), axis=1)
	
	#dev_x = np.concatenate((dev_x1, dev_x2), axis=1)
	#print(train_x[0])
	
    
	print ('training data has', len(train_x), 'samples')

	print ('SVM regressor')        
	kf = KFold(n_splits=10, random_state=2, shuffle=True)
	folds = kf.split(train_x, train_y)   
	kfold = 0
	coef = []
	for idx1, idx2 in folds:
		kfold += 1
		#training data	
		training_input = [train_x[i] for i in idx1]
		training_output = [train_y[i] for i in idx1]
		#test data
		test_input = [train_x[i] for i in idx2]
		test_output = [train_y[i] for i in idx2]            
		SVM.fit(training_input, training_output)
		output = measure_reg(test_output, SVM.predict(test_input))
		coef.append(output[0])
		print ('** ', kfold, 'fold: pearson coef = ', coef[-1])      
		#print ('** ', 'testing coef = ', measure_reg(dev_y, XGboost.predict(dev_x)))
	svm_coef.append(coef)
	
	print ('XGBoost regressor')        
	kf = KFold(n_splits=10, random_state=2, shuffle=True)
	folds = kf.split(train_x, train_y)   
	kfold = 0
	coef = []
	for idx1, idx2 in folds:
		kfold += 1
		#training data	
		training_input = [train_x[i] for i in idx1]
		training_output = [train_y[i] for i in idx1]
		#test data
		test_input = [train_x[i] for i in idx2]
		test_output = [train_y[i] for i in idx2]            
		XGboost.fit(training_input, training_output)
		output = measure_reg(test_output, XGboost.predict(test_input))
		coef.append(output[0])
		print ('** ', kfold, 'fold: pearson coef = ', coef[-1])      
		#print ('** ', 'testing coef = ', measure_reg(dev_y, XGboost.predict(dev_x)))
	boost_coef.append(coef)

	print ('MLP regressor')
	kf = KFold(n_splits=10, random_state=2, shuffle=True)
	folds = kf.split(train_x, train_y) 
	kfold = 0
	coef = []
	for idx1, idx2 in folds:
		kfold += 1
		#training data	
		training_input = [train_x[i] for i in idx1]
		training_output = [train_y[i] for i in idx1]
		#test data
		test_input = [train_x[i] for i in idx2]
		test_output = [train_y[i] for i in idx2]		
		MLP.fit(training_input, training_output)
		output = measure_reg(test_output, MLP.predict(test_input))
		coef.append(output[0])
		print ('** ', kfold, 'fold: pearson coef = ', coef[-1])
		#print ('** ', 'testing coef = ', measure_reg(dev_y, MLP.predict(dev_x)))
	mlp_coef.append(coef)

import numpy as np
print ('')
print (' '*10,'   SVM  ', '    XGBoost', '  MLP')
for i, _emotion in enumerate(['anger', 'fear', 'joy', 'sadness']):
	print('%10s%10.4f%10.4f%10.4f'%(_emotion, np.mean(svm_coef[i]), np.mean(boost_coef[i]), np.mean(mlp_coef[i])))