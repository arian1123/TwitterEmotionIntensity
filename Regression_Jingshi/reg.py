#! usr/bin/python
# coding=utf-8

from utils import *
from preprocess import *
from feature_factory import *
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

XGboost = GradientBoostingRegressor(n_estimators=200)
MLP = MLPRegressor(hidden_layer_sizes=[100, 50], activation='logistic')

for _emotion in ['anger', 'fear', 'joy', 'sadness']:
	print ('')
	print ('Emotion:', _emotion)
	train_x, train_y = load_reg(emotion=_emotion)
	dev_x, dev_y = load_reg(path='./data/2018-EI-reg-En-dev', emotion=_emotion)

	cdict = build_dict_from_corpus(train_x, min_freq=5)
	train_x = lexicon_feature(train_x, cdict)
	#dev_x = lexicon_feature(dev_x, cdict)

	print ('training data has', len(train_x), 'samples')
	print ('XGBoost regressor')
#	kfold = 0
#	for idx1, idx2 in StratifiedKFold(n_splits=10).split(train_x, train_y):
#		kfold += 1
#		XGboost.fit(train_x[idx1,:], train_y[idx1])
#		print ('** ', kfold, 'fold: pearson coef = ', measure_reg(train_y[idx2], XGboost.predict(train_x[idx2,:])))
#		print ('** ', 'testing coef = ', measure_reg(dev_y, XGboost.predict(dev_x)))
#
#	print ('MLP regressor')
#	kfold = 0
#	for idx1, idx2 in StratifiedKFold(n_splits=10).split(train_x, train_y):
#		kfold += 1
#		MLP.fit(train_x[idx1,:], train_y[idx1])
#		print ('** ', kfold, 'fold: pearson coef = ', measure_reg(train_y[idx2], MLP.predict(train_x[idx2,:])))
#		print ('** ', 'testing coef = ', measure_reg(dev_y, MLP.predict(dev_x)))
		
	#folds
	
	
	kf = KFold(n_splits=10, random_state=2)
	folds = kf.split(train_x, train_y)
	kfold = 0
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
		print ('** ', kfold, 'fold: ')
		measure_reg(test_output, XGboost.predict(test_input))
	
	print ('MLP regressor')
	kf = KFold(n_splits=10, random_state=2)
	folds = kf.split(train_x, train_y)
	kfold = 0
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
		print ('** ', kfold, 'fold: ')
		measure_reg(test_output, MLP.predict(test_input))