#! usr/bin/python
# coding=utf-8

from utils import *
from preprocess import *
from feature_factory import *
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

SVM = SVR()
XGboost = GradientBoostingRegressor(n_estimators=200)
MLP = MLPRegressor(hidden_layer_sizes=[100, 50], activation='logistic')

for _emotion in ['anger', 'fear', 'joy', 'sadness']:
	print ('')
	print ('Emotion:', _emotion)
	train_x, train_y = load_reg(emotion=_emotion)
	dev_x, dev_y = load_original_reg(path='./data/2018-EI-reg-En-dev', emotion=_emotion)

	cdict = build_dict_from_corpus(train_x, min_freq=5)
	train_x = lexicon_feature(train_x, cdict)
	dev_x = lexicon_feature(dev_x, cdict)

	print ('training data has', len(train_x), 'samples')
	print ('SVM regressor')
	kf = KFold(n_splits=10, random_state=2, shuffle=True)
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
		SVM.fit(training_input, training_output)
		print ('** ', kfold, 'fold: ', measure_reg(test_output, SVM.predict(test_input)), '** ', 'test on dev: ', measure_reg(dev_y, SVM.predict(dev_x)))
		
	
	print ('XGBoost regressor')
	
	kf = KFold(n_splits=10, random_state=2 , shuffle=True)
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
		print ('** ', kfold, 'fold: ', measure_reg(test_output, XGboost.predict(test_input)), '** ', 'test on dev: ', measure_reg(dev_y, XGboost.predict(dev_x)))

	
	print ('MLP regressor')
	kf = KFold(n_splits=10, random_state=2, shuffle=True)
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
		print ('** ', kfold, 'fold: ', measure_reg(test_output, MLP.predict(test_input)), '** ', 'test on dev: ', measure_reg(dev_y, MLP.predict(dev_x)))
