#! usr/bin/python
# coding=utf-8

from utils import *
from preprocess import *
from feature_factory import *
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor

#reg = GradientBoostingRegressor(n_estimators=200)
reg = MLPRegressor(hidden_layer_sizes=[100, 50], activation='logistic')

for _emotion in ['anger', 'fear', 'joy', 'sadness']:
	train_x, train_y = load_reg(emotion=_emotion)
	dev_x, dev_y = load_reg(path='./data/2018-EI-reg-En-dev', emotion=_emotion)
	
	cdict = build_dict_from_corpus(train_x, min_freq=5)
	train_x = lexicon_feature(train_x, cdict)
	dev_x = lexicon_feature(dev_x, cdict)
	
	reg.fit(train_x, train_y)
	z = reg.predict(dev_x)
	
	print (measure_reg(dev_y, z))

# XGBoost
#0.372739715258
#0.403489378863
#0.464128258254
#0.436785900773


# MLP
#0.292609947712
#0.392031740267
#0.457844795001
#0.435065708575

