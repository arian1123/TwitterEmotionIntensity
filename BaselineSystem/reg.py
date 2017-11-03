#! usr/bin/python
# coding=utf-8

from baseline import *
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor

train_x, train_y = load_reg()
dev_x, dev_y = load_reg(path='./data/2018-EI-reg-En-dev')

cdict = build_dict_from_corpus(train_x, min_freq=3)
train_x = lexicon_feature(train_x, cdict)
dev_x = lexicon_feature(dev_x, cdict)

reg = SVR()
reg.fit(train_x, train_y)
z = reg.predict(dev_x)

print (measure_reg(dev_y, z))

reg = GradientBoostingRegressor()
reg.fit(train_x, train_y)
z = reg.predict(dev_x)

print (measure_reg(dev_y, z))
