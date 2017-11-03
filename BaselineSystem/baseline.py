#! usr/bin/python
# coding=utf-8

import re
import os, sys
import numpy as np
from sklearn.metrics import recall_score, confusion_matrix

path = './data'

def define_emoji():
	'''
	This is a pre-filtering procedure to find emoji.
	Manual deletion for characters from other languages (such as Aracbian, Russian) and illegal forms are needed.
	After that, we can get emoji.txt
	'''
	emoji = []
	for d in os.listdir(path):
		if d == '.DS_Store': 
			continue
	
		for f in os.listdir(os.path.join(path, d)):
			ukn_text = [re.sub('[a-zA-Z0-9\s+\.\!\?\/_,$%^*()+\[\]\"\'`\\\]+|[|+——！~@#￥%……&*:;-=-£]', ' ', t.strip()) 
		                for t in open(os.path.join(path, d, f)).readlines()]
			#print(len(ukn_text))
			for t in ukn_text:
				if t != '':
					re.sub(' +',' ',t)
					re.sub('', ' ', t)
					for _t in t.split():
						#if len(_t) == 4 and _t not in emoji:
						for i in range(len(_t)):
							if _t[i] not in emoji:
								emoji.append(_t[i])
								
#						if _t not in emoji:
#							print(_t)
#							emoji.append(_t)

	with open('test.txt', 'w') as f:
		for e in emoji:
			f.write(e+'\n')
	f.close()
#	for e in emoji:
#		print(e)
	return emoji
     
			

def feed_to_embedding():
	'''
	extract all tweets into a txt file for training embeddings
	'''
	map_emoji = dict()
	prefix = 'emoji'
	with open('emoji.txt') as f:
		emoji = [l.strip() for l in f.readlines()]
		
	for i, e in enumerate(emoji):
		map_emoji.update({e:prefix+str(i)})
		
	with open('./embedding/test.txt', 'w') as _f:
		for d in os.listdir(path):
			# for each file
			for f in os.listdir(os.path.join(path, d)):
				text = [t.strip().split('\t')[1] 
			            for t in open(os.path.join(path, d, f)).readlines()]  # get tweets
				# for each tweets, replace emoji with unique text
				for t in text:
					for e in emoji:
						if e in t:
							t = t.replace(e, map_emoji[e])
					_f.write(t+'\n')
				
def build_dict_from_corpus(x, min_freq):
	'''	
	build a dictionary from corpus x
	from most frequent to least frequent
	'''	
	dictionary = {}
	for _x in x:
		for _w in _x.split():
			if _w not in dictionary:
				dictionary.update({_w: 1})
			else:
				dictionary[_w] += 1

	# sort the dictionary based on each word's frequency
	filter_dict = sorted(dictionary.items(), key=lambda d:d[1], reverse = True)
	# filter out some words with low frequency
	filter_dict = [d for d in filter_dict if d[1] >= min_freq]

	dictionary = {}
	i = 0
	for d in filter_dict:
		dictionary.update({d[0]:i})
		i += 1

	return dictionary

def lexicon_feature(x, dictionary):
	'''
	0-1 coding of a sentence
	'''
	new_x = np.zeros([len(x), len(dictionary)])
	i = 0
	for _x in x:
		for w in _x.split():
			if w in dictionary:
				new_x[i][dictionary[w]] = 1
		i += 1

	return np.array(new_x)

def measure_clf(y, z):
	print ('accuracy = ', np.mean(y == z))	
	print ('micro-recall = ', recall_score(y, z, average='micro'))	
	print ('macro-recall = ', recall_score(y, z, average='macro'))
	print ('confusion matrix = ', confusion_matrix(y, z))
	
def measure_reg(y, z):
	return np.corrcoef(y, z)[0,1]
	
def load_reg(path='./data/EI-reg-English-Train', emotion='sadness'):
	map_emoji = dict()
	prefix = 'emoji'
	with open('emoji.txt') as f:
		emoji = [l.strip() for l in f.readlines()]
		
	for i, e in enumerate(emoji):
		map_emoji.update({e:prefix+str(i)})
		#print(e + ':' + prefix + str(i))
		
	for f in os.listdir(path):
		if f.find(emotion) >= 0:
			text = [l.split('\t')[1:]
			        for l in open(os.path.join(path, f)).readlines()]
			break
	
	x, y = [t[0] for t in text], [float(t[2]) for t in text]
	#print(y[0])
	for i in range(len(x)):
		for e in emoji:
			if e in x[i]:
				x[i] = x[i].replace(e, map_emoji[e])
				#print(x[i])
	
	return x, y
	
if __name__ == '__main__':
	#define_emoji()		
	feed_to_embedding()
	#load_reg()
	#feed_to_embedding()
	#train_x, train_y = load_reg()
	#dev_x, dev_y = load_reg(path='./data/2018-EI-reg-En-dev')
	#cdict = build_dict_from_corpus(train_x, min_freq=3)
	#train_x = lexicon_feature(train_x, cdict)
	#print(train_x[200])