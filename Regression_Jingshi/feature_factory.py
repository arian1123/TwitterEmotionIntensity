import numpy as np

def build_dict_from_corpus(x, min_freq):
	'''	
	build a dictionary from corpus x
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
				new_x[i][dictionary[w]] += 1
		i += 1

	return np.array(new_x)
