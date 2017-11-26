#! usr/bin/python
# coding=utf-8

import re
import os, sys
import nltk
from nltk.sentiment.util import mark_negation
import numpy as np

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

def def_regular_emoji():
	'''
	use a unique symbol emoji_#No. to replace an emoji
	'''
	map_emoji = dict()
	prefix = ' emoji'  # extra space ensures independence
	with open('emoji.txt') as f:
		emoji = [l.strip() for l in f.readlines()]

	for i, e in enumerate(emoji):
		map_emoji.update({e:prefix+str(i)+' '})   # extra space ensures independence	

	return map_emoji
	
def feed_to_embedding():
	'''
	extract all tweets into a txt file for training embeddings
	'''
	map_emoji = def_regular_emoji()
	emoji = map_emoji.keys()
		
	with open('./embedding/test.txt', 'w') as _f:
		for d in os.listdir(path):
			if d == '.DS_Store': 
				continue
			# notice that we process the files that are regularized before
			for f in os.listdir(os.path.join(path, d)):
				if f.find('_re_') >= 0:
					print ('processing', os.path.join(path, d, f))
					text = [t.strip().split('\t')[1] 
				            for t in open(os.path.join(path, d, f)).readlines()]  # get tweets
					# for each tweets, replace emoji with unique text
					for t in text:
						for e in emoji:
							if e in t:
								t = t.replace(e, map_emoji[e])
						_f.write(t+'\n')
	
	# check correctness. For example, joint emoji (like emoji5emoji5) is wrong
	uni_emoji = set()
	with open('./embedding/test.txt') as _f:
		for l in _f.readlines():
			l = l.strip().split()
			tmp = [_l for _l in l if _l.find('emoji') >= 0]
			uni_emoji = uni_emoji.union(set(tmp))
	
	print (uni_emoji)

def load_reg(path='./data/EI-reg-English-Train', emotion='sadness'):
	map_emoji = def_regular_emoji()
	emoji = map_emoji.keys()
	
	for f in os.listdir(path):
		if f.find(emotion) >= 0 and f.find('_re_') >= 0:
			text = [l.split('\t')[1:]
			        for l in open(os.path.join(path, f)).readlines()]
			break
	
	x, y = [t[0].split('#')[0] for t in text], [float(t[2]) for t in text]
	return x, y

def load_original_reg(path='./data/EI-reg-English-Train', emotion='sadness'):
	map_emoji = def_regular_emoji()
	emoji = map_emoji.keys()
	
	for f in os.listdir(path):
		if f.find(emotion) >= 0 and f.find('_re_') < 0:
			text = [l.split('\t')[1:]
			        for l in open(os.path.join(path, f)).readlines()]
			break
	
	x, y = [t[0].split('#')[0] for t in text], [float(t[2]) for t in text]
	return x, y

def load_original_clf(path='./data/EI-oc-En-Train', emotion='sadness'):
    map_emoji = def_regular_emoji()
    emoji = map_emoji.keys()

    for f in os.listdir(path):
        if f.find(emotion) >= 0:
            text = [l.split('\t')[1:]
                    for l in open(os.path.join(path, f)).readlines()]
            break

    x = [t[0].split('#')[0].lower() for t in text]
    y = [int(t[2].split(':')[0]) for t in text]
    return x, y

def regular_tweet(x):
	'''
	to regular a single tweet
	'''
	map_emoji = def_regular_emoji()
	emoji = map_emoji.keys()	
	
	# 1. note that '#' leads tags
	#x = x.split('#')[0]
	
	# 2. regular emoji
	for e in emoji:
		if e in x:
			x = x.replace(e, map_emoji[e])
	
	# 3. filter out
	filter_table = ['\\n', '/n',
	                '@[a-zA-Z0-9]+']
	for f in filter_table:
		x = re.sub(f, ' ', x)
		
	# 4. regular special words
	x = re.sub(r"(\d+)kgs ", lambda m: m.group(1) + ' kg ', x)        # e.g. 4kgs => 4 kg
	x = re.sub(r"(\d+)kg ", lambda m: m.group(1) + ' kg ', x)         # e.g. 4kg => 4 kg
	x = re.sub(r"(\d+)k ", lambda m: m.group(1) + '000 ', x)          # e.g. 4k => 4000
	x = re.sub(r"\$(\d+)", lambda m: m.group(1) + ' dollar ', x)
	x = re.sub(r"(\d+)\$", lambda m: m.group(1) + ' dollar ', x)

	# acronym
	x = re.sub(r"can\'t", "can not", x)
	x = re.sub(r"can’t", "can not", x)
	x = re.sub(r"cannot", "can not ", x)
	x = re.sub(r"what\'s", "what is", x)
	x = re.sub(r"What’s", "what is", x)
	x = re.sub(r"\'ve ", " have ", x)
	x = re.sub(r"’ve ", " have ", x)
	x = re.sub(r"n\'t", " not ", x)
	x = re.sub(r"n’t", " not ", x)
	x = re.sub(r"i\'m", "i am ", x)
	x = re.sub(r"i’m", "i am ", x)
	x = re.sub(r"I\'m", "i am ", x)
	x = re.sub(r"I’m", "i am ", x)
	x = re.sub(r"\'re", " are ", x)
	x = re.sub(r"’re", " are ", x)
	x = re.sub(r"\'d", " would ", x)
	x = re.sub(r"’d", " would ", x)
	x = re.sub(r"\'ll", " will ", x)
	x = re.sub(r"’ll", " will ", x)
	x = re.sub(r"yrs", " years ", x)
	x = re.sub(r"c\+\+", "cplusplus", x)
	x = re.sub(r"c \+\+", "cplusplus", x)
	x = re.sub(r"c \+ \+", "cplusplus", x)
	x = re.sub(r"c#", "csharp", x)
	x = re.sub(r"f#", "fsharp", x)
	x = re.sub(r"g#", "gsharp", x)
	x = re.sub(r" e mail ", " email ", x)
	x = re.sub(r" e \- mail ", " email ", x)
	x = re.sub(r" e\-mail ", " email ", x)
	x = re.sub(r",000", '000', x)
	x = re.sub(r"\'s", " ", x)
	x = re.sub(r"’s", " ", x)

	# spelling correction
	x = re.sub(r"ph\.d", "phd", x)
	x = re.sub(r"PhD", "phd", x)
	x = re.sub(r"fu\*k", "fuck", x)
	x = re.sub(r"f\*ck", "fuck", x)
	x = re.sub(r"f\*\*k", "fuck", x)
	x = re.sub(r"wtf", "what the fuck", x)
	x = re.sub(r"Wtf", "what the fuck", x)
	x = re.sub(r"WTF", "what the fuck", x)
	x = re.sub(r"pokemons", "pokemon", x)
	x = re.sub(r"pokémon", "pokemon", x)
	x = re.sub(r"pokemon go ", "pokemon-go ", x)
	x = re.sub(r" e g ", " eg ", x)
	x = re.sub(r" b g ", " bg ", x)
	x = re.sub(r" 9 11 ", " 911 ", x)
	x = re.sub(r" j k ", " jk ", x)
	x = re.sub(r" fb ", " facebook ", x)
	x = re.sub(r"facebooks", " facebook ", x)
	x = re.sub(r"facebooking", " facebook ", x)
	x = re.sub(r"insidefacebook", "inside facebook", x)
	x = re.sub(r"donald trump", "trump", x)
	x = re.sub(r"the big bang", "big-bang", x)
	x = re.sub(r"the european union", "eu", x)
	x = re.sub(r" usa ", " america ", x)
	x = re.sub(r" us ", " america ", x)
	x = re.sub(r" u s ", " america ", x)
	x = re.sub(r" U\.S\. ", " america ", x)
	x = re.sub(r" US ", " america ", x)
	x = re.sub(r" American ", " america ", x)
	x = re.sub(r" America ", " america ", x)
	x = re.sub(r" quaro ", " quora ", x)
	x = re.sub(r" mbp ", " macbook-pro ", x)
	x = re.sub(r" mac ", " macbook ", x)
	x = re.sub(r"macbook pro", "macbook-pro", x)
	x = re.sub(r"macbook-pros", "macbook-pro", x)
	x = re.sub(r" 1 ", " one ", x)
	x = re.sub(r" 2 ", " two ", x)
	x = re.sub(r" 3 ", " three ", x)
	x = re.sub(r" 4 ", " four ", x)
	x = re.sub(r" 5 ", " five ", x)
	x = re.sub(r" 6 ", " six ", x)
	x = re.sub(r" 7 ", " seven ", x)
	x = re.sub(r" 8 ", " eight ", x)
	x = re.sub(r" 9 ", " nine ", x)
	x = re.sub(r"googling", " google ", x)
	x = re.sub(r"googled", " google ", x)
	x = re.sub(r"googleable", " google ", x)
	x = re.sub(r"googles", " google ", x)
	x = re.sub(r" rs(\d+)", lambda m: ' rs ' + m.group(1), x)
	x = re.sub(r"(\d+)rs", lambda m: ' rs ' + m.group(1), x)
	x = re.sub(r"€", " eu ", x)
	x = re.sub(r"€", " euro ", x)
	x = re.sub(r"£", " pound ", x)
	x = re.sub(r"dollars", " dollar ", x)

	# punctuation
	x = re.sub(r"\*", " * ", x)
	x = re.sub(r"\\n", " ", x)
	x = re.sub(r"\+", " + ", x)
	x = re.sub(r"'", " ", x)
	x = re.sub(r"-", " - ", x)
	x = re.sub(r"/", " / ", x)
	x = re.sub(r"\\", " \ ", x)
	x = re.sub(r"=", " = ", x)
	x = re.sub(r"\^", " ^ ", x)
	x = re.sub(r":", " : ", x)
	x = re.sub(r"\.", " . ", x)
	x = re.sub(r",", " , ", x)
	x = re.sub(r"\?", " ? ", x)
	x = re.sub(r"!", " ! ", x)
	x = re.sub(r"\"", " \" ", x)
	x = re.sub(r"&", " & ", x)
	x = re.sub(r"\|", " | ", x)
	x = re.sub(r";", " ; ", x)
	x = re.sub(r"\(", " ( ", x)
	x = re.sub(r"\)", " ( ", x)

	# punc as prefix of a word should be separated
	x = re.sub(r"(?<=[a-zA-Z\d])_+", " _ ", x)
	x = re.sub(r"(?<=[a-zA-Z\d])-+", " - ", x)
	x = re.sub(r"(?<=[a-zA-Z\d])–+", " - ", x)
	x = re.sub(r"(?<=[a-zA-Z\d])—+", " - ", x)
	x = re.sub(r"(?<=[a-zA-Z\d])―+", " ― ", x)
	x = re.sub(r"(?<=[a-zA-Z\d])“+", " “ ", x)
	x = re.sub(r"(?<=[a-zA-Z\d])”+", " ” ", x)
	x = re.sub(r"(?<=[a-zA-Z\d])‘+", " ‘ ", x)
	x = re.sub(r"(?<=[a-zA-Z\d])’+", " ’ ", x)
	x = re.sub(r"(?<=[a-zA-Z\d])#+", " # ", x)
	x = re.sub(r"(?<=[a-zA-Z\d])…+", " … ", x)
	# punc as postfix of a word should be separated
	x = re.sub(r"_+(?=[a-zA-Z\d])", " _ ", x)
	x = re.sub(r"-+(?=[a-zA-Z\d])", " - ", x)
	x = re.sub(r"–+(?=[a-zA-Z\d])", " - ", x)
	x = re.sub(r"—+(?=[a-zA-Z\d])", " - ", x)
	x = re.sub(r"―+(?=[a-zA-Z\d])", " ― ", x)
	x = re.sub(r"“+(?=[a-zA-Z\d])", " “ ", x)
	x = re.sub(r"”+(?=[a-zA-Z\d])", " ” ", x)
	x = re.sub(r"‘+(?=[a-zA-Z\d])", " ‘ ", x)
	x = re.sub(r"’+(?=[a-zA-Z\d])", " ’ ", x)
	x = re.sub(r"#+(?=[a-zA-Z\d])", " # ", x)
	x = re.sub(r"…+(?=[a-zA-Z\d])", " … ", x)
	# symbol replacement
	x = re.sub(r"&", " and ", x)
	x = re.sub(r"\|", " or ", x)
	x = re.sub(r"=", " equal ", x)
	x = re.sub(r"\+", " plus ", x)
	x = re.sub(r"₹", " rs ", x) 
	x = re.sub(r"\$", " dollar ", x)
	
	# 4. seperate puncuation, because they look like a postfix for the final words
	punc = re.findall('[.!?]+', x)
	for p in punc:
		x = (' '+p).join(x.split(p))

	return x

def regular_file(path):
	in_file = open(path)
	out_file = open(path.split('.txt')[0]+'_re_'+'.txt', 'w')
	
	all_l = []
	for l in in_file.readlines():
		l = l.strip().split('\t')
		l[1] = regular_tweet(l[1])
		all_l.append(l)
	# all_l[0]: number, all_l[1]: tweet, all_l[2]: emotion, all_l[3]: score
	
	# pip install python-levenshtein	
	# import Levenshtein
	
	# delete tweets that are similar, combine their scores to one by averaging, threshold is 0.5
	delete_idx = []
	# for i in range(len(all_l)-1):
	# 	if i not in delete_idx:
	# 		similar_idx, ave_score = [], []
	# 		for j in range(i+1, len(all_l)):
	# 			if float(Levenshtein.distance(all_l[i][1], all_l[j][1])) / float(min(len(all_l[i][1]), len(all_l[j][1]))) < 0.5:
	# 				similar_idx.append(j)
	# 				delete_idx.append(j)
	# 				ave_score = [float(all_l[j][3])]
	# 		if len(similar_idx) > 0:
	# 			all_l[i][3] = (float(all_l[i][3]) + sum(ave_score)) / (1 + len(ave_score))
	# 			all_l[i][3] = str(all_l[i][3])
	
	for i, l in enumerate(all_l):
		if i not in delete_idx:
			l = '\t'.join(l)
			out_file.write(l+'\n')
	
	print ('Finished writing', out_file)


### Create function to break apart contractions to its derivative words
### A text file containing this('contractions.txt') should be located at the
### working directory along with this script.

def break_contractions(text):
    #### Import dictionary of contractions: contractions.txt
    with open('contractions.txt', 'r') as inf:
        contractions = eval(inf.read())

    pattern = re.compile(r'\b(' + '|'.join(contractions.keys()) + r')\b')
    result = pattern.sub(lambda x: contractions[x.group()], text)
    return (result)


### Create function to lemmatize (stem) words to their root
### This requires the NLTK wordnet dataset.

def lemmatize_words(text):
    # Create a lemmatizer object
    wordnet_lemmatizer = nltk.stem.WordNetLemmatizer()
    return (wordnet_lemmatizer.lemmatize(text.lower()))
    # out = []
    # for word in text:
    #     out.append(wordnet_lemmatizer.lemmatize(word.lower()))
    # return (out)


#### Create function to remove stopwords (e.g., and, if, to)
#### Removes stopwords from a list of words (i.e., to be used on lyrics after splitting).
#### This requires the NLTK stopwords dataset.
def remove_stopwords(text):
    # Create set of all stopwords
    stopword_set = set(w.lower() for w in nltk.corpus.stopwords.words())
    out = ''
    for word in text.split(' '):
        # Convert words to lower case alphabetical letters only
        # word = ''.join(w.lower() for w in word if w.isalpha())
        if word not in stopword_set:
            out += word
    # Return only words that are not stopwords
    return (out)



	
if __name__ == '__main__':
	# 1.
	#define_emoji()		

	# 2.
	for _emotion in ['anger', 'fear', 'joy', 'sadness']:
		regular_file('./data/EI-reg-English-Train/EI-reg-en_'+_emotion+'_train.txt')
		regular_file('./data/2018-EI-reg-En-dev/2018-EI-reg-En-'+_emotion+'-dev.txt')
# 		regular_file('./data/EI-oc-En-train/EI-oc-En-'+_emotion+'-train.txt')
# 		regular_file('./data/2018-EI-oc-En-dev/2018-EI-oc-En-'+_emotion+'-dev.txt')
	# 3.
	feed_to_embedding()
#	import tensorflow as tf
#	hello = tf.constant('Hello, TensorFlow!')
#	sess = tf.Session()
#	print(sess.run(hello))

