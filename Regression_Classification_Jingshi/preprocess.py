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
		if d != 'EI-reg-En-train':
			continue
	
		for f in os.listdir(os.path.join(path, d)):
			ukn_text = [re.sub('[a-zA-Z0-9\s+\.\!\?\/_,$%^*()+\[\]\"\'`\\\]+|[|+——！~@#￥%……&*:;-=-£]', ' ', t.strip())
		                for t in open(os.path.join(path, d, f)).readlines()]
			for t in ukn_text:
				if t != '':
					re.sub(' +',' ',t)
					re.sub('', ' ', t)
					for _t in t.split():
						for i in range(len(_t)):
							if _t[i] not in emoji:
								emoji.append(_t[i])


	with open('test.txt', 'w') as f:
		for e in emoji:
			f.write(e+'\n')
	f.close()

	return emoji

def regular_emoji():
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

def emoji_to_lexicon():
	prefix = ' emoji'  # extra space ensures independence
	with open('emoji.txt') as f:
		emoji = [l.strip() for l in f.readlines()]

	with open('emoji_lexicon.txt', 'w') as out_file:
		for i, e in enumerate(emoji):
			out_file.write(prefix+str(i)+'\n')
	out_file.close()


def load_2017_reg(path='./data/2017train', emotion='sadness'):
	map_emoji = regular_emoji()
	emoji = map_emoji.keys()
	
	for f in os.listdir(path):
		if f.find(emotion) >= 0:
			text = [l.split('\t')[1:]
			        for l in open(os.path.join(path, f)).readlines()]
			break
	x, y = [t[0] for t in text], [float(t[2]) for t in text]
	return x, y

def load_2018_reg(path='./data/EI-reg-En-train', emotion='sadness'):
	map_emoji = regular_emoji()
	emoji = map_emoji.keys()
	
	for f in os.listdir(path):
		if f.find(emotion) >= 0 and f.find('_re_') < 0:
			text = [l.split('\t')[1:]
			        for l in open(os.path.join(path, f)).readlines()]
			break
	text = text[1:]
	x, y = [t[0] for t in text], [float(t[2]) for t in text]
	return x, y

def load_2018_oc(path='./data/EI-oc-En-train', emotion='sadness'):
	map_emoji = regular_emoji()
	emoji = map_emoji.keys()

	for f in os.listdir(path):
		if f.find(emotion) >= 0:
			text = [l.split('\t')[1:]
					for l in open(os.path.join(path, f)).readlines()]
			break
	text = text[1:]
	x = [t[0] for t in text]
	y = [int(t[2].split(':')[0]) for t in text]
	return x, y

def regular_tweet(x):
	'''
	to regular a single tweet
	'''
	map_emoji = regular_emoji()
	emoji = map_emoji.keys()	

	# regular emoji
	for e in emoji:
		if e in x:
			x = x.replace(e, map_emoji[e])

	# filter out line inserting symbols and usernames
	filter_table = ['\\n', '@[a-zA-Z0-9]+']
	for f in filter_table:
		x = re.sub(f, ' ', x)

	# break contractions
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

	# spelling correction, special words, and acronym
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
	x = re.sub(r"(\d+)kgs ", lambda m: m.group(1) + ' kg ', x)        # e.g. 4kgs => 4 kg
	x = re.sub(r"(\d+)kg ", lambda m: m.group(1) + ' kg ', x)         # e.g. 4kg => 4 kg
	x = re.sub(r"(\d+)k ", lambda m: m.group(1) + '000 ', x)          # e.g. 4k => 4000

	# seperate punctuations
	x = re.sub(r"\*", " * ", x)
	x = re.sub(r"\\n", " ", x)
	x = re.sub(r"\+", " + ", x)
	x = re.sub(r"'", " ' ", x)
	x = re.sub(r"-", " - ", x)
	x = re.sub(r"/", " / ", x)
	x = re.sub(r"\\", " \ ", x)
	x = re.sub(r"=", " = ", x)
	x = re.sub(r"\^", " ^ ", x)
	x = re.sub(r":", " : ", x)
	x = re.sub(r",", " , ", x)
	x = re.sub(r"\?", " ? ", x)
	x = re.sub(r"!", " ! ", x)
	x = re.sub(r"\"", " \" ", x)
	x = re.sub(r"&", " & ", x)
	x = re.sub(r"\|", " | ", x)
	x = re.sub(r";", " ; ", x)
	x = re.sub(r"\(", " ( ", x)
	x = re.sub(r"\)", " ( ", x)
	x = re.sub(r"!", " ! ", x)
	x = re.sub(r",", " , ", x)

	# punc as postfix of a word should be separated
	x = re.sub(r"(?<=[a-zA-Z\d])_+", " _ ", x)
	x = re.sub(r"(?<=[a-zA-Z\d])-+", " - ", x)
	x = re.sub(r"(?<=[a-zA-Z\d])―+", " ― ", x)
	x = re.sub(r"(?<=[a-zA-Z\d])“+", " “ ", x)
	x = re.sub(r"(?<=[a-zA-Z\d])”+", " ” ", x)
	x = re.sub(r"(?<=[a-zA-Z\d])‘+", " ‘ ", x)
	x = re.sub(r"(?<=[a-zA-Z\d])’+", " ’ ", x)
	x = re.sub(r'(?<=[a-zA-Z\d])"+', ' " ', x)
	x = re.sub(r"(?<=[a-zA-Z\d])'+", " ' ", x)
	x = re.sub(r"(?<=[a-zA-Z\d])#+", " # ", x)
	x = re.sub(r"(?<=[a-zA-Z\d])\.{1}", " . ", x)

	# punc as prefix of a word should be separated
	x = re.sub(r"_+(?=[a-zA-Z\d])", " _ ", x)
	x = re.sub(r"-+(?=[a-zA-Z\d])", " - ", x)
	x = re.sub(r"―+(?=[a-zA-Z\d])", " ― ", x)
	x = re.sub(r"“+(?=[a-zA-Z\d])", " “ ", x)
	x = re.sub(r"”+(?=[a-zA-Z\d])", " ” ", x)
	x = re.sub(r"‘+(?=[a-zA-Z\d])", " ‘ ", x)
	x = re.sub(r"’+(?=[a-zA-Z\d])", " ’ ", x)
	x = re.sub(r"'+(?=[a-zA-Z\d])", " ' ", x)
	x = re.sub(r'"+(?=[a-zA-Z\d])', ' " ', x)
	x = re.sub(r"#+(?=[a-zA-Z\d])", " # ", x)
	x = re.sub(r"…+(?=[a-zA-Z\d])", " … ", x)
	x = re.sub(r"\.(?=[a-zA-Z\d])", ". ", x)

	# symbol replacement
	x = re.sub(r"&", " and ", x)
	x = re.sub(r"\|", " or ", x)
	x = re.sub(r"=", " equal ", x)
	x = re.sub(r"\+", " plus ", x)
	x = re.sub(r"₹", " rs ", x) 
	x = re.sub(r"\$", " dollar ", x)

	# delete hashtag symbol
	x = re.sub(r"#", "", x)

	# remove multiple spaces
	x = x.strip()
	while '  ' in x:
		x = x.replace('  ', ' ')

	return x

if __name__ == '__main__':
	# 1.
	define_emoji()

	# 2.
	emoji_to_lexicon()

