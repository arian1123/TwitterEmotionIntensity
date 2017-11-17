#pre processing and feature extraction for training files

import os
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


#Use this class to prepare and parse tweets for feature generation
class TweetParser:

    #pass in directory with training data files
    def __init__(self, filepath):

        self.tweet_list = [] #list of dictionaries to hold tweets data
        self.tweet_list_dataframe = []

        self.process_file(filepath)
        self.tweet_list_dataframe = pd.DataFrame(self.tweet_list)

    #end init

    def process_file(self, path):

        file = open(path)
        for f in file:

            f = f.replace("\n", "")
            tokens = f.split("\t")

            data = {}
            data['id'] = tokens[0]
            data['content'] = self.clean_tweet(tokens[1])
            data['emotion'] = tokens[2]
            data['emotion_intensity'] = tokens[3]

            self.tweet_list.append(data)

    #end read_file

    #from Regression_Jingshi/preprocess.py
    def clean_tweet(self, x):

        # 1. filter out
        filter_table = ['\\n', '/n',
                        '@[a-zA-Z0-9]+']
        for f in filter_table:
            x = re.sub(f, ' ', x)

        # 2. regular special words
        x = re.sub(r"(\d+)kgs ", lambda m: m.group(1) + ' kg ', x)  # e.g. 4kgs => 4 kg
        x = re.sub(r"(\d+)kg ", lambda m: m.group(1) + ' kg ', x)  # e.g. 4kg => 4 kg
        x = re.sub(r"(\d+)k ", lambda m: m.group(1) + '000 ', x)  # e.g. 4k => 4000
        x = re.sub(r"\$(\d+)", lambda m: m.group(1) + ' dollar ', x)
        x = re.sub(r"(\d+)\$", lambda m: m.group(1) + ' dollar ', x)

        # acronym
        x = re.sub(r"can\'t", "can not", x)
        x = re.sub(r"cannot", "can not ", x)
        x = re.sub(r"what\'s", "what is", x)
        x = re.sub(r"What\'s", "what is", x)
        x = re.sub(r"\'ve ", " have ", x)
        x = re.sub(r"n\'t", " not ", x)
        x = re.sub(r"i\'m", "i am ", x)
        x = re.sub(r"I\'m", "i am ", x)
        x = re.sub(r"\'re", " are ", x)
        x = re.sub(r"\'d", " would ", x)
        x = re.sub(r"\'ll", " will ", x)
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
        x = re.sub(r"the european union", " eu ", x)
        x = re.sub(r"dollars", " dollar ", x)

        # punctuation
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

        # symbol replacement
        x = re.sub(r"&", " and ", x)
        x = re.sub(r"\|", " or ", x)
        x = re.sub(r"=", " equal ", x)
        x = re.sub(r"\+", " plus ", x)
        x = re.sub(r"₹", " rs ", x)
        x = re.sub(r"\$", " dollar ", x)

        # 3. seperate puncuation, because they look like a postfix for the final words
        punc = re.findall('[.!?]+', x)
        for p in punc:
            x = (' ' + p).join(x.split(p))

        return x

    #end regular tweet


#generate variety of features
class TweetFeatureGenerator:

    def __init__(self, TweetParser):

        self.tweet_data = TweetParser #TweetProcessor Object, generated features from parsed data

    #end init


    #generate bag of words model
    #Param:
    #bool truncate, remove words from dictionary under word frequency threshold
    def build_bag_words(self, truncate=True):

        vectorizer = CountVectorizer()
        corpus = self.tweet_data.tweet_list_dataframe['content'].tolist()
        bag_of_words = vectorizer.fit_transform(corpus).toarray()
        dictionary = vectorizer.get_feature_names()

        if(truncate == True):
            dictionary, bag_of_words = self.truncate_dictionary(bag_of_words, dictionary, min_freq=5)

        return dictionary, bag_of_words

    #end build_bag_words


    #delete words from dictionary from bag of words and dictionary under frequency threshold
    def truncate_dictionary(self, bag_of_words, dict, min_freq=5):

        new_dict = []
        new_bag_words = [] #list of lists, convert to np.array, transpose

        for idx, word in enumerate(dict):

            if(np.sum(bag_of_words[:,idx]) >= min_freq):

                new_dict.append(word)
                new_bag_words.append(bag_of_words[:,idx])

        new_bag_words = np.array(new_bag_words).T

        return new_dict, new_bag_words

    #end truncate_dictionary