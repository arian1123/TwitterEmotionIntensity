#pre processing and feature extraction for training files

import os
import re
import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import scipy as sp

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
            data['hashtags'] = self.extract_hashtags(tokens[1])

            self.tweet_list.append(data)

    #
    #
    #end read_file
    #
    #


    #extract hashtags from tweet
    def extract_hashtags(self, tweet_body):

        hashtags = re.findall(r"#(\w+)", tweet_body)
        return hashtags

    #end extract hashtags


    #use DepecheMood to get a 0-1 intensity of hashtagged term
    def hashtag_intensity(self, term):

        print('blah')


    #end hashtag_intensity


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

        #delete hashtag symbol
        x = re.sub(r"#", "", x)

        return x

    #end regular tweet


#generate variety of features
class TweetFeatureGenerator:

    def __init__(self, TweetParser, emotion='angry', bag_of_words=True,
                 glove=True, hashtag_intensity=True, truncate_dict=True):

        self.tweet_data = TweetParser #TweetProcessor Object, generated features from parsed data
        self.truncate_dict = truncate_dict
        self.depechemood_dict = {} #populate later, if need be
        self.emotion = emotion
        self.placehold_emotion_intensity = 0.5 #use value to fill missing/incomplete value for intensity

        #get all individual features
        if(bag_of_words == True):
            self.vocab, self.bag_words_model = self.build_bag_words()  # input

        if(hashtag_intensity == True):
            self.hashtag_intensity_model = self.build_hashtag_intensity()

        #self.features_vector = np.column_stack((self.bag_words_model, self.hashtag_intensity_model))

    #end init

    #
    #
    # Functions for Bag of Words Model
    #
    #
    #

    #generate bag of words model (for use in feature vector)
    #Param:
    #bool truncate, remove words from dictionary under word frequency threshold
    def build_bag_words(self):

        vectorizer = CountVectorizer()
        corpus = self.tweet_data.tweet_list_dataframe['content'].tolist()
        bag_of_words = vectorizer.fit_transform(corpus).toarray()
        dictionary = vectorizer.get_feature_names()

        if(self.truncate_dict == True):
            dictionary, bag_of_words = self.truncate_dictionary(bag_of_words, dictionary, min_freq=5)

        return dictionary, bag_of_words

    #end build_bag_words


    #delete words from dictionary from bag of words and dictionary under frequency threshold
    # THINK OF WAYS TO NUANCE WORD DELETION
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

    #
    #
    # Functions for Hashtag intensity
    #
    #
    #

    #create DepecheMood sentiment dictionary as python data structure

    ######## add word tag to dictionary
    def get_depechemood_dict(self, type='freq'):

        path = "Arian/DepecheMood/DepecheMood_" + type + ".txt"

        file = open(path)
        dict = {}
        emotions = [t.replace("\n", "").lower() for t in next(file).split("\t")[1:]]

        for f in file:

            tokens = f.split("\t")
            word = tokens[0].split("#", 1)[0]
            raw_intensities = [float(i.replace("\n", "")) for i in tokens[1:]] #convert str to float
            intensities = [float(i)/max(raw_intensities) for i in raw_intensities] #normalize intensity values from 0-1

            dict[word] = {e: intensities[idx] for idx, e in enumerate(emotions)}

        return dict

    #end get_depechemood_dict

    #generate hashtag intensity model (for use in feature vector)
    def build_hashtag_intensity(self):

        self.depechemood_dict = self.get_depechemood_dict()
        hashtags = self.tweet_data.tweet_list_dataframe['hashtags'].tolist()
        model = np.array([self.hashtag_intensity(h, emotion=self.emotion) for h in hashtags])

        return model

    #end build_hashtag_intensity

    #get intensity of hashtag words from depechemood dictionary
    # return average of all hashtags for single tweet for passed emotion
    def hashtag_intensity(self, hashtags, emotion='angry'):

        depeche_emotion = self.map_emotion_depechemood(emotion)

        if not hashtags:
            avg_intensity = self.placehold_emotion_intensity
        else:
            avg_intensity = np.average([self.get_depechemood_score(i, depeche_emotion) for i in hashtags])

        return avg_intensity

    #end hashtag_intnsities

    #return matching emotion for depechemood dictionary
    def map_emotion_depechemood(self, emotion):

        if(emotion == 'anger'): return 'angry'
        if(emotion == 'fear'): return 'afraid'
        if(emotion == 'joy'): return 'happy'
        if(emotion == 'sadness'): return 'sad'
        else: return False
    #end map_emotion_depechemood

    #get a depechemood intensity for word + target emotion
    #LEMMATIZE WORDS
    def get_depechemood_score(self, word="", emotion="angry"):

        #format input properly
        w = word.lower()
        e = emotion.lower()

        #if word is not found in dict, return 0 /// THINK OF ALTERNATIVE METHOD HANDLING
        if(w in self.depechemood_dict):
            score = self.depechemood_dict[w][e]
        else:
            score = self.placehold_emotion_intensity

        print(str(w) + " " + str(score))

        return score

    #end get_depechemood_score

    #
    #
    # Emoji extraction and evaluation
    #
    #

#End class TweetFeatureGenerator