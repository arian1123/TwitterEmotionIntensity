# coding=utf-8
"""
Created on November 15 2017

@author: Jingshi & Arian
"""
import re
import os, sys
from copy import deepcopy
import nltk
from nltk.sentiment.util import mark_negation
import numpy as np

#Use this class to prepare and parse tweets for feature generation
class Preprocessor:

    # pass in directory with training data files
    def __init__(self, emotion = 'anger', classify = False, year = 2018, define_emoji = False, emoji_to_lexicon = False):
        '''
        this class contains all the methods for preprocessing as described in stage 1, 
        and the method to read the row training tweets. It has three major functions, extract emojis, mapping 
        emojis to unique strings and save them, and regular tweets. You can specify different values to the 
        parameters to let the Preprocessor do different things. If you want to extract emojis, you can use preprocess.
        Preprocessor(define_emoji = True), it will extract emojis from 2018's training data of regression 
        task for the default emotion, and save them to a file named test.txt. If you want to mapping emojis 
        to unique strings and save them, you can use preprocess.Preprocessor(emoji_to_lexicon = True), 
        it will map and save the mapped emojis to a file named emoji_lexicon.txt for the default emotion. 
        If you want to regular tweets, you can simply use preprocess.Preprocessor(), 
        it will regular tweets for the default emotion.
        '''
        self.year = year
        self.emotion = emotion
        self.classify = classify
        self.path = './data'
        # extract emojis and save them to test.txt
        if define_emoji:
            # the task for 2018 regression dataset
            if self.classify == False and self.year == 2018:
                self.path += '/EI-reg-En-train'
            
            # the task for 2018 classification dataset
            elif self.classify and self.year == 2018:
                self.path += '/EI-oc-En-train'
            
            # the task for 2017 regression dataset
            elif self.year == 2017:
                self.path += '/2017train'
            self.define_emoji()
        
        # map emojis to unique strings 
        elif emoji_to_lexicon:
            self.emoji_to_lexicon()
        
        # regular raw tweets
        else:
            
            # the task for 2018 regression dataset
            if self.classify == False and self.year == 2018:
                self.path += '/EI-reg-En-train'
                self.train_x, self.train_y = self.load_2018_reg(emotion= self.emotion)

                # use joy's training data for sadness
                if self.emotion == 'sadness':
                    train_joy_x, train_joy_y = self.load_2018_reg(emotion='joy')
                    for i in range(len(train_joy_y)):
                        train_joy_y[i] = 1 - train_joy_y[i]
                    self.train_x = self.train_x + train_joy_x
                    self.train_y = self.train_y + train_joy_y

                # hashtags
                self.hashtags = deepcopy(self.train_x)
                for i in range(len(self.train_x)):
                    self.hashtags[i] = self.extract_hashtags(self.train_x[i])

                # preprocessing
                for i in range(len(self.train_x)):
                    self.train_x[i] = self.regular_tweet(self.train_x[i])
            
            # the task for 2018 classification dataset
            elif self.classify and self.year == 2018:
                self.path += '/EI-oc-En-train'
                self.train_x, self.train_y = self.load_2018_oc(emotion= self.emotion)

                # hashtags
                self.hashtags = deepcopy(self.train_x)
                for i in range(len(self.train_x)):
                    self.hashtags[i] = self.extract_hashtags(self.train_x[i])

                # preprocessing
                for i in range(len(self.train_x)):
                    self.train_x[i] = self.regular_tweet(self.train_x[i])
            
            # the task for 2017 regression dataset
            elif self.year == 2017:
                self.path += '/2017train'
                self.train_x, self.train_y = self.load_2017_reg(emotion= self.emotion)

                # hashtags
                self.hashtags = deepcopy(self.train_x)
                for i in range(len(self.train_x)):
                    self.hashtags[i] = self.extract_hashtags(self.train_x[i])

                # preprocessing
                for i in range(len(self.train_x)):
                    self.train_x[i] = self.regular_tweet(self.train_x[i])

    # end init

    def extract_hashtags(self, tweet_body):
        '''
        extract hashtags from tweet
        '''

        hashtags = re.findall(r"#(\w+)", tweet_body)
        return hashtags


    def define_emoji(self):
        '''
        This is a pre-filtering procedure to find emoji.
        Manual deletion for characters from other languages (such as Aracbian, Russian) and illegal forms are needed.
        After that, we can get emoji.txt
        '''
        emoji = []
        # for d in os.listdir(path):
        # 	if d != 'EI-reg-En-train':
        # 		continue

        for f in os.listdir(self.path):
            ukn_text = [re.sub('[a-zA-Z0-9\s+\.\!\?\/_,$%^*()+\[\]\"\'`\\\]+|[|+——！~@#￥%……&*:;-=-£]', ' ', t.strip())
                        for t in open(os.path.join(self.path, f)).readlines()]
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


    def regular_emoji(self):
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


    def emoji_to_lexicon(self):
        '''
        this method maps each emoji to a unique string, 
        and save the unique strings to a file named emoji_lexicon.txt.
        for example, a smile face emoji may be mapped to 'emoji12'.
        '''
        prefix = ' emoji'  # extra space ensures independence
        with open('emoji.txt') as f:
            emoji = [l.strip() for l in f.readlines()]

        with open('emoji_lexicon.txt', 'w') as out_file:
            for i, e in enumerate(emoji):
                out_file.write(prefix+str(i)+'\n')
        out_file.close()


    def load_2017_reg(self, path='./data/2017train', emotion='sadness'):
        '''
        this method read the row training data from 2017 regression task
        '''
        for f in os.listdir(path):
            if f.find(emotion) >= 0:
                text = [l.split('\t')[1:]
                        for l in open(os.path.join(path, f)).readlines()]
                break
        x, y = [t[0] for t in text], [float(t[2]) for t in text]
        return x, y


    def load_2018_reg(self, path='./data/EI-reg-En-train', emotion='sadness'):
        '''
        this method read the row training data from 2018 regression task
        '''
        for f in os.listdir(path):
            if f.find(emotion) >= 0 and f.find('_re_') < 0:
                text = [l.split('\t')[1:]
                        for l in open(os.path.join(path, f)).readlines()]
                break
        text = text[1:]
        x, y = [t[0] for t in text], [float(t[2]) for t in text]
        return x, y


    def load_2018_oc(self, path='./data/EI-oc-En-train', emotion='sadness'):
        '''
        this method read the row training data from 2018 classification task
        '''
        for f in os.listdir(path):
            if f.find(emotion) >= 0:
                text = [l.split('\t')[1:]
                        for l in open(os.path.join(path, f)).readlines()]
                break
        text = text[1:]
        x = [t[0] for t in text]
        y = [int(t[2].split(':')[0]) for t in text]
        return x, y


    def regular_tweet(self, x):
        '''
        to regular a single tweet
        '''
        map_emoji = self.regular_emoji()
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



