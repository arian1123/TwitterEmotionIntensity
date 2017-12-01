#pre processing and feature extraction for training files

import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import TweetTokenizer
from sklearn import preprocessing
from random import shuffle
from nltk.stem import WordNetLemmatizer
import nltk
import scipy as sp
from Arian import lexicon_functions
from sklearn.decomposition import PCA

#Use this class to prepare and parse tweets for feature generation
class TweetParser:

    #pass in directory with training data files
    def __init__(self, filepath, emotion="anger"):

        self.tweet_list = [] #list of dictionaries to hold tweets data
        self.tweet_list_dataframe = []
        self.emotion = emotion

        for file in filepath:
            self.process_file(file)

        self.tweet_list_dataframe = pd.DataFrame(self.tweet_list)

    #end init

    def process_file(self, path):

        #flag to see if we are parsing the file of the emotion or its opposite emotion i.e. sadness vs joy
        if(path.find(self.emotion) != -1):
            emotion_file = True
        else:
            emotion_file = False

        file = open(path)
        for f in file:

            f = f.replace("\n", "")
            tokens = f.split("\t")

            data = {}
            data['id'] = tokens[0]
            data['content'] = self.clean_tweet(tokens[1])
            data['emotion'] = tokens[2]

            #if parsing opposite emotion set emotion intensity to 1 -
            if(emotion_file):
                data['emotion_intensity'] = tokens[3]
            else:
                data['emotion_intensity'] = 1 - float(tokens[3])

            data['hashtags'] = self.extract_hashtags(tokens[1])
            data['pos_tags'] = self.tag_token_pos(data['content'])

            self.tweet_list.append(data)

        shuffle(self.tweet_list)

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

    def tag_token_pos(self, content):

        tokens = nltk.tokenize.word_tokenize(content)
        pos = nltk.pos_tag(tokens)
        return pos

    #end tag_token_pos


    #use DepecheMood to get a 0-1 intensity of hashtagged term
    def hashtag_intensity(self, term):

        print('blah')


    #end hashtag_intensity


    ####### update to use emoji to word dictioanry
    def def_regular_emoji(self):
        '''
        use a unique symbol emoji_#No. to replace an emoji
        '''
        map_emoji = dict()
        prefix = ' emoji'  # extra space ensures independence
        with open('Arian/data/emoji_map.txt') as f:
            emoji = [l.strip() for l in f.readlines()]

        for i, e in enumerate(emoji):
            map_emoji.update({e: prefix + str(i) + ' '})  # extra space ensures independence

        return map_emoji

    #end def_regular_emoji


    #from Regression_Jingshi/preprocess.py
    def clean_tweet(self, x):

        map_emoji = self.def_regular_emoji()
        emoji = map_emoji.keys()

        # 2. regular emoji
        for e in emoji:
            if e in x:
                x = x.replace(e, map_emoji[e])


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

    def __init__(self, TweetParser, bag_of_words=False,
                 word2vec=False, hashtag_intensity=False, truncate_dict=False, tfidf=False, lexicon=False):

        self.tweet_data = TweetParser #TweetProcessor Object, generated features from parsed data
        self.truncate_dict = truncate_dict
        self.depechemood_dict = {} #populate later, if need be
        self.emotion = self.tweet_data.emotion
        self.placehold_emotion_intensity = 0.5 #use value to fill missing/incomplete value for intensity
        self.pca_n = 140

        features = []

        #get all individual features
        if(bag_of_words == True):
            self.vocab, self.bag_words_model = self.build_bag_words()
            features.append(self.bag_words_model)

        if(hashtag_intensity == True):
            self.hashtag_intensity_model = self.build_hashtag_intensity()
            features.append(self.hashtag_intensity_model)

        if(tfidf == True):
            self.tf_idf_model = self.build_tf_idf()
            features.append(self.tf_idf_model)

        if(word2vec == True):
            self.word2vec_model = self.build_word2vec_model()
            features.append(self.word2vec_model)

        if(lexicon == True):
            self.lexicon_model, self.lexicon_vectors_dict, self.used_lexicon_list = self.build_lexicon_model()
            features.append(self.lexicon_model)

        #build total features vector from all individual features

        if(features is not []):


            self.features_vector = self.pca_feature(features[0])

            if(len(features) > 1):

                for f in features[1:]:
                    f = self.pca_feature(f)
                    self.features_vector = np.column_stack((self.features_vector, f))

    #end init

    #compress dimension of individual feature vector
    def pca_feature(self, feature):

        pca = PCA(n_components=self.pca_n)
        shape = np.array(feature).shape

        #list
        if(len(shape) <= 1):
            return feature
        elif shape[1] <= self.pca_n:
            return feature
        else:
            return pca.fit_transform(feature)

    #end pca_feature

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
            pos_tag = tokens[0].split("#", 1)[1]
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

        return score

    #end get_depechemood_score

    #
    #
    # Word2Vec implementation
    #
    #

    def get_edinburg_embeddings(self):
        edinburg_embeddings = open("Arian/data/embedding/w2v.twitter.edinburgh10M.400d.csv").readlines()
        edinburg_word_list = [line.split('\t')[-1].strip() for line in edinburg_embeddings]

        return edinburg_embeddings, edinburg_word_list

    #end get_edinburgh_embeddings

    def vectors_to_vector(self, arr):

        global vec
        vec = np.zeros(400)
        if len(arr) == 0:
            return vec
        for i in range(len(arr[0])):
            vals = [float(a[i]) for a in arr]
            vec[i] = sum(vals)
        return vec

    #end vectorsToVector

    def tweet_to_embeddings(self, tweet):

        tokenizer = TweetTokenizer()
        tokens = tokenizer.tokenize(tweet)
        tweet_vec = [np.zeros(400) for _ in range(len(tokens))]

        for i, token in enumerate(tokens):
            if token in self.edinburg_word_list:
                tweet_vec[i] = self.edinburg_embeddings[self.edinburg_word_list.index(token)].split('\t')[:-1]

        return self.vectors_to_vector(tweet_vec)

    #end tweetToEmbeddings

    def build_word2vec_model(self):

        self.edinburg_embeddings, self.edinburg_word_list = self.get_edinburg_embeddings()
        corpus = self.tweet_data.tweet_list_dataframe['content'].tolist()
        train_w2v = corpus
        for i in range(len(corpus)):
            train_w2v[i] = self.tweet_to_embeddings(corpus[i])
        # normalize
        #train_w2v = preprocessing.normalize(train_w2v, norm='l2')

        return train_w2v

    #end build_word2vec_model

    #
    #
    # td-idf model
    #
    #

    def build_tf_idf(self):

        vectorizer = TfidfVectorizer(min_df=3)
        corpus = self.tweet_data.tweet_list_dataframe['content'].tolist()
        vectorizer.fit(corpus)
        train_x0 = vectorizer.transform(corpus).todense()
        train_x0 = train_x0.tolist()

        return train_x0


    #end build_td_idf

    #
    #
    # Lexicon features
    #
    #

    def build_lexicon_model(self, Emoji=False, AFINN=False, BingLiu=False, MPQA=False, NRC_Hash_Emo=False, SentiStrength=False,
                            all_lexicons=True):

        #list of tweets
        train_x = self.tweet_data.tweet_list_dataframe['content'].tolist()

        lex_dict, lex_list  = self.get_lexicon_ids()

        #list of lexicon names being used
        lexicons = []

        #dict of list of lexicon vectors
        lex_vectors = {}

        #add all lexicons individually
        if(all_lexicons == True):
            lexicons = lex_list
        else:
            if(Emoji == True):
                lexicons.append(lex_dict['emoji'])
            if(AFINN == True):
                lexicons.append((lex_dict['bingliu']))
            if(BingLiu == True):
                lexicons.append(lex_dict['bingliu'])
            if(MPQA == True):
                lexicons.append(lex_dict['mpqa'])
            if(NRC_Hash_Emo == True):
                lexicons.append(lex_dict['nrc_hashemo'])
            if(SentiStrength == True):
                lexicons.append(lex_dict['sentistrength'])

        #pathological case
        if(lexicons is []):
            return

        #initialize all individual dictionary vectors
        for lex in lexicons:
            lex_vectors[lex] = []

        #populate dictionary vectors
        for tweet in train_x:

            for lex in lexicons:

                vec = self.tweet_to_lexicon_vec(tweet, lexicon=lex)
                lex_vectors[lex].append(vec)

        #concatenate model
        model = lex_vectors[lexicons[0]]

        if (len(lexicons) > 1):

            for lex in lexicons[1:]:
                model = np.column_stack((model, lex_vectors[lex]))

        #condense vector size
        pca = PCA(n_components=300)
        model = pca.fit_transform(model)

        return model, lex_vectors, lexicons

    #end build lexicon

    def tweet_to_lexicon_vec(self, tweet, lexicon="Emoji"):

        lex_dict = self.get_lexicon_ids()[0]

        if(lexicon == lex_dict['emoji']):
            return lexicon_functions.tweetToEmoji(tweet)
        elif(lexicon == lex_dict['afinn']):
            return lexicon_functions.tweetToAFINNVector(tweet)
        elif(lexicon == lex_dict['bingliu']):
            return lexicon_functions.tweetToBingLiuVector(tweet)
        elif(lexicon == lex_dict['mpqa']):
            return lexicon_functions.tweetToMPQAVector(tweet)
        elif(lexicon == lex_dict['nrc_hashemo']):
            return lexicon_functions.tweetToHSEVector(tweet, self.emotion)
        elif(lexicon == lex_dict['sentistrength']):
            return lexicon_functions.tweetToSentiStrength(tweet)
        else:
            return

    #end tweet_to_lexicon_vec

    def get_lexicon_ids(self):

        #dictionary name constant identifiers
        dict = {}
        list = []
        emoji = "Emoji"
        afinn = "AFINN"
        bingliu = "BingLiu"
        mpqa = "MPQA"
        nrc = "NRC_Hash_Emo"
        senti = "SentiStrength"
        dict['emoji'] = emoji
        list.append(emoji)
        dict['afinn'] = afinn
        list.append(afinn)
        dict['bingliu'] = bingliu
        list.append(bingliu)
        dict['mpqa'] = mpqa
        list.append(mpqa)
        dict['nrc_hashemo'] = nrc
        list.append(nrc)
        dict['sentistrength'] = senti
        list.append(senti)

        #return dictionary and list of all dictionaries used
        return dict, list

    #end get_lexicon_ids



#End class TweetFeatureGenerator