
import pickle
from evaluation_metrics import *
from preprocess import *
from feature_extraction import *
from lexicons import *
from sentistrength import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
import numpy as np
from nltk.tokenize import TweetTokenizer
from copy import deepcopy
from sklearn.decomposition import PCA

#generate variety of features
class TweetFeatureGenerator:

    def __init__(self, Preprocessor, emotion = 'anger'):
        '''
        This class extracts all the features from preprocessed traininig tweets, reduce the dimensions of lexicon features, 
        then save them to a file under 'data' folder.
        
        The features include tf-idf, bag of words, Edinburgh embeddings, 
        GloVe embeddings, hashtag intensity, emoji lexicon feature, 
        and affect lexicon features (AFINN, BingLiu, MPQA, NRC-EmoLex, 
        NRC-Hash-Emo, NRC-Hash-Sent, NRC10E, Sentiment140, SentiStrength).
        '''
        self.train_x = Preprocessor.train_x
        self.Preprocessor = Preprocessor
        self.emotion = emotion
        self.placehold_emotion_intensity = 0.5  # use value to fill missing/incomplete value for hashtag intensity
        # load Edinburgh word vectors
        self.edinburgh_embeddings = open("./embedding/w2v.twitter.edinburgh10M.400d.csv").readlines()
        self.word_list = [line.split('\t')[-1].strip() for line in self.edinburgh_embeddings]
        # load GloVe word vectors
        self.glove_embeddings = open("./embedding/glove.twitter.27B.200d.txt").readlines()
        self.glove_word_list = [line.split(' ')[0].strip() for line in self.glove_embeddings]
        self.tokenizer = TweetTokenizer()

        # Extracting features
        self.train_tfidf = self.tfidf(self.train_x)
        self.train_BoW = self.bag_of_words(self.train_x)
        self.train_edinburgh = self.edinburgh(self.train_x, False)
        self.train_glove = self.glove(self.train_x, False)
        self.hashtag_intense = self.build_hashtag_intensity()
        self.Emoji = self.Emoji_feature(self.train_x)
        self.AFINN = self.AFINN_feature(self.train_x)
        self.BingLiu = self.BingLiu_feature(self.train_x)
        self.MPQA = self.MPQA_feature(self.train_x)
        self.NRC_EmoLex = self.NRC_EmoLex_feature(self.train_x, self.emotion)
        self.NRC10E = self.NRC10E_feature(self.train_x, self.emotion)
        self.NRC_Hash_Emo = self.NRC_Hash_Emo_feature(self.train_x, self.emotion)
        self.NRC_Hash_Sent = self.NRC_Hash_Sent_feature(self.train_x)
        self.Sentiment140 = self.Sentiment140_feature(self.train_x)
        self.SentiStrength = self.SentiStrength_feature(self.train_x)

        # reduce the dimension to 140 based on a huristic that each tweet has a maximum length of 140.
        self.Emoji = self.reduce_dimension(self.Emoji)
        self.AFINN = self.reduce_dimension(self.AFINN)
        self.BingLiu = self.reduce_dimension(self.BingLiu)
        self.MPQA = self.reduce_dimension(self.MPQA)
        self.NRC_EmoLex = self.reduce_dimension(self.NRC_EmoLex)
        self.NRC10E = self.reduce_dimension(self.NRC10E)
        self.NRC_Hash_Emo = self.reduce_dimension(self.NRC_Hash_Emo)
        self.NRC_Hash_Sent = self.reduce_dimension(self.NRC_Hash_Sent)
        self.Sentiment140 = self.reduce_dimension(self.Sentiment140)
        self.SentiStrength = self.reduce_dimension(self.SentiStrength)

        # save the features to files
        self.write_to_file(self.emotion, 'tfidf', self.train_tfidf, False)
        self.write_to_file(self.emotion, 'BoW', self.train_BoW, False)
        self.write_to_file(self.emotion, 'edinburgh', self.train_edinburgh, False)
        self.write_to_file(self.emotion, 'glove', self.train_glove, False)
        self.write_to_file(self.emotion, 'Emoji', self.Emoji, False)
        self.write_to_file(self.emotion, 'AFINN', self.AFINN, True)
        self.write_to_file(self.emotion, 'BingLiu', self.BingLiu, True)
        self.write_to_file(self.emotion, 'MPQA', self.MPQA, True)
        self.write_to_file(self.emotion, 'NRC_EmoLex', self.NRC_EmoLex, True)
        self.write_to_file(self.emotion, 'NRC10E', self.NRC10E, True)
        self.write_to_file(self.emotion, 'NRC_Hash_Emo', self.NRC_Hash_Emo, True)
        self.write_to_file(self.emotion, 'NRC_Hash_Sent', self.NRC_Hash_Sent, True)
        self.write_to_file(self.emotion, 'Sentiment140', self.Sentiment140, True)
        self.write_to_file(self.emotion, 'SentiStrength', self.SentiStrength, True)
        self.write_to_file(self.emotion, 'hashtag_intense', self.hashtag_intense, False)


    def bag_of_words(self, train_x):
        '''
        bag of words feature
        '''
        cdict = self.build_dict_from_corpus(train_x, min_freq=5)
        train_BoW = self.lexicon_feature(train_x, cdict)
        return train_BoW


    def build_dict_from_corpus(self, x, min_freq):
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


    def lexicon_feature(self, x, dictionary):
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


    def tfidf(self, train_x):
        '''
        tf-idf feature
        '''
        vectorizer = TfidfVectorizer(min_df=3)
        vectorizer.fit(train_x)
        train_tfidf = vectorizer.transform(train_x).todense()
        train_tfidf = train_tfidf.tolist()
        return train_tfidf


    def edinburgh(self, train_x, is_normalize):
        '''
        edinburgh embedding feature
        '''
        train_edinburgh = deepcopy(train_x)
        for i in range(len(train_x)):
            train_edinburgh[i] = self.tweetToEdinburgh(train_x[i])
        # normalize
        if is_normalize:
            train_edinburgh = preprocessing.normalize(train_edinburgh, norm='l2')
        return train_edinburgh


    def tweetToEdinburgh(self, tweet):
        '''
        edinburgh embedding for a single tweet
        '''

        tokens = self.tokenizer.tokenize(tweet)
        tweet_vec = [np.zeros(400) for _ in range(len(tokens))]
        for i, token in enumerate(tokens):
            if token in self.word_list:
                tweet_vec[i] = self.edinburgh_embeddings[self.word_list.index(token)].split('\t')[:-1]
        return self.vectorsToVector(tweet_vec, 400)


    def glove(self, train_x, is_normalize):
        '''
        glove embedding feature
        '''
        train_glove = deepcopy(train_x)
        for i in range(len(train_x)):
            train_glove[i] = self.tweetToGlove(train_x[i])
        # normalize
        if is_normalize:
            train_glove = preprocessing.normalize(train_glove, norm='l2')
        return train_glove


    def tweetToGlove(self, tweet):
        '''
        glove embedding for a single tweet
        '''
        glove_tokens = self.tokenizer.tokenize(tweet)
        tweet_vec = [np.zeros(200) for _ in range(len(glove_tokens))]
        for i, token in enumerate(glove_tokens):
            if token in self.glove_word_list:
                tweet_vec[i] = self.glove_embeddings[self.glove_word_list.index(token)].split(' ')[1:]
        return self.vectorsToVector(tweet_vec, 200)


    def vectorsToVector(self, arr, dim):
        '''
        vectors to vector
        '''
        global vec
        vec = np.zeros(dim)
        if len(arr) == 0:
            return vec
        for i in range(len(arr[0])):
            vals = [float(a[i]) for a in arr]
            vec[i] = sum(vals)
        return vec


    def Emoji_feature(self, train_x):
        '''
        Emoji feature
        '''
        Emoji = deepcopy(train_x)
        for i in range(len(train_x)):
            tmp = deepcopy(train_x[i])
            Emoji[i] = tweetToEmoji(train_x[i])
            train_x[i] = tmp
        return Emoji


    def AFINN_feature(self, train_x):
        '''
        AFINN feature
        '''
        AFINN = deepcopy(train_x)
        for i in range(len(train_x)):
            tmp = deepcopy(train_x[i])
            AFINN[i] = tweetToAFINNVector(train_x[i])
            train_x[i] = tmp
        return AFINN


    def BingLiu_feature(self, train_x):
        '''
        BingLiu feature
        '''
        BingLiu = deepcopy(train_x)
        for i in range(len(train_x)):
            tmp = deepcopy(train_x[i])
            BingLiu[i] = tweetToBingLiuVector(train_x[i])
            train_x[i] = tmp
        return BingLiu


    def MPQA_feature(self, train_x):
        '''
        MPQA feature
        '''
        MPQA = deepcopy(train_x)
        for i in range(len(train_x)):
            tmp = deepcopy(train_x[i])
            MPQA[i] = tweetToMPQAVector(train_x[i])
            train_x[i] = tmp
        return MPQA


    def NRC_EmoLex_feature(self, train_x, emotion):
        '''
        NRC Emolex feature
        '''
        NRC_EmoLex = deepcopy(train_x)
        for i in range(len(train_x)):
            tmp = deepcopy(train_x[i])
            NRC_EmoLex[i] = tweetToEmoLexVector(train_x[i], emotion)
            train_x[i] = tmp
        return NRC_EmoLex


    def NRC10E_feature(self, train_x, emotion):
        '''
        NRC10E feature
        '''
        NRC10E = deepcopy(train_x)
        for i in range(len(train_x)):
            tmp = deepcopy(train_x[i])
            NRC10E[i] = tweetToEmo10EVector(train_x[i], emotion)
            train_x[i] = tmp
        return NRC10E


    def NRC_Hash_Emo_feature(self, train_x, emotion):
        '''
        NRC Hash Emo feature
        '''
        NRC_Hash_Emo = deepcopy(train_x)
        for i in range(len(train_x)):
            tmp = deepcopy(train_x[i])
            NRC_Hash_Emo[i] = tweetToHSEVector(train_x[i], emotion)
            train_x[i] = tmp
        return NRC_Hash_Emo


    def NRC_Hash_Sent_feature(self, train_x):
        '''
        NRC Hash Sent feature
        '''
        NRC_Hash_Sent = deepcopy(train_x)
        for i in range(len(train_x)):
            tmp = deepcopy(train_x[i])
            NRC_Hash_Sent[i] = tweetToHSVector(train_x[i])
            train_x[i] = tmp
        return NRC_Hash_Sent


    def Sentiment140_feature(self, train_x):
        '''
        Sentiment140 feature
        '''
        Sentiment140 = deepcopy(train_x)
        for i in range(len(train_x)):
            tmp = deepcopy(train_x[i])
            Sentiment140[i] = tweetToSentiment140Vector(train_x[i])
            train_x[i] = tmp
        return Sentiment140


    def SentiStrength_feature(self, train_x):
        '''
        SentiStrength feature
        '''
        SentiStrength = deepcopy(train_x)
        for i in range(len(train_x)):
            tmp = deepcopy(train_x[i])
            SentiStrength[i] = sentistrength.tweetToSentiStrength(train_x[i])
            train_x[i] = tmp
        return SentiStrength


    def reduce_dimension(self, feature):
        '''
        reduce dimension of a feature, by using PCA
        '''
        pca_pre = PCA(n_components=5000)
        pca = PCA(n_components=140)
        if (len(feature[0]) < 50000):
            feature = pca.fit_transform(feature)
        if (len(feature[0]) >= 50000):
            feature = pca_pre.fit_transform(feature)
            feature = pca.fit_transform(feature)
        return feature


    def write_to_file(self, emotion_name, feature_name, feature, lexicon):
        '''
        save the features
        '''
        if lexicon:
            file_name = './data/Features_reg/Lexicons/' + feature_name + '/' + emotion_name + '.txt'
        else:
            file_name = './data/Features_reg/' + feature_name + '/' + emotion_name + '.txt'
        np.savetxt(file_name, feature)


    #
    #
    # Functions for Hashtag intensity
    #
    #
    #
    def build_hashtag_intensity(self):
        '''
        generate hashtag intensity model (for use in feature vector)
        '''
        self.depechemood_dict = self.get_depechemood_dict()
        hashtags = self.Preprocessor.hashtags
        model = np.array([self.hashtag_intensity(h, emotion=self.emotion) for h in hashtags])

        return model.reshape((len(model), 1))

    def get_depechemood_dict(self, type='freq'):
        '''
        Create DepecheMood sentiment dictionary as python data structure.
        Add word tag to dictionary.
        '''
        path = "./DepecheMood/DepecheMood_" + type + ".txt"

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


    def hashtag_intensity(self, hashtags, emotion='angry'):
        '''
        Get intensity of hashtag words from depechemood dictionary.
        Return average of all hashtags for single tweet for passed emotion.
        '''
        depeche_emotion = self.map_emotion_depechemood(emotion)

        if not hashtags:
            avg_intensity = self.placehold_emotion_intensity
        else:
            avg_intensity = np.average([self.get_depechemood_score(i, depeche_emotion) for i in hashtags])

        return avg_intensity

    def map_emotion_depechemood(self, emotion):
        '''
        return matching emotion for depechemood dictionary.
        '''

        if(emotion == 'anger'): return 'angry'
        if(emotion == 'fear'): return 'afraid'
        if(emotion == 'joy'): return 'happy'
        if(emotion == 'sadness'): return 'sad'
        else: return False

    def get_depechemood_score(self, word="", emotion="angry"):
        '''
        get a depechemood intensity for word + target emotion.
        '''

        #format input properly
        w = word.lower()
        e = emotion.lower()

        #if word is not found in dict, return 0
        if(w in self.depechemood_dict):
            score = self.depechemood_dict[w][e]
        else:
            score = self.placehold_emotion_intensity

        return score








