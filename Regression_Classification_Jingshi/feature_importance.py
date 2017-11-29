#! usr/bin/python
# coding=utf-8

from utils import *
from preprocess import *
from feature_factory import *
from lexicons import *
from sentistrength import *
from sklearn.svm import SVR
#from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
import numpy as np
from nltk.tokenize import TweetTokenizer
from copy import deepcopy
from sklearn.decomposition import PCA
from matplotlib import pylab as plt


def vectorsToVector(arr, dim):

    global vec
    vec = np.zeros(dim)
    if len(arr) == 0:
        return vec
    for i in range(len(arr[0])):
        vals = [float(a[i]) for a in arr]
        vec[i] = sum(vals)
    return vec

def tweetToEdinburg(tweet):
    tokens = tokenizer.tokenize(tweet)
    tweet_vec = [np.zeros(400) for _ in range(len(tokens))]

    for i, token in enumerate(tokens):
        if token in word_list:
            tweet_vec[i] = edinburg_embeddings[word_list.index(token)].split('\t')[:-1]

    return vectorsToVector(tweet_vec, 400)

def tweetToGlove(tweet):
    tokens = tokenizer.tokenize(tweet)
    tweet_vec = [np.zeros(200) for _ in range(len(tokens))]

    for i, token in enumerate(tokens):
        if token in glove_word_list:
            tweet_vec[i] = glove_embeddings[glove_word_list.index(token)].split(' ')[1:]

    return vectorsToVector(tweet_vec, 200)

if __name__ == '__main__':

    #SVM = SVR()
    XGboost = GradientBoostingRegressor(n_estimators = 600)
    #MLP = MLPRegressor(hidden_layer_sizes=[100, 50], activation='logistic')

    # tokenizer = TweetTokenizer()
    # edinburg_embeddings = open("./embedding/w2v.twitter.edinburgh10M.400d.csv").readlines()
    # word_list = [line.split('\t')[-1].strip() for line in edinburg_embeddings]
    #
    # glove_embeddings = open("./embedding/glove.twitter.27B/glove.twitter.27B.200d.txt").readlines()
    # glove_word_list = [line.split(' ')[0].strip() for line in glove_embeddings]

    #svm_coef, boost_coef, mlp_coef = [], [], []
    for _emotion in ['anger', 'fear', 'joy', 'sadness']:
        print ('')
        print ('Emotion:', _emotion)
        train_x, train_y = load_original_reg(emotion = _emotion)

        # use joy's training data for sadness
        if _emotion=='sadness' :
            train_joy_x, train_joy_y = load_original_reg(emotion = 'joy')
            for i in range(len(train_joy_y)):
                train_joy_y[i] = 1 - train_joy_y[i]
            train_x = train_x + train_joy_x
            train_y = train_y + train_joy_y

        # preprocessing
        for i in range(len(train_x)):
            train_x[i] = regular_tweet(train_x[i])

        # # tf-idf
        # vectorizer = TfidfVectorizer(min_df=3)
        # vectorizer.fit(train_x)
        # train_tfidf = vectorizer.transform(train_x).todense()
        # train_tfidf = train_tfidf.tolist()
        # # normalize
        # train_tfidf = preprocessing.normalize(train_tfidf, norm='l2')
        #
        # # bag of words
        # cdict = build_dict_from_corpus(train_x, min_freq=5)
        # train_bag_of_words = lexicon_feature(train_x, cdict)
        # # normalize
        # train_bag_of_words = preprocessing.normalize(train_bag_of_words, norm='l2')

        # initialize lexicons
        AFINN = deepcopy(train_x)
        BingLiu = deepcopy(train_x)
        MPQA = deepcopy(train_x)
        NRC_Hash_Emo = deepcopy(train_x)
        SentiStrength = deepcopy(train_x)

        for i in range(len(train_x)):
            tmp = deepcopy(train_x[i])
            AFINN[i] = tweetToAFINNVector(train_x[i])
            train_x[i] = tmp
            tmp = deepcopy(train_x[i])

            BingLiu[i] = tweetToBingLiuVector(train_x[i])
            train_x[i] = tmp
            tmp = deepcopy(train_x[i])

            MPQA[i] = tweetToMPQAVector(train_x[i])
            train_x[i] = tmp
            tmp = deepcopy(train_x[i])

            NRC_Hash_Emo[i] = tweetToHSEVector(train_x[i], _emotion)
            train_x[i] = tmp
            tmp = deepcopy(train_x[i])

            SentiStrength[i] = sentistrength.tweetToSentiStrength(train_x[i])
            train_x[i] = tmp
            tmp = deepcopy(train_x[i])



        # normalize lexicons
        # AFINN = preprocessing.normalize(AFINN, norm='l2')
        # BingLiu = preprocessing.normalize(BingLiu, norm='l2')
        # MPQA = preprocessing.normalize(MPQA, norm='l2')
        # NRC_Hash_Emo = preprocessing.normalize(NRC_Hash_Emo, norm='l2')
        # SentiStrength = preprocessing.normalize(SentiStrength, norm='l2')

        # get All_Lexicons by concatenating all the normalized single lexicons
        All_Lexicons = np.concatenate((AFINN, BingLiu, MPQA, NRC_Hash_Emo, SentiStrength), axis=1)
        # normalize All_Lexicons
        # All_Lexicons = preprocessing.normalize(All_Lexicons, norm='l2')

        # # w2v embedding
        #
        # # edinburgh
        # train_x_tmp = deepcopy(train_x)
        # train_edinburg = train_x
        # for i in range(len(train_x)):
        #     train_edinburg[i] = tweetToEdinburg(train_x[i])
        # # normalize
        # train_edinburg = preprocessing.normalize(train_edinburg, norm='l2')
        #
        # # glove
        # train_glove = train_x_tmp
        # for i in range(len(train_x_tmp)):
        #     train_glove[i] = tweetToGlove(train_x_tmp[i])
        # # normalize
        # train_glove = preprocessing.normalize(train_glove, norm='l2')

        print('Done feature generation, next PCA')

        # PCA
        pca = PCA(n_components=1)

        # train_bag_of_words = pca.fit_transform(train_bag_of_words)
        # train_tfidf = pca.fit_transform(train_tfidf)
        AFINN = pca.fit_transform(AFINN)
        BingLiu = pca.fit_transform(BingLiu)
        MPQA = pca.fit_transform(MPQA)
        NRC_Hash_Emo = pca.fit_transform(NRC_Hash_Emo)
        SentiStrength = pca.fit_transform(SentiStrength)
        # train_edinburg = pca.fit_transform(train_edinburg)
        # train_glove = pca.fit_transform(train_glove)
        All_Lexicons = pca.fit_transform(All_Lexicons)

        #train_x = train_bag_of_words
        train_x = np.concatenate((AFINN, BingLiu, MPQA, NRC_Hash_Emo, SentiStrength, All_Lexicons), axis=1)
        feature_names = np.array(['AFINN', 'BingLiu', 'MPQA', 'NRC_Hash_Emo', 'SentiStrength', 'All_Lexicons'])

        XGboost.fit(train_x, train_y)
        # Plot feature importance
        feature_importance = XGboost.feature_importances_
        # make importances relative to max importance
        feature_importance = 100.0 * (feature_importance / feature_importance.max())
        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + .5
        plt.subplot(1, 2, 2)
        plt.barh(pos, feature_importance[sorted_idx], align='center')
        plt.yticks(pos, feature_names[sorted_idx])
        plt.xlabel('Relative Importance')
        plt.title(_emotion.title())
        plt.show()

        print('training data has', len(train_x), 'samples', len(train_x[1]), 'dims')