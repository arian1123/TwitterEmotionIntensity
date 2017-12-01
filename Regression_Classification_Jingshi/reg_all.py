#! usr/bin/python
# coding=utf-8

from utils import *
from preprocess import *
from feature_factory import *
from lexicons import *
from sentistrength import *
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
import numpy as np
from nltk.tokenize import TweetTokenizer
from copy import deepcopy
from sklearn.decomposition import PCA

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

    # SVM = SVR()
    # XGboost = GradientBoostingRegressor(n_estimators=200)
    MLP = MLPRegressor(hidden_layer_sizes=[100, 50], activation='logistic')

    # tokenizer = TweetTokenizer()
    # edinburg_embeddings = open("./embedding/w2v.twitter.edinburgh10M.400d.csv").readlines()
    # word_list = [line.split('\t')[-1].strip() for line in edinburg_embeddings]

    # glove_embeddings = open("./embedding/glove.twitter.27B/glove.twitter.27B.200d.txt").readlines()
    # glove_word_list = [line.split(' ')[0].strip() for line in glove_embeddings]

    pearson_coef, spearman_coef = [], []
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
        # for i in range(len(train_x)):
        #     train_x[i] = regular_tweet(train_x[i])

        # tf-idf
        # vectorizer = TfidfVectorizer(min_df=3)
        # vectorizer.fit(train_x)
        # train_tfidf = vectorizer.transform(train_x).todense()
        # train_tfidf = train_tfidf.tolist()
        #
        # bag of words
        cdict = build_dict_from_corpus(train_x, min_freq=5)
        train_BoW = lexicon_feature(train_x, cdict)
        #
        # # w2v
        # edingurgh embedding
        # train_edinburgh = deepcopy(train_x)
        # for i in range(len(train_x)):
        #     train_edinburgh[i] = tweetToEdinburg(train_x[i])
        # # normalize
        # train_edinburgh = preprocessing.normalize(train_edinburgh, norm='l2')
        #
        #
        #
        # # glove embedding
        # train_glove = deepcopy(train_x)
        # for i in range(len(train_x)):
        #     train_glove[i] = tweetToGlove(train_x[i])
        # # normalize
        # train_glove = preprocessing.normalize(train_glove, norm='l2')
        #
        # # initialize lexicons
        # Emoji = deepcopy(train_x)
        # AFINN = deepcopy(train_x)
        # BingLiu = deepcopy(train_x)
        # MPQA = deepcopy(train_x)
        # NRC_Hash_Emo = deepcopy(train_x)
        # SentiStrength = deepcopy(train_x)
        #
        # for i in range(len(train_x)):
        #     tmp = deepcopy(train_x[i])
        #
        #     Emoji[i] = tweetToEmoji(train_x[i])
        #     train_x[i] = tmp
        #     tmp = deepcopy(train_x[i])
        #
        #
        #     AFINN[i] = tweetToAFINNVector(train_x[i])
        #     train_x[i] = tmp
        #     tmp = deepcopy(train_x[i])
        #
        #     BingLiu[i] = tweetToBingLiuVector(train_x[i])
        #     train_x[i] = tmp
        #     tmp = deepcopy(train_x[i])
        #
        #     MPQA[i] = tweetToMPQAVector(train_x[i])
        #     train_x[i] = tmp
        #     tmp = deepcopy(train_x[i])
        #
        #     NRC_Hash_Emo[i] = tweetToHSEVector(train_x[i], _emotion)
        #     train_x[i] = tmp
        #     tmp = deepcopy(train_x[i])
        #
        #     SentiStrength[i] = sentistrength.tweetToSentiStrength(train_x[i])
        #     train_x[i] = tmp
        #     tmp = deepcopy(train_x[i])
        #
        #
        # #print(len(train_x0[0]), ' and ', len(train_x1[0]), ' and ', len(train_x2[0]))
        # #print(train_x0[0], ' and ', train_x1[0], ' and ', train_x2[0])
        # pca = PCA(n_components= 300)
        #
        # # train_tfidf = pca.fit_transform(train_tfidf)
        # # train_BoW = pca.fit_transform(train_BoW)
        # # train_edinburgh = pca.fit_transform(train_edinburgh)
        # # train_glove = pca.fit_transform(train_glove)
        #
        # AFINN = pca.fit_transform(AFINN)
        # BingLiu = pca.fit_transform(BingLiu)
        # MPQA = pca.fit_transform(MPQA)
        # NRC_Hash_Emo = pca.fit_transform(NRC_Hash_Emo)
        # SentiStrength = pca.fit_transform(SentiStrength)


        # # print(train_tfidf[0])
        # # print(train_edinburgh[0])
        # # print(NRC_Hash_Emo[0])
        #
        # train_x = np.concatenate((train_tfidf, train_BoW, train_edinburgh, train_glove, Emoji, AFINN, BingLiu, MPQA, NRC_Hash_Emo, SentiStrength), axis=1)
        train_x = train_BoW

        print ('training data has', len(train_x), 'samples', len(train_x[1]), 'dims')

        # print ('SVM regressor')
        # kf = KFold(n_splits=10, random_state=2, shuffle=True)
        # folds = kf.split(train_x, train_y)
        # kfold = 0
        # coef = []
        # for idx1, idx2 in folds:
        #     kfold += 1
        #     #training data
        #     training_input = [train_x[i] for i in idx1]
        #     training_output = [train_y[i] for i in idx1]
        #     #test data
        #     test_input = [train_x[i] for i in idx2]
        #     test_output = [train_y[i] for i in idx2]
        #     SVM.fit(training_input, training_output)
        #     output = measure_reg(test_output, SVM.predict(test_input))
        #     coef.append(output[0])
        #     print ('** ', kfold, 'fold: pearson coef = ', coef[-1])
        # svm_coef.append(coef)
        #
        # print ('XGBoost regressor')
        # kf = KFold(n_splits=10, random_state=2, shuffle=True)
        # folds = kf.split(train_x, train_y)
        # kfold = 0
        # coef = []
        # for idx1, idx2 in folds:
        #     kfold += 1
        #     #training data
        #     training_input = [train_x[i] for i in idx1]
        #     training_output = [train_y[i] for i in idx1]
        #     #test data
        #     test_input = [train_x[i] for i in idx2]
        #     test_output = [train_y[i] for i in idx2]
        #     XGboost.fit(training_input, training_output)
        #     output = measure_reg(test_output, XGboost.predict(test_input))
        #     coef.append(output[0])
        #     print ('** ', kfold, 'fold: pearson coef = ', coef[-1])
        # boost_coef.append(coef)

        print ('MLP regressor')
        kf = KFold(n_splits=10, random_state=2, shuffle=True)
        folds = kf.split(train_x, train_y)
        kfold = 0
        Pearson = []
        Spearman = []
        for idx1, idx2 in folds:
            kfold += 1
            #training data
            training_input = [train_x[i] for i in idx1]
            training_output = [train_y[i] for i in idx1]
            #test data
            test_input = [train_x[i] for i in idx2]
            test_output = [train_y[i] for i in idx2]
            MLP.fit(training_input, training_output)
            output = measure_reg(test_output, MLP.predict(test_input))
            Pearson.append(output[0])
            Spearman.append(output[1])
            print ('** ', kfold, 'fold: pearson coef = ', Pearson[-1])
        pearson_coef.append(Pearson)
        spearman_coef.append(Spearman)

    import numpy as np
    print ('')
    print (' '*10,'   Pearson', '  Spearman')
    for i, _emotion in enumerate(['anger', 'fear', 'joy', 'sadness']):
        print('%10s%10.4f%10.4f'%(_emotion, np.mean(pearson_coef[i]), np.mean(spearman_coef[i])))