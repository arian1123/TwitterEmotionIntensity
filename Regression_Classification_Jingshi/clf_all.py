#! usr/bin/python
# coding=utf-8

from utils import *
from preprocess import *
from feature_factory import *
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
import numpy as np
from nltk.tokenize import TweetTokenizer


def vectorsToVector(arr):
    global vec
    vec = np.zeros(400)
    if len(arr) == 0:
        return vec
    for i in range(len(arr[0])):
        vals = [float(a[i]) for a in arr]
        vec[i] = sum(vals)
    return vec


def tweetToEmbeddings(tweet):
    tokens = tokenizer.tokenize(tweet)
    tweet_vec = [np.zeros(400) for _ in range(len(tokens))]

    for i, token in enumerate(tokens):
        if token in word_list:
            tweet_vec[i] = edinburg_embeddings[word_list.index(token)].split('\t')[:-1]

    return vectorsToVector(tweet_vec)


if __name__ == '__main__':

    clf1 = MLPClassifier(hidden_layer_sizes=[100, 20])
    # clf2 = GradientBoostingClassifier(n_estimators=100)

    # tokenizer = TweetTokenizer()
    # edinburg_embeddings = open("./embedding/w2v.twitter.edinburgh10M.400d.csv").readlines()
    # word_list = [line.split('\t')[-1].strip() for line in edinburg_embeddings]

    mlp_accu, mlp_corr = [], []
    # gdb_accu, gdb_corr = [], []
    for _emotion in ['anger', 'fear', 'joy', 'sadness']:
        print('')
        print('Emotion:', _emotion)
        train_x, train_y = load_original_oc(emotion=_emotion)

        # # use joy's training data for sadness
        # if _emotion == 'sadness':
        #     train_joy_x, train_joy_y = load_original_oc(emotion='joy')
        #     for i in range(len(train_joy_y)):
        #         train_joy_y[i] = 3 - train_joy_y[i]
        #     train_x = train_x + train_joy_x
        #     train_y = train_y + train_joy_y
        #
        # # use sadness's training data for joy
        # if _emotion == 'joy':
        #     train_sadness_x, train_sadness_y = load_original_oc(emotion='sadness')
        #     for i in range(len(train_sadness_y)):
        #         train_sadness_y[i] = 3 - train_sadness_y[i]
        #     train_x = train_x + train_sadness_x
        #     train_y = train_y + train_sadness_y

        # # preprocessing
        # for i in range(len(train_x)):
        #     train_x[i] = regular_tweet(train_x[i])

        # tf-idf
        # vectorizer = TfidfVectorizer(min_df=3)
        # vectorizer.fit(train_x)
        # train_x0 = vectorizer.transform(train_x).todense()
        # train_x0 = train_x0.tolist()
        # # w2v
        # train_w2v = train_x
        # for i in range(len(train_x)):
        #     train_w2v[i] = tweetToEmbeddings(train_x[i])
        # # normalize
        # train_w2v = preprocessing.normalize(train_w2v, norm='l2')

        # # bag of words
        # cdict = build_dict_from_corpus(train_x, min_freq=5)
        # train_x1 = lexicon_feature(train_x, cdict)
        #
        # # glove embedding
        # train_x2 = sum_of_word_embedding(train_x)
        # # normalize the glove vector to a unit vector, original glove vector ranges from about -20 to +20.
        # train_x2 = preprocessing.normalize(train_x2, norm='l2')

        train_tfidf = np.loadtxt('./data/Features_oc/tfidf/' + _emotion + '.txt')
        train_BoW = np.loadtxt('./data/Features_oc/BoW/' + _emotion + '.txt')
        train_edinburgh = np.loadtxt('./data/Features_oc/edinburgh/' + _emotion + '.txt')
        train_glove = np.loadtxt('./data/Features_oc/glove/' + _emotion + '.txt')
        #
        Emoji = np.loadtxt('./data/Features_oc/Emoji/' + _emotion + '.txt')

        AFINN = np.loadtxt('./data/Features_oc/Lexicons/AFINN/' + _emotion + '.txt')
        BingLiu = np.loadtxt('./data/Features_oc/Lexicons/BingLiu/' + _emotion + '.txt')
        MPQA = np.loadtxt('./data/Features_oc/Lexicons/MPQA/' + _emotion + '.txt')
        NRC_EmoLex = np.loadtxt('./data/Features_oc/Lexicons/NRC_EmoLex/' + _emotion + '.txt')
        NRC_Hash_Emo = np.loadtxt('./data/Features_oc/Lexicons/NRC_Hash_Emo/' + _emotion + '.txt')
        NRC_Hash_Sent = np.loadtxt('./data/Features_oc/Lexicons/NRC_Hash_Sent/' + _emotion + '.txt')
        NRC10E = np.loadtxt('./data/Features_oc/Lexicons/NRC10E/' + _emotion + '.txt')
        Sentiment140 = np.loadtxt('./data/Features_oc/Lexicons/Sentiment140/' + _emotion + '.txt')
        SentiStrength = np.loadtxt('./data/Features_oc/Lexicons/SentiStrength/' + _emotion + '.txt')
        All_Lexicons = np.concatenate((AFINN, BingLiu, MPQA, NRC_EmoLex, NRC_Hash_Emo, NRC_Hash_Sent, NRC10E, Sentiment140, SentiStrength, Emoji), axis=1)


        # PCA
        pca = PCA(n_components = 200)

        train_BoW = pca.fit_transform(train_BoW)
        train_tfidf = pca.fit_transform(train_tfidf)
        # train_edinburgh = pca.fit_transform(train_edinburgh)
        # train_glove = pca.fit_transform(train_glove)


        train_x = np.concatenate((All_Lexicons, train_tfidf, train_edinburgh), axis = 1)
        # train_x = np.concatenate((All_Lexicons, train_edinburgh, train_tfidf), axis=1)

        # train_x = SentiStrength

        print('training data has', len(train_x), 'samples', len(train_x[1]), 'dims')

        print('MLP')
        kf = KFold(n_splits=10, random_state=2, shuffle=True)
        folds = kf.split(train_x, train_y)
        kfold = 0
        accuracy = []
        correlation = []
        for idx1, idx2 in folds:
            kfold += 1
            # training data
            training_input = [train_x[i] for i in idx1]
            training_output = [train_y[i] for i in idx1]
            # test data
            test_input = [train_x[i] for i in idx2]
            test_output = [train_y[i] for i in idx2]
            clf1.fit(training_input, training_output)
            output = measure_clf(test_output, clf1.predict(test_input))
            output2 = measure_reg(test_output, clf1.predict(test_input))
            accuracy.append(output[0])
            correlation.append(output2[0])
            #print('** ', kfold, 'fold: accuracy = ', coef[-1])
        mlp_corr.append(correlation)
        mlp_accu.append(accuracy)

        # print('GDB')
        # kf = KFold(n_splits=10, random_state=2, shuffle=True)
        # folds = kf.split(train_x, train_y)
        # kfold = 0
        # accuracy = []
        # correlation = []
        # for idx1, idx2 in folds:
        #     kfold += 1
        #     # training data
        #     training_input = [train_x[i] for i in idx1]
        #     training_output = [train_y[i] for i in idx1]
        #     # test data
        #     test_input = [train_x[i] for i in idx2]
        #     test_output = [train_y[i] for i in idx2]
        #     clf2.fit(training_input, training_output)
        #     output = measure_clf(test_output, clf2.predict(test_input))
        #     output2 = measure_reg(test_output, clf2.predict(test_input))
        #     accuracy.append(output[0])
        #     correlation.append(output2[0])
        #     #print('** ', kfold, 'fold: accuracy = ', coef[-1])
        # gdb_corr.append(correlation)
        # gdb_accu.append(accuracy)


    import numpy as np

    print('Pearson Correlation')
    print(' ' * 10, '   MLP  ')
    for i, _emotion in enumerate(['anger', 'fear', 'joy', 'sadness']):
        print('%10s%10.4f' % (_emotion, np.mean(mlp_corr[i])))

    print('Accuracy')
    print(' ' * 10, '   MLP  ')
    for i, _emotion in enumerate(['anger', 'fear', 'joy', 'sadness']):
        print('%10s%10.4f' % (_emotion, np.mean(mlp_accu[i])))
