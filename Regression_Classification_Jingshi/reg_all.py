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
from random import shuffle
from nltk.tokenize import TweetTokenizer
from copy import deepcopy
from sklearn.decomposition import PCA


if __name__ == '__main__':

    SVM = SVR()
    XGboost = GradientBoostingRegressor(n_estimators=200)
    MLP = MLPRegressor(hidden_layer_sizes=[100, 50], activation='logistic')


    # svm_coef = []
    # boost_pearson_coef, boost_spearman_coef = [], []
    mlp_pearson_coef, mlp_spearman_coef = [], []

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

        train_tfidf = np.loadtxt('./data/Features_reg/tfidf/' + _emotion + '.txt')
        train_BoW = np.loadtxt('./data/Features_reg/BoW/' + _emotion + '.txt')
        train_edinburgh = np.loadtxt('./data/Features_reg/edinburgh/' + _emotion + '.txt')
        train_glove = np.loadtxt('./data/Features_reg/glove/' + _emotion + '.txt')
        #
        Emoji = np.loadtxt('./data/Features_reg/Emoji/' + _emotion + '.txt')

        AFINN = np.loadtxt('./data/Features_reg/Lexicons/AFINN/' + _emotion + '.txt')
        BingLiu = np.loadtxt('./data/Features_reg/Lexicons/BingLiu/' + _emotion + '.txt')
        MPQA = np.loadtxt('./data/Features_reg/Lexicons/MPQA/' + _emotion + '.txt')
        NRC_EmoLex = np.loadtxt('./data/Features_reg/Lexicons/NRC_EmoLex/' + _emotion + '.txt')
        NRC_Hash_Emo = np.loadtxt('./data/Features_reg/Lexicons/NRC_Hash_Emo/' + _emotion + '.txt')
        NRC_Hash_Sent = np.loadtxt('./data/Features_reg/Lexicons/NRC_Hash_Sent/' + _emotion + '.txt')
        NRC10E = np.loadtxt('./data/Features_reg/Lexicons/NRC10E/' + _emotion + '.txt')
        Sentiment140 = np.loadtxt('./data/Features_reg/Lexicons/Sentiment140/' + _emotion + '.txt')
        SentiStrength = np.loadtxt('./data/Features_reg/Lexicons/SentiStrength/' + _emotion + '.txt')
        All_Lexicons = np.concatenate((AFINN, BingLiu, MPQA, NRC_EmoLex, NRC_Hash_Emo, NRC_Hash_Sent, NRC10E, Sentiment140, SentiStrength, Emoji), axis=1)


        # # PCA
        # pca = PCA(n_components=140)
        #
        # train_BoW = pca.fit_transform(train_BoW)
        # train_tfidf = pca.fit_transform(train_tfidf)
        # train_edinburgh = pca.fit_transform(train_edinburgh)
        # train_glove = pca.fit_transform(train_glove)


        # train_x = np.concatenate((All_Lexicons, train_tfidf, train_edinburgh, train_glove, train_BoW), axis = 1)
        train_x = np.concatenate((All_Lexicons, train_edinburgh, train_tfidf), axis=1)


        # train_x = train_BoW

        # train_x = SentiStrength
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
        #
        # print ('XGBoost regressor')
        # kf = KFold(n_splits=10, random_state=2, shuffle=True)
        # folds = kf.split(train_x, train_y)
        # kfold = 0
        # Pearson = []
        # Spearman = []
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
        #     Pearson.append(output[0])
        #     Spearman.append(output[1])
        #     print ('** ', kfold, 'fold: pearson coef = ', Pearson[-1])
        # boost_pearson_coef.append(Pearson)
        # boost_spearman_coef.append(Spearman)

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
            # print(training_input[0])
            training_output = [train_y[i] for i in idx1]
            #test data
            test_input = [train_x[i] for i in idx2]
            test_output = [train_y[i] for i in idx2]
            MLP.fit(training_input, training_output)
            output = measure_reg(test_output, MLP.predict(test_input))
            Pearson.append(output[0])
            Spearman.append(output[1])
            print ('** ', kfold, 'fold: pearson coef = ', Pearson[-1])
        mlp_pearson_coef.append(Pearson)
        mlp_spearman_coef.append(Spearman)

    import numpy as np
    # print ('') # Note: SVM is good for two-class classification problems
    # print(' ' * 10, '   SVM  ', '    XGBoost', '  MLP')
    # for i, _emotion in enumerate(['anger', 'fear', 'joy', 'sadness']):
    #     print('%10s%10.4f%10.4f%10.4f' % (_emotion, np.mean(svm_coef[i]), np.mean(boost_pearson_coef[i]), np.mean(mlp_pearson_coef[i])))
    #
    #
    # print ('')
    # print ('XGBoost')  # Good at sparse data set. If dim is less than 1400
    # print (' '*10,'   Pearson', '  Spearman')
    # for i, _emotion in enumerate(['anger', 'fear', 'joy', 'sadness']):
    #     print('%10s%10.4f%10.4f'%(_emotion, np.mean(boost_pearson_coef[i]), np.mean(boost_spearman_coef[i])))

    print ('')
    print('MLP') # Good at dense data set. If dim is larger than 1400
    print (' '*10,'   Pearson', '  Spearman')
    for i, _emotion in enumerate(['anger', 'fear', 'joy', 'sadness']):
        print('%10s%10.4f%10.4f'%(_emotion, np.mean(mlp_pearson_coef[i]), np.mean(mlp_spearman_coef[i])))