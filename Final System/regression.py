#! usr/bin/python
# coding=utf-8

from evaluation_metrics import *
from preprocess import *
from feature_extraction import *
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

class Regression:

    def __init__(self, tfidf = False, BoW = False, edinburgh = False, glove = False, Hashtag_Intense = False, Lexicons = False):
        SVM = SVR()
        XGboost = GradientBoostingRegressor(n_estimators=200)
        MLP = MLPRegressor(hidden_layer_sizes=[100, 50], activation='logistic')

        svm_pearson_coef, svm_spearman_coef = [], []
        boost_pearson_coef, boost_spearman_coef = [], []
        mlp_pearson_coef, mlp_spearman_coef = [], []

        for _emotion in ['anger', 'fear', 'joy', 'sadness']:
            print('')
            print('Emotion:', _emotion)

            # load training dataset
            train_x, train_y = self.load_2018_reg(emotion = _emotion)

            # use joy's training data for sadness
            if _emotion=='sadness' :
                train_joy_x, train_joy_y = self.load_2018_reg(emotion = 'joy')
                for i in range(len(train_joy_y)):
                    train_joy_y[i] = 1 - train_joy_y[i]
                train_x = train_x + train_joy_x
                train_y = train_y + train_joy_y

            train_x_changed = False
            # load features

            if tfidf:
                train_tfidf = np.loadtxt('./data/Features_reg/tfidf/' + _emotion + '.txt')
                if train_x_changed:
                    train_x = np.concatenate((train_x, train_tfidf), axis = 1)
                else:
                    train_x = train_tfidf
                    train_x_changed = True

            if BoW:
                train_BoW = np.loadtxt('./data/Features_reg/BoW/' + _emotion + '.txt')
                if train_x_changed:
                    train_x = np.concatenate((train_x, train_BoW), axis = 1)
                else:
                    train_x = train_BoW
                    train_x_changed = True

            if edinburgh:
                train_edinburgh = np.loadtxt('./data/Features_reg/edinburgh/' + _emotion + '.txt')
                if train_x_changed:
                    train_x = np.concatenate((train_x, train_edinburgh), axis = 1)
                else:
                    train_x = train_edinburgh
                    train_x_changed = True

            if glove:
                train_glove = np.loadtxt('./data/Features_reg/glove/' + _emotion + '.txt')
                if train_x_changed:
                    train_x = np.concatenate((train_x, train_glove), axis = 1)
                else:
                    train_x = train_glove
                    train_x_changed = True

            if Hashtag_Intense:
                hashtag_intense = np.loadtxt('./data/Features_reg/hashtag_intense/' + _emotion + '.txt')
                if train_x_changed:
                    train_x = np.concatenate((train_x, hashtag_intense), axis = 1)
                else:
                    train_x = hashtag_intense
                    train_x_changed = True

            if Lexicons:
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
                if train_x_changed:
                    train_x = np.concatenate((train_x, All_Lexicons), axis = 1)
                else:
                    train_x = All_Lexicons
                    train_x_changed = True

            # print ('training data has', len(train_x), 'samples', len(train_x[1]), 'dims')

            print ('SVM regressor')
            svm_pearson_coef, svm_spearman_coef = self.ten_fold_cross_validation(train_x, train_y, SVM, svm_pearson_coef, svm_spearman_coef)

            print ('XGBoost regressor')
            boost_pearson_coef, boost_spearman_coef = self.ten_fold_cross_validation(train_x, train_y, XGboost, boost_pearson_coef, boost_spearman_coef)

            print ('MLP regressor')
            mlp_pearson_coef, mlp_spearman_coef = self.ten_fold_cross_validation(train_x, train_y, MLP, mlp_pearson_coef, mlp_spearman_coef)

        self.print_evaluations(svm_pearson_coef, svm_spearman_coef, boost_pearson_coef, boost_spearman_coef, mlp_pearson_coef, mlp_spearman_coef)


    def ten_fold_cross_validation(self, train_x, train_y, regressor, pearson, spearman):
        kf = KFold(n_splits=10, random_state=2, shuffle=True)
        folds = kf.split(train_x, train_y)
        kfold = 0
        Pearson = []
        Spearman = []
        for idx1, idx2 in folds:
            kfold += 1
            # training data
            training_input = [train_x[i] for i in idx1]
            training_output = [train_y[i] for i in idx1]
            # test data
            test_input = [train_x[i] for i in idx2]
            test_output = [train_y[i] for i in idx2]
            regressor.fit(training_input, training_output)
            output = measure_reg(test_output, regressor.predict(test_input))
            Pearson.append(output[0])
            Spearman.append(output[1])
            print('** ', kfold, 'fold: pearson coef = ', Pearson[-1])
        pearson.append(Pearson)
        spearman.append(Spearman)
        return pearson, spearman

    def print_evaluations(self, svm_pearson_coef, boost_pearson_coef, mlp_pearson_coef, svm_spearman_coef, boost_spearman_coef, mlp_spearman_coef):
        print(' ' * 10, '   SVM  ', '    XGBoost', '  MLP')
        for i, _emotion in enumerate(['anger', 'fear', 'joy', 'sadness']):
            print('%10s%10.4f%10.4f%10.4f' % (_emotion, np.mean(svm_pearson_coef[i]), np.mean(boost_pearson_coef[i]), np.mean(mlp_pearson_coef[i])))
        print('Evaluation metric: Pearson correlation \n')

        print(' ' * 10, '   SVM  ', '    XGBoost', '  MLP')
        for i, _emotion in enumerate(['anger', 'fear', 'joy', 'sadness']):
            print('%10s%10.4f%10.4f%10.4f' % (_emotion, np.mean(svm_spearman_coef[i]), np.mean(boost_spearman_coef[i]), np.mean(mlp_spearman_coef[i])))
        print('Evaluation metric: Spearman correlation \n ')


    def load_2017_reg(self, path='./data/2017train', emotion='sadness'):
        for f in os.listdir(path):
            if f.find(emotion) >= 0:
                text = [l.split('\t')[1:]
                        for l in open(os.path.join(path, f)).readlines()]
                break
        x, y = [t[0] for t in text], [float(t[2]) for t in text]
        return x, y


    def load_2018_reg(self, path='./data/EI-reg-En-train', emotion='sadness'):
        for f in os.listdir(path):
            if f.find(emotion) >= 0 and f.find('_re_') < 0:
                text = [l.split('\t')[1:]
                        for l in open(os.path.join(path, f)).readlines()]
                break
        text = text[1:]
        x, y = [t[0] for t in text], [float(t[2]) for t in text]
        return x, y


    def load_2018_oc(self, path='./data/EI-oc-En-train', emotion='sadness'):
        for f in os.listdir(path):
            if f.find(emotion) >= 0:
                text = [l.split('\t')[1:]
                        for l in open(os.path.join(path, f)).readlines()]
                break
        text = text[1:]
        x = [t[0] for t in text]
        y = [int(t[2].split(':')[0]) for t in text]
        return x, y



