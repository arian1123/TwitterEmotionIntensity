
import pickle
from utils import *
from preprocess import *
from feature_factory import *
from lexicons import *
from sentistrength import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
import numpy as np
from nltk.tokenize import TweetTokenizer
from copy import deepcopy
from sklearn.decomposition import PCA



def build_dict_from_corpus(x, min_freq):
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

def lexicon_feature(x, dictionary):
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


def sum_of_word_embedding(x, path_to_embed_pkl='./embedding/embed.pkl', dim=50):
    data = pickle.load(open(path_to_embed_pkl, 'rb'))
    vocab = data['word']
    embed = data['vector']
    nx = np.zeros((len(x), dim))
    
    for i, _x in enumerate(x):
        for _w in _x.split():
            if _w in vocab:
                nx[i] += embed[vocab.index(_w), :]
            else:
                nx[i] += embed[vocab.index('unknown'), :]
    return nx

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

def write_to_file(emotion_name, feature_name, feature, lexicon):
    if lexicon:
        file_name = './data/Features_oc/Lexicons/' + feature_name + '/' + emotion_name + '.txt'
    else:
        file_name = './data/Features_oc/' + feature_name + '/' + emotion_name + '.txt'
    np.savetxt(file_name, feature)


if __name__ == '__main__':

    # SVM = SVR()
    # XGboost = GradientBoostingRegressor(n_estimators=200)
    # MLP = MLPRegressor(hidden_layer_sizes=[100, 50], activation='logistic')

    tokenizer = TweetTokenizer()
    edinburg_embeddings = open("./embedding/w2v.twitter.edinburgh10M.400d.csv").readlines()
    word_list = [line.split('\t')[-1].strip() for line in edinburg_embeddings]

    glove_embeddings = open("./embedding/glove.twitter.27B/glove.twitter.27B.200d.txt").readlines()
    glove_word_list = [line.split(' ')[0].strip() for line in glove_embeddings]

    for _emotion in ['anger', 'fear', 'joy', 'sadness']:
        # if _emotion == 'fear':
        #     continue
        print ('')
        print ('Emotion:', _emotion)
        train_x, train_y = load_original_oc(emotion = _emotion)

        # # use joy's training data for sadness
        # if _emotion=='sadness' :
        #     train_joy_x, train_joy_y = load_original_reg(emotion = 'joy')
        #     for i in range(len(train_joy_y)):
        #         train_joy_y[i] = 1 - train_joy_y[i]
        #     train_x = train_x + train_joy_x
        #     train_y = train_y + train_joy_y

        # preprocessing
        for i in range(len(train_x)):
            train_x[i] = regular_tweet(train_x[i])

        #tf-idf
        vectorizer = TfidfVectorizer(min_df=3)
        vectorizer.fit(train_x)
        train_tfidf = vectorizer.transform(train_x).todense()
        train_tfidf = train_tfidf.tolist()
        print(len(train_tfidf[0]))
        #
        #bag of words
        cdict = build_dict_from_corpus(train_x, min_freq=5)
        train_BoW = lexicon_feature(train_x, cdict)
        print(len(train_BoW[0]))

        # w2v
        #edingurgh embedding
        train_edinburgh = deepcopy(train_x)
        for i in range(len(train_x)):
            train_edinburgh[i] = tweetToEdinburg(train_x[i])
        # normalize
        # train_edinburgh = preprocessing.normalize(train_edinburgh, norm='l2')



        # glove embedding
        train_glove = deepcopy(train_x)
        for i in range(len(train_x)):
            train_glove[i] = tweetToGlove(train_x[i])
        # # normalize
        # train_glove = preprocessing.normalize(train_glove, norm='l2')

        # # initialize lexicons
        # Emoji = deepcopy(train_x)
        # AFINN = deepcopy(train_x)
        # BingLiu = deepcopy(train_x)
        # MPQA = deepcopy(train_x)
        # NRC_EmoLex = deepcopy(train_x)
        # NRC10E = deepcopy(train_x)
        # NRC_Hash_Emo = deepcopy(train_x)
        # NRC_Hash_Sent = deepcopy(train_x)
        # Sentiment140 = deepcopy(train_x)
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
        #     NRC_EmoLex[i] = tweetToEmoLexVector(train_x[i], _emotion)
        #     train_x[i] = tmp
        #     tmp = deepcopy(train_x[i])
        #
        #     NRC10E[i] = tweetToEmo10EVector(train_x[i], _emotion)
        #     train_x[i] = tmp
        #     tmp = deepcopy(train_x[i])
        #
        #     NRC_Hash_Emo[i] = tweetToHSEVector(train_x[i], _emotion)
        #     train_x[i] = tmp
        #     tmp = deepcopy(train_x[i])
        #
        #     NRC_Hash_Sent[i] = tweetToHSVector(train_x[i])
        #     train_x[i] = tmp
        #     tmp = deepcopy(train_x[i])
        #
        #     Sentiment140[i] = tweetToSentiment140Vector(train_x[i])
        #     train_x[i] = tmp
        #     tmp = deepcopy(train_x[i])
        #
        #     SentiStrength[i] = sentistrength.tweetToSentiStrength(train_x[i])
        #     train_x[i] = tmp
        #     tmp = deepcopy(train_x[i])


        #print(len(train_x0[0]), ' and ', len(train_x1[0]), ' and ', len(train_x2[0]))
        #print(train_x0[0], ' and ', train_x1[0], ' and ', train_x2[0])
        # reduce the dimention to 140 since each tweet has a maximum length of 140.
        print('finish generating features, next reduce dims by PCA')
        # pca_pre = PCA(n_components = 5000)
        # pca = PCA(n_components = 140)

        # train_tfidf = pca.fit_transform(train_tfidf)
        # train_BoW = pca.fit_transform(train_BoW)
        # train_edinburgh = pca.fit_transform(train_edinburgh)
        # train_glove = pca.fit_transform(train_glove)


        # Emoji = pca.fit_transform(Emoji)
        # AFINN = pca.fit_transform(AFINN)
        # BingLiu = pca.fit_transform(BingLiu)
        # MPQA = pca.fit_transform(MPQA)
        # NRC_EmoLex = pca.fit_transform(NRC_EmoLex)
        # NRC10E = pca.fit_transform(NRC10E)
        # NRC_Hash_Emo = pca.fit_transform(NRC_Hash_Emo)
        #
        # print(len(NRC_Hash_Sent[0])) # dim 54129
        # NRC_Hash_Sent = pca_pre.fit_transform(NRC_Hash_Sent)
        # print('good in the first pca')
        # NRC_Hash_Sent = pca.fit_transform(NRC_Hash_Sent)
        #
        #
        # print(len(Sentiment140[0])) # dim 62468
        # Sentiment140 = pca_pre.fit_transform(Sentiment140)
        # print('good in the first pca')
        # Sentiment140 = pca.fit_transform(Sentiment140)
        #
        # print(len(SentiStrength[0])) # dim 2700
        # SentiStrength = pca.fit_transform(SentiStrength)


        # # print(train_tfidf[0])
        # # print(train_edinburgh[0])
        # # print(NRC_Hash_Emo[0])

        print('finish PCA, start to write to files')
        write_to_file(_emotion, 'tfidf', train_tfidf, False)
        write_to_file(_emotion, 'BoW', train_BoW, False)
        write_to_file(_emotion, 'edinburgh', train_edinburgh, False)
        write_to_file(_emotion, 'glove', train_glove, False)

        # write_to_file(_emotion, 'Emoji', Emoji, False)
        #
        # write_to_file(_emotion, 'AFINN', AFINN, True)
        # write_to_file(_emotion, 'BingLiu', BingLiu, True)
        # write_to_file(_emotion, 'MPQA', MPQA, True)
        # write_to_file(_emotion, 'NRC_EmoLex', NRC_EmoLex, True)
        # write_to_file(_emotion, 'NRC10E', NRC10E, True)
        # write_to_file(_emotion, 'NRC_Hash_Emo', NRC_Hash_Emo, True)
        # write_to_file(_emotion, 'NRC_Hash_Sent', NRC_Hash_Sent, True)
        # write_to_file(_emotion, 'Sentiment140', Sentiment140, True)
        # write_to_file(_emotion, 'SentiStrength', SentiStrength, True)

        print ('Done writing to files, next check total dimentions')
        # print ('Total dims are', (len(Emoji[0]) + len(AFINN[0]) + len(BingLiu[0]) + len(MPQA[0]) + len(NRC_EmoLex[0])
        #                           + len(NRC10E[0]) + len(NRC_Hash_Emo[0]) + len(NRC_Hash_Sent[0]) + len(NRC_Hash_Sent[0])
        #                           + len(Sentiment140[0]) + len(SentiStrength[0]))

    # test = np.loadtxt('./data/Features_reg/Lexicons/Sentiment140/anger.txt')
    # print(test[0])



