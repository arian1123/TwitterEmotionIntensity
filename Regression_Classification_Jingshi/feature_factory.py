
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

def bag_of_words(train_x):
    '''
    bag of words feature
    '''
    cdict = build_dict_from_corpus(train_x, min_freq=5)
    train_BoW = lexicon_feature(train_x, cdict)
    return train_BoW

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

def tfidf(train_x):
    '''
    tf-idf feature
    '''
    vectorizer = TfidfVectorizer(min_df=3)
    vectorizer.fit(train_x)
    train_tfidf = vectorizer.transform(train_x).todense()
    train_tfidf = train_tfidf.tolist()
    return train_tfidf

def edinburgh(train_x, is_normalize):
    '''
    edinburgh embedding feature
    '''
    train_edinburgh = deepcopy(train_x)
    for i in range(len(train_x)):
        train_edinburgh[i] = tweetToEdinburgh(train_x[i])
    # normalize
    if is_normalize:
        train_edinburgh = preprocessing.normalize(train_edinburgh, norm='l2')
    return train_edinburgh

def tweetToEdinburgh(tweet):
    '''
    edinburgh embedding for a single tweet
    '''
    tokens = tokenizer.tokenize(tweet)
    tweet_vec = [np.zeros(400) for _ in range(len(tokens))]
    for i, token in enumerate(tokens):
        if token in word_list:
            tweet_vec[i] = edinburgh_embeddings[word_list.index(token)].split('\t')[:-1]
    return vectorsToVector(tweet_vec, 400)

def glove(train_x, is_normalize):
    '''
    glove embedding feature
    '''
    train_glove = deepcopy(train_x)
    for i in range(len(train_x)):
        train_glove[i] = tweetToGlove(train_x[i])
    # normalize
    if is_normalize:
        train_glove = preprocessing.normalize(train_glove, norm='l2')
    return train_glove

def tweetToGlove(tweet):
    '''
    glove embedding for a single tweet
    '''
    glove_tokens = tokenizer.tokenize(tweet)
    tweet_vec = [np.zeros(200) for _ in range(len(glove_tokens))]
    for i, token in enumerate(glove_tokens):
        if token in glove_word_list:
            tweet_vec[i] = glove_embeddings[glove_word_list.index(token)].split(' ')[1:]
    return vectorsToVector(tweet_vec, 200)

def vectorsToVector(arr, dim):
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

def Emoji_feature(train_x):
    '''
    Emoji feature
    '''
    Emoji = deepcopy(train_x)
    for i in range(len(train_x)):
        tmp = deepcopy(train_x[i])
        Emoji[i] = tweetToEmoji(train_x[i])
        train_x[i] = tmp
    return Emoji

def AFINN_feature(train_x):
    '''
    AFINN feature
    '''
    AFINN = deepcopy(train_x)
    for i in range(len(train_x)):
        tmp = deepcopy(train_x[i])
        AFINN[i] = tweetToAFINNVector(train_x[i])
        train_x[i] = tmp
    return AFINN

def BingLiu_feature(train_x):
    '''
    BingLiu feature
    '''
    BingLiu = deepcopy(train_x)
    for i in range(len(train_x)):
        tmp = deepcopy(train_x[i])
        BingLiu[i] = tweetToBingLiuVector(train_x[i])
        train_x[i] = tmp
    return BingLiu

def MPQA_feature(train_x):
    '''
    MPQA feature
    '''
    MPQA = deepcopy(train_x)
    for i in range(len(train_x)):
        tmp = deepcopy(train_x[i])
        MPQA[i] = tweetToMPQAVector(train_x[i])
        train_x[i] = tmp
    return MPQA

def NRC_EmoLex_feature(train_x, emotion):
    '''
    NRC Emolex feature
    '''
    NRC_EmoLex = deepcopy(train_x)
    for i in range(len(train_x)):
        tmp = deepcopy(train_x[i])
        NRC_EmoLex[i] = tweetToEmoLexVector(train_x[i], emotion)
        train_x[i] = tmp
    return NRC_EmoLex

def NRC10E_feature(train_x, emotion):
    '''
    NRC10E feature
    '''
    NRC10E = deepcopy(train_x)
    for i in range(len(train_x)):
        tmp = deepcopy(train_x[i])
        NRC10E[i] = tweetToEmo10EVector(train_x[i], emotion)
        train_x[i] = tmp
    return NRC10E


def NRC_Hash_Emo_feature(train_x, emotion):
    '''
    NRC Hash Emo feature
    '''
    NRC_Hash_Emo = deepcopy(train_x)
    for i in range(len(train_x)):
        tmp = deepcopy(train_x[i])
        NRC_Hash_Emo[i] = tweetToHSEVector(train_x[i], emotion)
        train_x[i] = tmp
    return NRC_Hash_Emo

def NRC_Hash_Sent_feature(train_x):
    '''
    NRC Hash Sent feature
    '''
    NRC_Hash_Sent = deepcopy(train_x)
    for i in range(len(train_x)):
        tmp = deepcopy(train_x[i])
        NRC_Hash_Sent[i] = tweetToHSVector(train_x[i])
        train_x[i] = tmp
    return NRC_Hash_Sent

def Sentiment140_feature(train_x):
    '''
    Sentiment140 feature
    '''
    Sentiment140 = deepcopy(train_x)
    for i in range(len(train_x)):
        tmp = deepcopy(train_x[i])
        Sentiment140[i] = tweetToSentiment140Vector(train_x[i])
        train_x[i] = tmp
    return Sentiment140

def SentiStrength_feature(train_x):
    '''
    SentiStrength feature
    '''
    SentiStrength = deepcopy(train_x)
    for i in range(len(train_x)):
        tmp = deepcopy(train_x[i])
        SentiStrength[i] = sentistrength.tweetToSentiStrength(train_x[i])
        train_x[i] = tmp
    return SentiStrength

def reduce_dimention(feature):
    '''
    reduce dimention of a feature, by using PCA
    '''
    pca_pre = PCA(n_components=5000)
    pca = PCA(n_components=140)
    if (len(feature[0]) < 50000):
        feature = pca.fit_transform(feature)
    if (len(feature[0]) >= 50000):
        feature = pca_pre.fit_transform(feature)
        feature = pca.fit_transform(feature)
    return feature

def write_to_file(emotion_name, feature_name, feature, lexicon):
    '''
    save the features
    '''
    if lexicon:
        file_name = './data/Features_reg/Lexicons/' + feature_name + '/' + emotion_name + '.txt'
    else:
        file_name = './data/Features_reg/' + feature_name + '/' + emotion_name + '.txt'
    np.savetxt(file_name, feature)

if __name__ == '__main__':

    tokenizer = TweetTokenizer()

    # load Edinburgh word vectors
    edinburgh_embeddings = open("./embedding/w2v.twitter.edinburgh10M.400d.csv").readlines()
    word_list = [line.split('\t')[-1].strip() for line in edinburgh_embeddings]

    # load GloVe word vectors
    glove_embeddings = open("./embedding/glove.twitter.27B/glove.twitter.27B.200d.txt").readlines()
    glove_word_list = [line.split(' ')[0].strip() for line in glove_embeddings]

    for _emotion in ['anger', 'fear', 'joy', 'sadness']:
        print ('')
        print ('Emotion:', _emotion)
        # load training dataset
        train_x, train_y = load_2018_reg(emotion = _emotion)

        # use joy's training data for sadness
        if _emotion=='sadness' :
            train_joy_x, train_joy_y = load_2018_reg(emotion = 'joy')
            for i in range(len(train_joy_y)):
                train_joy_y[i] = 1 - train_joy_y[i]
            train_x = train_x + train_joy_x
            train_y = train_y + train_joy_y

        # preprocessing
        for i in range(len(train_x)):
            train_x[i] = regular_tweet(train_x[i])

        # Extracting features
        train_tfidf = tfidf(train_x)
        train_BoW = bag_of_words(train_x)
        train_edinburgh = edinburgh(train_x, False)
        train_glove = glove(train_x, False)
        Emoji = Emoji_feature(train_x)
        AFINN = AFINN_feature(train_x)
        BingLiu = BingLiu_feature(train_x)
        MPQA = MPQA_feature(train_x)
        NRC_EmoLex = NRC_EmoLex_feature(train_x, _emotion)
        NRC10E = NRC10E_feature(train_x, _emotion)
        NRC_Hash_Emo = NRC_Hash_Emo_feature(train_x, _emotion)
        NRC_Hash_Sent = NRC_Hash_Sent_feature(train_x)
        Sentiment140 = Sentiment140_feature(train_x)
        SentiStrength = SentiStrength_feature(train_x)

        # reduce the dimention to 140 based on a huristic that each tweet has a maximum length of 140.
        Emoji = reduce_dimention(Emoji)
        AFINN = reduce_dimention(AFINN)
        BingLiu = reduce_dimention(BingLiu)
        MPQA = reduce_dimention(MPQA)
        NRC_EmoLex = reduce_dimention(NRC_EmoLex)
        NRC10E = reduce_dimention(NRC10E)
        NRC_Hash_Emo = reduce_dimention(NRC_Hash_Emo)
        NRC_Hash_Sent = reduce_dimention(NRC_Hash_Sent)
        Sentiment140 = reduce_dimention(Sentiment140)
        SentiStrength = reduce_dimention(SentiStrength)

        # save the features to files
        write_to_file(_emotion, 'tfidf', train_tfidf, False)
        write_to_file(_emotion, 'BoW', train_BoW, False)
        write_to_file(_emotion, 'edinburgh', train_edinburgh, False)
        write_to_file(_emotion, 'glove', train_glove, False)
        write_to_file(_emotion, 'Emoji', Emoji, False)
        write_to_file(_emotion, 'AFINN', AFINN, True)
        write_to_file(_emotion, 'BingLiu', BingLiu, True)
        write_to_file(_emotion, 'MPQA', MPQA, True)
        write_to_file(_emotion, 'NRC_EmoLex', NRC_EmoLex, True)
        write_to_file(_emotion, 'NRC10E', NRC10E, True)
        write_to_file(_emotion, 'NRC_Hash_Emo', NRC_Hash_Emo, True)
        write_to_file(_emotion, 'NRC_Hash_Sent', NRC_Hash_Sent, True)
        write_to_file(_emotion, 'Sentiment140', Sentiment140, True)
        write_to_file(_emotion, 'SentiStrength', SentiStrength, True)




