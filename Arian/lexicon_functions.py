import numpy as np
from nltk.tokenize import TweetTokenizer
from nltk.corpus import sentiwordnet as swn # sentiwordnet lexicon

### Lexicons ###
emo10e = open("Arian/data/Lexicons/uni-bwn-pos-dp-BCC-Lex.csv").readlines()
hashtag_senti = open("Arian/data/Lexicons/Lexicons/NRC-Hashtag-Sentiment-Lexicon-v1.0/HS-unigrams.txt", "r").readlines()
emolex = open("Arian/data/Lexicons/Lexicons/NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt").readlines()
hashtag_emo = open("Arian/data/Lexicons/Lexicons/NRC-Hashtag-Emotion-Lexicon-v0.2/NRC-Hashtag-Emotion-Lexicon-v0.2.txt").readlines()
sentiment140 = open("Arian/data/Lexicons/Lexicons/NRC-Emoticon-Lexicon-v1.0/Emoticon-unigrams.txt").readlines() #(Sentiment 140)
# TODO: Negations are made in the following lexicon
# hastag_senti_afflexneglex = open("data/raw/Lexicons/Lexicons/NRC-Hashtag-Sentiment-AffLexNegLex-v1.0/HS-AFFLEX-NEGLEX-unigrams.txt").readlines()
# TODO: Add All NRC Lexicons
bingliu_pos = open("Arian/data/Lexicons/BingLiu/BingLiu_positive-words.txt").readlines()
bingliu_neg = open("Arian/data/Lexicons/BingLiu/BingLiu_negative-words.txt").readlines()
mpqa = open("Arian/data/Lexicons/MPQA/MPQA.tff").readlines()
afinn = open("Arian/data/Lexicons/AFINN/afinn.txt").readlines()
emoji  = open("Arian/data/emoji_lexicon.txt").readlines()

### Create Tokenizer object ###
tokenizer = TweetTokenizer()

### Transforming the tweet into many vectors ###
def tweetToSWNVector(tweet):
    tokens = tokenizer.tokenize(tweet)
    vec = np.zeros(3)
    pos_score, neg_score, obj_score = 0, 0, 0
    for token in tokens:
        l = list(swn.senti_synsets(token))
        if len(l) == 0:
            continue
        else: # if the word exists
            pos_score += l[0].pos_score()
            neg_score -= l[0].neg_score()
            obj_score += l[0].obj_score()
    vec[0], vec[1], vec[2] = pos_score, neg_score, obj_score
    return vec


def tweetToAFINNVector(tweet):
    vec = np.zeros(len(afinn))
    tokens = tokenizer.tokenize(tweet)

    for i, line in enumerate(afinn):
        if line.split('\t')[0] in tokens:
            vec[i] = float(line.split('\t')[1])
    return vec

def tweetToEmo10EVector(tweet, emotion):
    vec = np.zeros(len(emo10e)-1)
    tokens = tokenizer.tokenize(tweet)
    emotion_index = emo10e[0].split('\t').index(emotion)
    for i in range(0, len(emo10e)):
        if i == 0:
            continue
        else:
            if emo10e[i].split('\t')[0] in tokens:
                vec[i] = emo10e[i].split('\t')[emotion_index]
    return vec

def tweetToHSVector(tweet):
    vec = np.zeros(len(hashtag_senti))
    tokens = tokenizer.tokenize(tweet)
    for i, line in enumerate(hashtag_senti):
        if line.split('\t')[0] in tokens:
            vec[i] = float(line.split('\t')[1])
    return vec

def tweetToEmoLexVector(tweet, emotion):
    vec = np.zeros(14182)  # for each individual emotion
    tokens = tokenizer.tokenize(tweet)
    item = 0
    for line in emolex:
        if len(line) <= 1:
            continue
        #print(len(line))
        if line.split('\t')[1] == emotion:
            if line.split('\t')[0] in tokens:
                vec[item] = int(line.split('\t')[2])
            item += 1
    return vec

def tweetToHSEVector(tweet, emotion):
    vec = np.zeros(6000)
    tokens = tokenizer.tokenize(tweet)
    item = 0
    corr = False # Reached correct emotion
    for line in hashtag_emo:
        l = line.split('\t')
        if l[0] == emotion:
            if not corr: # If first time reaching correct emotion
                corr = True
            if l[1] in tokens:
                vec[item] = float(line.split('\t')[2])
            item += 1
        else:
            if corr: # If you have been on the correct emotion
                break
    global hse_len
    hse_len = len(vec[:item])
    return vec[:item]

def tweetToSentiment140Vector(tweet):
    vec = np.zeros(len(sentiment140))
    tokens = tokenizer.tokenize(tweet)
    for i, line in enumerate(sentiment140):
        if line.split('\t')[0] in tokens:
            vec[i] = float(line.split('\t')[1])
    return vec

def tweetToMPQAVector(tweet):
    vec = np.zeros(len(mpqa))
    tokens = tokenizer.tokenize(tweet)
    for i, line in enumerate(mpqa):
        l = line.split(" ")
        word = l[2].split("=")[1]
        if word in tokens:
            polarity = l[5].split("=")[1].strip()
            if polarity == "negative":
                vec[i] = -1
            elif polarity == "positive":
                vec[i] = 1
    return vec

def tweetToBingLiuVector(tweet):
    vec = np.zeros(len(bingliu_neg)+len(bingliu_pos))
    tokens = tokenizer.tokenize(tweet)
    neg_len = len(bingliu_neg) # TODO: come up with better variable name
    for i, line in enumerate(bingliu_neg):
        if line.strip() in tokens:
            vec[i] = -1
    for i, line in enumerate(bingliu_pos):
        if line.strip() in tokens:
            vec[i+neg_len] = 1
    return vec

def tweetToEmoji(tweet):
    vec = np.zeros(len(emoji))
    tokens = tweet.split(' ')
    for i, line in enumerate(emoji):
        if line.strip() in tokens:
            vec[i] += 1
    return vec

### Combine all the vectors ###
def tweetToSparseLexVector(tweet, emotion): # to create the final vector
    args = (tweetToSWNVector(tweet), tweetToSentiStrength(tweet), tweetToEmo10EVector(tweet, emotion), tweetToHSVector(tweet), tweetToEmoLexVector(tweet, emotion), tweetToHSEVector(tweet, emotion), tweetToSentiment140Vector(tweet), tweetToMPQAVector(tweet), tweetToBingLiuVector(tweet), tweetToAFINNVector(tweet))
    return np.concatenate(args)

### Total length of the vector ###
def getLength():
    return 3 + getSentiStrengthLength() + len(emo10e)-1 + len(hashtag_senti) + 14182 + hse_len + len(sentiment140) + len(mpqa) + len(bingliu_pos) + len(bingliu_neg) + len(afinn)


#
#
# SentiStrength
#
#
#

# Files
booster = open("Arian/data/Lexicons/SentiStrength/BoosterWordList.txt").readlines()
emoticon_table = open("Arian/data/Lexicons/SentiStrength/EmoticonLookupTable.txt").readlines()
emotion_table = open("Arian/data/Lexicons/SentiStrength/EmotionLookupTable.txt").readlines()
idioms = open("Arian/data/Lexicons/SentiStrength/IdiomLookupTable.txt").readlines()

def getIdiomVec(tweet):
    vec = np.zeros(len(idioms))
    for i, line in enumerate(idioms):
        if line.split('\t')[0] in tweet:
            vec[i] = line.split('\t')[1].strip()
    return vec

def getEmoticonVec(tweet):
    vec = np.zeros(len(emoticon_table))
    tokens = tokenizer.tokenize(tweet)
    for i, line in enumerate(emoticon_table):
        if line.split('\t')[0] in tokens:
            vec[i] = line.split('\t')[1].strip()
    return vec

def getEmotionVec(tweet):
    vec = np.zeros(len(emotion_table))
    tokens = tokenizer.tokenize(tweet)
    for i, line in enumerate(emotion_table):
        for token in tokens:
            if line.split('\t')[0].split("*")[0] in token:
                vec[i] = line.split('\t')[1].strip()
                break
    return vec

def getBoosterVec(tweet):
    vec = np.zeros(len(booster))
    tokens = tokenizer.tokenize(tweet)
    for i, line in enumerate(booster):
        if line.split('\t')[0] in tokens:
            vec[i] = line.split('\t')[1].strip()
    return vec

def tweetToSentiStrength(tweet):
    idiom_vec = getIdiomVec(tweet)
    emotion_vec = getEmotionVec(tweet)
    emoticon_vec = getEmoticonVec(tweet)
    booster_vec = getBoosterVec(tweet)
    args = (idiom_vec, emotion_vec, emoticon_vec, booster_vec)
    return np.concatenate(args)

def getSentiStrengthLength():
    return len(booster) + len(emoticon_table) + len(emotion_table) + len(idioms)