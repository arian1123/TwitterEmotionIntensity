# coding=utf-8
"""
Created on November 20 2017

@author: Jingshi & Arian
"""
import preprocess
import evaluation_metrics
import feature_extraction
import regression
import classification
import numpy as np

def main(stage1 = False, stage2 = False, stage3 = False, classify = False, year = 2018, tfidf=False, BoW=False, edinburgh=False, glove=False, Hashtag_Intense=False, Lexicons=False):
    '''
    This is the main funcion that integrates all stages and methods of our system.
    Each of the three stages are commented as following.
    '''
    # Stage 1: Preprocessing
    if stage1:
        preprocessing_extract_emojis(classify = classify, year = year)
        preprocessing_map_emojis(classify = classify, year = year)

    # Stage 2: Featrue Extraction
    if stage2:
        extract_features(classify = classify, year = year)

    # Stage 3: Regression or Classification
    if stage3:
        if classify:
            run_classification(tfidf=tfidf, BoW=BoW, edinburgh=edinburgh, glove=glove, Hashtag_Intense=Hashtag_Intense, Lexicons=Lexicons)

        else:
            run_regression(year = year, tfidf=tfidf, BoW=BoW, edinburgh=edinburgh, glove=glove, Hashtag_Intense=Hashtag_Intense, Lexicons=Lexicons)

def preprocessing_extract_emojis(classify = False, year = 2018):
    '''
    This funcion runs part of the preprocessing stage.
    It extracts all the emojis in training dataset,
    save them in text.txt, manually delete characters from other languages
    (such as Japanese, Chinese, Arabian, etc.) and illegal forms,
    and save them as a file named emoji.txt.
    '''
    preprocess.Preprocessor(define_emoji= True, year = year, classify= classify)

def preprocessing_map_emojis(classify = False, year = 2018):
    '''
    This funcion runs part of the preprocessing stage.
    It maps each emoji in emoji.txt to a unique string (e.g. map 😄 to 'emoji12')
    and save these unique strings to a file named emoji_lexicon.txt.
    '''
    preprocess.Preprocessor(emoji_to_lexicon= True, year = year, classify= classify)

def extract_features(classify = False, year = 2018):
    '''
    This funcion runs the feature extraction stage.
    It extracts all the features and save them to a file under 'data' folder.
    Note the row tweets are preprocessed by Preprocessor in line 52.
    '''
    for _emotion in ['anger', 'fear', 'joy', 'sadness']:
        print ('')
        print ('Emotion:', _emotion)
        my_preprocessor = preprocess.Preprocessor(emotion = _emotion, classify = classify, year = year)
        my_features = feature_extraction.TweetFeatureGenerator(my_preprocessor, emotion = _emotion)

def run_regression(year = 2018, tfidf=True, BoW=True, edinburgh=True, glove=True, Hashtag_Intense=True, Lexicons=True):
    '''
    This funcion runs the regression.
    It reads features from the file Features_reg under 'data' folder.
    Performs 10 - fold cross validation on training dataset with three following regressors:
    1) Support vector machine (regressor) of sklearn.
    2) Multi-layer Perceptron (regressor) of sklearn.
    3)Gradient Boosting (regressor) of sklearn.

    Then, print out the evaluations for each regressor and for each emotion.
    The evaluation metrics are pearson correlation and spearman correlation.
    '''
    regression.Regression(year = year, tfidf = tfidf, BoW = BoW, edinburgh = edinburgh, glove = glove, Hashtag_Intense = Hashtag_Intense, Lexicons = Lexicons)

def run_classification(tfidf=True, BoW=True, edinburgh=True, glove=True, Hashtag_Intense=True, Lexicons=True):
    '''
    This funcion runs the classification.
    It reads features from the file Features_oc under 'data' folder.
    Performs 10 - fold cross validation on training dataset with three following classifiers:
    1) Support vector machine (classifier) of sklearn.
    2) Multi-layer Perceptron (classifier) of sklearn.
    3)Gradient Boosting (classifier) of sklearn.

    Then, print out the evaluations for each classifier and for each emotion.
    The evaluation metrics is pearson correlation.
    '''
    classification.Classification(tfidf = tfidf, BoW = BoW, edinburgh = edinburgh, glove = glove, Hashtag_Intense = Hashtag_Intense, Lexicons = Lexicons)



if __name__ == "__main__":
    # To run the system, specify which stage you want to run by assigning True or False values to the states (i.e. stage1, stage2, and stage3).
    # You also need to specify whether you want to run regression or classification by assign True or False to classify.
    # Specify year of the data you want to use by assign values to year.
    # Then, you need to specify the features you want to use by assigning True or False to the features.
    # Note the features are tfidf, BoW (bag of words), edinburgh (Edinburgh embeddings), glove (GloVe embeddings),
    # Hashtag_Intense (Hashtag intensity), Lexicons (Affect lexicon features, which is the combination of all affect lexicon features)
    main(stage1 = True, stage2 = True, stage3 = True, classify = False, year = 2018, tfidf=True, BoW=True, edinburgh=True, glove=True, Hashtag_Intense=True, Lexicons=True)
