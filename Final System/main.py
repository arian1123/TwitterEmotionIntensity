import preprocess
import evaluation_metrics
import feature_extraction
import regression
import classification
import numpy as np

def main():
    '''
    This is the main funcion that integrates all stages and methods of our system.
    Each of the three stages are commented as following.
    To run a single stage, you can comment out the other two stages.
    '''
    # Stage 1: Preprocessing
    preprocessing_extract_emojis()
    preprocessing_map_emojis()

    # Stage 2: Featrue Extraction
    extract_features()

    # Stage 3: Regression or Classification
    run_regression(tfidf=True, BoW=True, edinburgh=True, glove=True, Hashtag_Intense=True, Lexicons=True)
    run_classification(tfidf=True, BoW=True, edinburgh=True, glove=True, Hashtag_Intense=True, Lexicons=True)

def preprocessing_extract_emojis():
    '''
    This funcion runs part of the preprocessing stage.
    It extracts all the emojis in training dataset,
    save them in text.txt, manually delete characters from other languages
    (such as Japanese, Chinese, Arabian, etc.) and illegal forms,
    and save them as a file named emoji.txt.
    '''
    preprocess.Preprocessor(define_emoji= True, year = 2018, classify= False)

def preprocessing_map_emojis():
    '''
    This funcion runs part of the preprocessing stage.
    It maps each emoji in emoji.txt to a unique string (e.g. map 😄 to 'emoji12')
    and save these unique strings to a file named emoji_lexicon.txt.
    '''
    preprocess.Preprocessor(emoji_to_lexicon= True, year = 2018, classify= False)

def extract_features():
    '''
    This funcion runs the feature extraction stage.
    It extracts all the features and save them to a file under 'data' folder.
    Note the row tweets are preprocessed by Preprocessor in line 52.
    '''
    for _emotion in ['anger', 'fear', 'joy', 'sadness']:
        print ('')
        print ('Emotion:', _emotion)
        my_preprocessor = preprocess.Preprocessor(emotion = _emotion, classify = True)
        my_features = feature_extraction.TweetFeatureGenerator(my_preprocessor, emotion = _emotion)

def run_regression(tfidf=True, BoW=True, edinburgh=True, glove=True, Hashtag_Intense=True, Lexicons=True):
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
    regression.Regression(tfidf = tfidf, BoW = BoW, edinburgh = edinburgh, glove = glove, Hashtag_Intense = Hashtag_Intense, Lexicons = Lexicons)

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
    main()
