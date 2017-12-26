import preprocess
import evaluation_metrics
import feature_extraction
import regression
import classification
import numpy as np



def main():

    # 1. Preprocessing
    preprocess.Preprocessor(define_emoji= True, year = 2018, classify= False)
    preprocess.Preprocessor(emoji_to_lexicon= True, year = 2018, classify= False)

    # 2. Featrue Extraction
    for _emotion in ['anger', 'fear', 'joy', 'sadness']:
        print ('')
        print ('Emotion:', _emotion)
        my_preprocessor = preprocess.Preprocessor(emotion = _emotion, classify = True)
        my_features = feature_extraction.TweetFeatureGenerator(my_preprocessor, emotion = _emotion)

    # 3. Regression or Classification
    regression.Regression(tfidf = True, BoW = True, edinburgh = True, glove = True, Hashtag_Intense = True, Lexicons = True)
    classification.Classification(tfidf = True, BoW = True, edinburgh = True, glove = True, Hashtag_Intense = True, Lexicons = True)

if __name__ == "__main__":
    main()