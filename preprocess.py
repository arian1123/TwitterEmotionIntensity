#pre processing and feature extraction for training files

from EmoInt import evaluate
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


#Use this class to prepare and parse tweets for feature generatio
class TweetPreprocessor:

    #pass in directory with training data files
    def __init__(self, directory):

        self.tweet_list = [] #list of dictionaries to hold tweets data
        self.tweet_list_dataframe = []

        # traverse down directory recursively
        for dirpath, dirs, files in os.walk(directory):

            for filename in files:

                #read each txt file
                if (filename.endswith(".txt")):

                    self.process_file(os.path.join(dirpath, filename))

        self.tweet_list_dataframe = pd.DataFrame(self.tweet_list)

    #end init

    def process_file(self, path):

        file = open(path)
        for f in file:

            f = f.replace("\n", "")
            tokens = f.split("\t")

            data = {}
            data['id'] = tokens[0]
            data['content'] = tokens[1]
            data['emotion'] = tokens[2]
            data['emotion_intensity'] = tokens[3]

            self.tweet_list.append(data)

    #end read_file


#generate variety of features
class TweetFeatureGenerator:

    def __init__(self, directory):

        self.tweet_data = TweetPreprocessor(directory) #TweetPreprocessorObject
        self.vocabulary_list, self.bag_words_model = self.build_bag_words()

    #end init


    #generate bag of words model
    def build_bag_words(self):

        vectorizer = CountVectorizer()
        corpus = self.tweet_data.tweet_list_dataframe['content'].tolist()
        bag_of_words = vectorizer.fit_transform(corpus)

        return vectorizer.get_feature_names(), bag_of_words.toarray()

    #end build_bag_words
