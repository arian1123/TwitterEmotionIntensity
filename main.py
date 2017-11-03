#main script to run system
import preprocess
import SVM
from sklearn.model_selection import KFold

def main():

    training_dir = "data"
    features_obj = preprocess.TweetFeatureGenerator(training_dir)
    bag_words = features_obj.bag_words_model
    emotion_intensities = features_obj.tweet_data.tweet_list_dataframe['emotion_intensity']

    kf = KFold(len(bag_words), 10, shuffle=True)

    for train_index, test_index in kf:
        print(train_index)

#end main

if __name__ == "__main__":
    main()