#main script to run system
from Arian import preprocess
from Arian import SVM
import scipy.stats
from sklearn.model_selection import KFold
import pandas as pd

def main():

    training_dir = "Arian/data"
    parsed_tweets = preprocess.TweetParser(training_dir)
    fg = preprocess.TweetFeatureGenerator(parsed_tweets)
    bag_words_model = fg.build_bag_words()[1] #input
    emotion_intensities = fg.tweet_data.tweet_list_dataframe['emotion_intensity'] #output

    #folds
    kf = KFold(n_splits=10, random_state=2)
    folds = kf.split(bag_words_model)

    print("Successfully generated fold indices.")

    temp_folds = [] #hold tuples holding split data for each fold

    #populate folds
    for train_index, test_index in folds:

        #training data
        training_input = [bag_words_model[i] for i in train_index]
        training_output = [emotion_intensities[i] for i in train_index]

        #test data
        test_input = [bag_words_model[i] for i in test_index]
        test_output = [emotion_intensities[i] for i in test_index]

        temp_folds.append((training_input, training_output, test_input, test_output))

    print("Populated training and test data for each fold.")

    fold_data = pd.DataFrame(temp_folds, columns=['training_input', 'training_output', 'test_input', 'test_output'])

    print("Created DataFrame of fold training and test data.")

    #Test through SVM + Pearson's correlation for each fold
    for idx, fold in fold_data.iterrows():

        print("Fitting SVM for Fold No. " + str(idx+1))
        clf = SVM.TwitterSVM(fold['training_input'], fold['training_output'])

        print("Generating predictions for test data of Fold No. " + str(idx+1))
        prediction = clf.predict(fold['test_input'])

        #correlation
        pears_corr = scipy.stats.pearsonr(prediction, fold['test_output'])[0]
        spear_corr = scipy.stats.spearmanr(prediction, fold['test_output'])[0]

        print("Pearson correlation: " + str(pears_corr))
        print("Spearman correlation: " + str(spear_corr))

#end main


#changeable function to run test individual features
def test():

    main()

#end test

if __name__ == "__main__":
    test()