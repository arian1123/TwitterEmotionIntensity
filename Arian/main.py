#main script to run system
from Arian import preprocess
from Arian import SVM
import scipy.stats
import pandas as pd
import os
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold


#changeable function to run test individual features
def main():

    for emotion in ['anger', 'fear', 'joy', 'sadness']:

        print("Emotion: " + emotion)

        tweets_directory = "Arian/data/el_reg_training/"

        for file in os.listdir(tweets_directory):
            if(file.find(emotion) != -1):
                emotion_tweets_file = os.path.join(tweets_directory, file)

        parsed_tweets = preprocess.TweetParser(emotion_tweets_file)
        fg = preprocess.TweetFeatureGenerator(parsed_tweets)
        vocab, bag_words_model = fg.build_bag_words() #input
        emotion_intensities = fg.tweet_data.tweet_list_dataframe['emotion_intensity'] #output

        # folds
        kf = KFold(n_splits=10, random_state=2)
        folds = kf.split(bag_words_model)

        temp_folds = []  # hold tuples holding split data for each fold

        # populate folds
        for train_index, test_index in folds:
            # training data
            training_input = [bag_words_model[i] for i in train_index]
            training_output = [float(emotion_intensities[i]) for i in train_index]

            # test data
            test_input = [bag_words_model[i] for i in test_index]
            test_output = [float(emotion_intensities[i]) for i in test_index]

            temp_folds.append((training_input, training_output, test_input, test_output))

        fold_data = pd.DataFrame(temp_folds, columns=['training_input', 'training_output', 'test_input', 'test_output'])

        # Test through SVM + Pearson's correlation for each fold
        for idx, fold in fold_data.iterrows():

            print("Fold No. " + str(idx + 1))

            ###SVM###
            svm = SVM.TwitterSVM(fold['training_input'], fold['training_output'])
            svm_prediction = svm.predict(fold['test_input'])

            # correlation
            svm_measure = measure_reg(svm_prediction.tolist(), fold['test_output'])
            print("SVM result: " + svm_measure)

            ###XBoost###
            XGboost = GradientBoostingRegressor(n_estimators=200)
            XGboost.fit(fold['training_input'], fold['training_output'])
            xgb_prediction = XGboost.predict(fold['test_input'])

            #correlation
            xgb_measure = measure_reg(xgb_prediction.tolist(), fold['test_output'])
            print("XGBoost result: " + xgb_measure)

            ###MLP####
            MLP = MLPRegressor(hidden_layer_sizes=[100, 50], activation='logistic')
            MLP.fit(fold['training_input'], fold['training_output'])
            mlp_prediction = MLP.predict(fold['test_input'])

            #correlation
            mlp_measure = measure_reg(mlp_prediction.tolist(), fold['test_output'])
            print("MLP result: " + mlp_measure)


        print("*********************")


#end main

def test():
    main()

#end test

def measure_reg(y, z):
    # return np.corrcoef(y, z)[0,1]
    pears_corr = scipy.stats.pearsonr(y, z)[0]
    spear_corr = scipy.stats.spearmanr(y, z)[0]
    output = "Pearson correlation: " + str(pears_corr) + "; Spearman correlation: " + str(spear_corr)
    return output

#end measure_reg

if __name__ == "__main__":
    test()