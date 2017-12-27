# Twitter Emotion Intensity

## Introduction:
SemEval 2018 Task 1 (Mohammed et al., 2018) addresses the task of detecting the intensity of emotion in tweets. In our approach we used a variety of preprocessing and combination of feature extraction methods, including tf-idf, bag of words, lexicons, etc, proposed in previous works. We then ran our model through different regressors/classifiers to gauge the best results.  

Our system consists of three main stages (1. preprocessing, 2. feature extraction, 3. regression/classification). 

Stage 1 preprocessing:
1) Extract all the emojis in training dataset, save them in test.txt, manually delete characters from other languages (such as Japanese, Chinese, Arabian, etc.) and illegal forms, and save them as a file named emoji.txt. (i.e. define_emoji in preprocess.py)
2) Map each emoji in emoji.txt to a unique string (e.g. map 😄  to 'emoji12') and save these unique strings to a file named emoji_lexicon.txt. (i.e. def_regular_emoji, and emoji_to_lexicon in preprocess.py)
3) Preprocessing raw tweets includes regular emoji (map each emoji to a unique string), spelling correction, acronym, special words, punctuation, symbol replacement, deleting hashtag symbols, and break contractions. (i.e. regular_tweet in preprocess.py)

Stage 2 feature extraction:  
  
1) Extract all the features. The features inculude tf-idf, bag of words, edinburgh embeddings, glove embeddings, hash tag intensities, emoji lexicon, and Affect lexicon features including AFINN, BingLiu, MPQA, NRC-EmoLex, NRC-Hash-Emo, NRC-Hash-Sent, NRC10E, Sentiment140, SentiStrength.
2) Reduce the dimensions of lexicon features because these features are sparse and their dimensions are high. 
3) Save them in files under the folder named data (i.e. Features_reg, Features_oc, and 2017FeaturesReg). 

Stage 3 regression/classification:  
  
In this stage, we use the following regressors/classifiers: 1) Support vector machine (regressor and classifier) of sklearn. 2) Multi-layer Perceptron (regressor and classifier) of sklearn. 3)Gradient Boosting (regressor and classifier) of sklearn. 

We perform 10-fold cross validation on the training dataset. 

The evaluation metric for regression task is Pearson correlation and Spearman correlation, for classification task is Pearson correlation. Note our main focus is on the regression task.

## To Run the System:

#### Preprocessing:

1. Run preprocessing_extract_emojis() in main.py to extract all the emojis in training dataset, save them in text.txt.   
2. Then, manually delete characters from other languages (such as Japanese, Chinese, Arabian, etc.) and illegal forms in text.txt, and save the cleaned text.txt as a new file named emoji.txt.  
3. Run preprocessing_map_emojis() in main.py to map each emoji in emoji.txt to a unique string (e.g. map 😄 to 'emoji12') and save these unique strings to a file named emoji_lexicon.txt.

#### Feature Extraction:

1. Run extract_features() in main.py. It will extract all the features, reduce the dimensions of lexicon features, and save them to a file under 'data' folder. Note this function also preprocess the row tweets (e.g. seperating punctuations, deleting hashtag symbols, and break contractions) before extracting features.

#### Regression or Classification:

1. Run run_regression() in main.py to perform 10-fold cross validation regression. In default, it selects all the features, but you can select features by assigning True or False values to the parameters. For example, run_regression(tfidf=True, BoW=False, edinburgh=False, glove=False, Hashtag_Intense=False, Lexicons=False) selects only tfidf as its feature. It will automatically print out the averaged Pearson correlations and averaged Spearman correlations from 10-fold cross validation on training dataset, for each emotion and each regressor. That is two tables, one's evaluation metric is Pearson correlation, the other's evaluation metric is Spearman correlation.
2. Similarly, run run_classification() in main.py to perform 10-fold cross validation classification. You can also select features by assigning True or False values to the parameters. It will automatically print out a table that shows the averaged Pearson correlations from 10-fold cross validation on training dataset, for each emotion and each classifier.  

Note: all the stages are integrated into main.py, specifically, main() in main.py. 


## Description of Files in this Folder:

**DepecheMood:** this folder contains three versions of the Lexicon 'DepecheMood'. The detailed description is in the README.txt in the folder.  

**data:** this folder contains 1) the training data for regression task and classification task in 2018, and the training data for the regression task in 2017. (i.e. EI-oc-En-train, EI-reg-En-train	, 2017train) 2) Affect lexicons (AFINN, BingLiu, MPQA, NRC-EmoLex, NRC-Hash-Emo, NRC-Hash-Sent, NRC10E, Sentiment140, SentiWordNet, SentiStrength). 3) The folders to store extracted features for corresponding training datasets. (i.e. Features_reg, Features_oc, 2017FeaturesReg)  

**embedding:** this folder contains Edinburgh word vectors and GloVe word vectors. They are pre-trained and can be downloaded from the links givien in the folder's readme file. Note that they need to be downloaded in order to run extract_features() in main.py.

**classification.py:** this script contains Classification class for the classification task. It will read the selected pre-stored features from respective files, train on three classifiers (Support vector machine classifier of sklearn, Multi-layer Perceptron classifier of sklearn, and Gradient Boosting classifier of sklearn.) using 10 fold cross validation on training dataset. Then, print the averaged pearson correlations for each emotion and each classifier, as a table.  

**emoji.txt:** this file stores the emojis that are extracted from training dataset.  

**emoji_lexicon.txt:** this file stores the mapped emojis (each emoji is mapped to a unique string as described in the introduction section, for example 'emoji12').  

**evaluation_metrics.py:** this script contains the common evaluation metrics for regression (Pearson correlation, and Spearman correlation) and classification (accuracy, micro recall, macro recall, confusion matrix, and Pearson correlation). Since the major evaluation matric for SemEval 2018 Task 1 and Task 2 is Pearson correlation, we focus on Pearson correlations.  

**feature_extraction.py:** this script contains TweetFeatureGenerator class with the methods to extract all the features from preprocessed training tweets. Reduce the dimentions of lexicon features, because their dimentions are high (e.g. over 5000). Then save these features to a file under 'data' folder. The features include tf-idf, bag of words, Edinburgh embeddings, GloVe embeddings, hashtag intensity, emoji lexicon feature, and affect lexicon features (AFINN, BingLiu, MPQA, NRC-EmoLex, NRC-Hash-Emo, NRC-Hash-Sent, NRC10E, Sentiment140, SentiStrength).

**lexicons.py:** this script contains the methods to extract affect lexicon features for a single preprocessed tweet. Note besides tweetToEmoji method, the other parts of this script come from lexicons.py in https://github.com/amanjaiman/DNNTwitterEmoInt.  

**main.py:** this is the main script that integrate and run all the stages, and methods. It includes preprocessing, feature extraction, and regression/classification. To run a single stage, you can comment out the other two stages in main(). 

**preprocess.py:** this script contains Preprocesor class with all the methods for preprocessing as described in stage 1, and the method to read the row training tweets. It has three major functions, extract emojis, mapping emojis to unique strings and save them, and regular tweets. You can specify different values to the parameters to let the Preprocessor do different things. If you want to extract emojis, you can use preprocess.Preprocessor(define_emoji = True), it will extract emojis from 2018's training data of regression task for the default emotion, and save them to a file named test.txt. If you want to mapping emojis to unique strings and save them, you can use preprocess.Preprocessor(emoji_to_lexicon = True), it will map and save the mapped emojis to a file named emoji_lexicon.txt for the default emotion. If you want to regular tweets, you can simply use preprocess.Preprocessor(), it will regular tweets for the default emotion.

**regression.py:** similar to classification.py, this script contains Regression class for the Regression task. It will read the selected pre-stored features from respective files, train on three regressors (Support vector machine regressor of sklearn, Multi-layer Perceptron regressor of sklearn, and Gradient Boosting regressor of sklearn.) using 10 fold cross validation on training dataset. Then, print the averaged Pearson correlations and averaged Spearman correlations for each emotion and each regressor, as two tables.  

**sentistrength.py:** this script contains the methods to convert a preprocessed tweet to a sentistrength lexicon feature. Note this script come from sentistrength.py in https://github.com/amanjaiman/DNNTwitterEmoInt.  

**test.txt:** this file contains the emojis that are extracted from training tweets. It also contains some noises such as other languages (e.g. Japanese, Chinese, Arabian, etc.) and illegal forms that needs to be deleted manually.  

**Regression_Results.xlsx:** is the evaluation results for different single features and combinations of features that we ran. Note that the results come from 10-fold cross validation on training dataset. The evaluation matric is Pearson Correlation. The high scores are in bold. The regressor used is MLPRegressor.





