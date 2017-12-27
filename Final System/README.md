# Twitter Emotion Intensity

## Introduction:
SemEval 2018 Task 1 (Mohammed et al., 2018) addresses the task of detecting the intensity of emotion in tweets. In our approach we used a variety of preprocessing and combination of feature extraction methods, including tf-idf, bag of words, lexicons, etc, proposed in previous works. We then ran our model through different regressors/classifiers to gauge the best results.  

Our system consists of three main stages (1. preprocessing, 2. feature extraction, 3. regression/classification), as described in the following: 

**Stage 1 preprocessing:**  
1) Extract all the emojis in training dataset, save them in test.txt, manually delete characters from other languages (such as Japanese, Chinese, Arabian, etc.) and illegal forms, and save them as a file named emoji.txt. (i.e. the method define_emoji() in preprocess.py)
2) Map each emoji in emoji.txt to a unique string (e.g. map 😄  to 'emoji12') and save these unique strings to a file named emoji_lexicon.txt. (i.e. the methods regular_emoji(), and emoji_to_lexicon() in preprocess.py)
3) Preprocessing raw tweets includes regular emoji (map each emoji to a unique string), spelling correction, acronym, special words, separating punctuations, symbol replacement, hashtag extraction, deleting hashtag symbols, and breaking contractions. (i.e. the methods extract_hashtags() and regular_tweet() in preprocess.py)

**Stage 2 feature extraction:**    
  
1) Extract all the features. The features include tf-idf, bag of words, Edinburgh embeddings, GloVe embeddings, hash tag intensities, emoji lexicon, and Affect lexicon features including AFINN, BingLiu, MPQA, NRC-EmoLex, NRC-Hash-Emo, NRC-Hash-Sent, NRC10E, Sentiment140, SentiStrength.
2) Reduce the dimensions of lexicon features because these features are sparse and their dimensions are high. 
3) Save the extracted features in files under the folder 'data' (i.e. the folders named Features_reg, Features_oc, and 2017FeaturesReg). 

**Stage 3 regression/classification:**    
  
1. In this stage, we use the following regressors/classifiers: 1) Support vector machine (regressor and classifier) of sklearn. 2) Multi-layer Perceptron (regressor and classifier) of sklearn. 3)Gradient Boosting (regressor and classifier) of sklearn, to perform 10-fold cross validation on the training dataset. 

2. Then we print out the averaged evaluations from the above 10-fold cross validation and record them in Regression_Results.xlsx. The evaluation metric for regression task is Pearson correlation and Spearman correlation, for classification task is Pearson correlation. Note our main focus is on the regression task.

## To Run the System:

To run the system, specify which stage you want to run by assigning True or False values to the states (i.e. stage1, stage2, and stage3). You also need to specify whether you want to run regression or classification by assign True or False to classify. Then, specify year of the data you want to use by assign values to year. Finally, you need to specify the features you want to use by assigning True or False to the features.  
Note the features are tfidf, BoW (bag of words), edinburgh (Edinburgh embeddings), glove (GloVe embeddings), Hashtag_Intense (Hashtag intensity), Lexicons (Affect lexicon features, which is the combination of all affect lexicon features)

#### Preprocessing:

1. Run main(stage1 = True, stage2 = False, stage3 = False, classify = False, year = 2018) in main.py to extract all the emojis in training dataset, save them in text.txt.   
2. Then, manually delete characters from other languages (such as Japanese, Chinese, Arabian, etc.) and illegal forms in text.txt, and save the cleaned text.txt as a new file named emoji.txt.  
3. Run main(stage1 = True, stage2 = False, stage3 = False, classify = False, year = 2018) in main.py again to map each emoji in emoji.txt to a unique string (e.g. map 😄 to 'emoji12') and save these unique strings to a file named emoji_lexicon.txt.

#### Feature Extraction:

1. Run main(stage1 = False, stage2 = True, stage3 = False, classify = False, year = 2018) in main.py. It will extract all the features, reduce the dimensions of lexicon features, and save them to a file under 'data' folder. 

#### Regression or Classification:

1. Run main(stage1 = False, stage2 = True, stage3 = False, classify = False, year = 2018, tfidf=True, BoW=True, edinburgh=True, glove=True, Hashtag_Intense=True, Lexicons=True) in main.py to to perform 10-fold cross validation regression. You can select features by assigning True values to the feature parameters. It will automatically print out the averaged Pearson correlations and averaged Spearman correlations from 10-fold cross validation on training dataset, for each emotion and each regressor. That is two tables, one's evaluation metric is Pearson correlation, the other's evaluation metric is Spearman correlation. 
2. To perform 10-fold cross validation classification, you just need to change to parameter classify's value to be true. You can also select features by assigning True or False values to the parameters. It will automatically print out a table that shows the averaged Pearson correlations from 10-fold cross validation on training dataset, for each emotion and each classifier.  

Note: all the stages are integrated into main.py, specifically, main() in main.py. 


## Description of Files in this Folder:

**DepecheMood:** this folder contains three versions of the Lexicon 'DepecheMood'. The detailed description is in the README.txt in the folder.  

**data:** this folder contains 1) the training data for regression task and classification task in 2018, and the training data for the regression task in 2017. (i.e. EI-oc-En-train, EI-reg-En-train	, 2017train) 2) Affect lexicons (AFINN, BingLiu, MPQA, NRC-EmoLex, NRC-Hash-Emo, NRC-Hash-Sent, NRC10E, Sentiment140, SentiWordNet, SentiStrength). 3) The folders to store extracted features for corresponding training datasets. (i.e. Features_reg, Features_oc, 2017FeaturesReg)  

**embedding:** this folder contains Edinburgh word vectors and GloVe word vectors. They are pre-trained and can be downloaded from the links given in the folder's readme file. Note that they need to be downloaded in order to run extract_features() in main.py.

**classification.py:** similar to regression.py, this script contains Classification class for the classification task. It will read the selected pre-stored features from respective files, train on three classifiers (Support vector machine classifier of sklearn, Multi-layer Perceptron classifier of sklearn, and Gradient Boosting classifier of sklearn.) using 10 fold cross validation on training dataset. Then, print the averaged Pearson correlations for each emotion and each classifier, as a table.  

**emoji.txt:** this file stores the "cleaned" emojis from test.txt which contains emojis with "noises" such as other languages (e.g. Japanese, Chinese, Arabian, etc.) and illegal forms extracted from training dataset. In other words, emoji.txt is created by manually deleting the "noises" in test.txt. 

**emoji_lexicon.txt:** this file stores the mapped emojis (each emoji is mapped to a unique string as described in the introduction section, for example 'emoji12').  

**evaluation_metrics.py:** this script contains the common evaluation metrics for regression (Pearson correlation, and Spearman correlation) and classification (accuracy, micro recall, macro recall, confusion matrix, and Pearson correlation). Since the major evaluation matric for SemEval 2018 Task 1 and Task 2 is Pearson correlation, we focus on Pearson correlations.  

**feature_extraction.py:** this script contains TweetFeatureGenerator class with the methods to extract all the features from preprocessed training tweets. Reduce the dimensions of lexicon features, because their dimensions are high (e.g. over 5000). Then save these features to a file under 'data' folder. The features include tf-idf, bag of words, Edinburgh embeddings, GloVe embeddings, hashtag intensity, emoji lexicon feature, and affect lexicon features (AFINN, BingLiu, MPQA, NRC-EmoLex, NRC-Hash-Emo, NRC-Hash-Sent, NRC10E, Sentiment140, SentiStrength).

**lexicons.py:** this script contains the methods to extract affect lexicon features for a single preprocessed tweet. Note besides tweetToEmoji method, the other parts of this script come from lexicons.py in https://github.com/amanjaiman/DNNTwitterEmoInt.  

**main.py:** this is the main script that integrate and run all the stages, and methods. It includes preprocessing, feature extraction, and regression/classification. You can specify the parameters in main() to let it run different stages for different years' data and different task(regression or classification). 

**preprocess.py:** this script contains Preprocessor class with all the methods for preprocessing as described in stage 1, and the method to read the row training tweets. It has three major functions, extract emojis, mapping emojis to unique strings and save them, and regular tweets (e.g. separating punctuations, deleting hashtag symbols, and break contractions). You can specify different values to the parameters to let the Preprocessor do different things. If you want to extract emojis, you can use preprocess.Preprocessor(define_emoji = True), it will extract emojis from 2018's training data of regression task for the default emotion, and save them to a file named test.txt. If you want to mapping emojis to unique strings and save them, you can use preprocess.Preprocessor(emoji_to_lexicon = True), it will map and save the mapped emojis to a file named emoji_lexicon.txt for the default emotion. If you want to regular tweets, you can simply use preprocess.Preprocessor(), it will regular tweets for the default emotion.

**regression.py:** similar to classification.py, this script contains Regression class for the Regression task. It will read the selected pre-stored features from respective files, train on three regressors (Support vector machine regressor of sklearn, Multi-layer Perceptron regressor of sklearn, and Gradient Boosting regressor of sklearn.) using 10 fold cross validation on training dataset. Then, print the averaged Pearson correlations and averaged Spearman correlations for each emotion and each regressor, as two tables.  

**sentistrength.py:** this script contains the methods to convert a preprocessed tweet to a sentistrength lexicon feature. Note this script come from sentistrength.py in https://github.com/amanjaiman/DNNTwitterEmoInt.  

**test.txt:** this file contains the emojis that are extracted from training tweets. It also contains some noises such as other languages (e.g. Japanese, Chinese, Arabian, etc.) and illegal forms that needs to be deleted manually.  

**Regression_Results.xlsx:** is the evaluation results for different single features and combinations of features that we ran. Note that the results come from 10-fold cross validation on training dataset. The evaluation matric is Pearson Correlation. The high scores are in bold. The regressor used is MLPRegressor.





