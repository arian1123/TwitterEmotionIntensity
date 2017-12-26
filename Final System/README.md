# Twitter Emotion Intensity

Our system consists of three stages (1. preprocessing, 2. feature extraction, 3. regression/classification). 

Stage 1 preprocessing:
1) Extract all the emojis in training dataset, save them in text.txt, manually delete characters from other languages (such as Japanese, Chinese, Arabian, etc.) and illegal forms, and save them as a file named emoji.txt. (i.e. define_emoji in preprocess.py)
2) Map each emoji in emoji.txt to a unique string (e.g. map 😄  to 'emoji12') and save these unique strings to a file named emoji_lexicon.txt. (i.e. def_regular_emoji, and emoji_to_lexicon in preprocess.py)
3) Preprocessing raw tweets includes regular emoji (map each emoji to a unique string), spelling correction, acronym, special words, punctuation, symbol replacement, deleting hashtag symbols, and break contractions. (i.e. regular_tweet in preprocess.py)

Stage 2 feature extraction:  
  
Extract all the features and save them in files under the folder named data (i.e. Features_reg, Features_oc, and 2017FeaturesReg). The features inculude tf-idf, bag of words, edinburgh embeddings, glove embeddings, hash tag intensities, emoji lexicon, and Affect lexicon features including AFINN, BingLiu, MPQA, NRC-EmoLex, NRC-Hash-Emo, NRC-Hash-Sent, NRC10E, Sentiment140, SentiStrength.

Stage 3 regression/classification:  
  
In this stage, we use the following regressors/classifiers: 1) Support vector machine (regressor and classifier) of sklearn. 2) Multi-layer Perceptron (regressor and classifier) of sklearn. 3)Gradient Boosting (regressor and classifier) of sklearn. 

We train and test our data by using 10-fold cross validation on the training dataset. 

## To Run the System:

1. Preprocessing.

Run preprocessing_extract_emojis() in main.py to extract all the emojis in training dataset, save them in text.txt.   
Then, manually delete characters from other languages (such as Japanese, Chinese, Arabian, etc.) and illegal forms, and save them as a file named emoji.txt.  
Run preprocessing_map_emojis() in main.py to map each emoji in emoji.txt to a unique string (e.g. map 😄 to 'emoji12') and save these unique strings to a file named emoji_lexicon.txt.

2. Feature Extraction.

Run extract_features() in main.py. It will extract all the features and save them to a file under 'data' folder. Note this function also preprocess the row tweets (e.g. seperating punctuations, deleting hashtag symbols, and break contractions) before extracting features.

3. Regression or Classification.

3.1) Run run_regression() in main.py to perform regression. It will automatically print out the Pearson correlations and Spearman correlations from 10-fold cross validation on training dataset, for each emotion and each regressor. That is two tables, one for Pearson correlation, the other for Spearman correlation.

3.2) Similarly, run run_classification() in main.py to perform classification. It will automatically print out a table that shows the Pearson correlations from 10-fold cross validation on training dataset, for each emotion and each classifier.



