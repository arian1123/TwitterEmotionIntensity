# Twitter Emotion Intensity

Our system consists of three stages (1. preprocessing, 2. feature extraction, 3. regression/classification). 

Stage 1 preprocessing:
1) Extracted all the emojis in our training dataset, manually deleting characters from other languages (such as Japanese, Chinese, Arabian, etc.) and illegal forms, and saved them as a file named emoji.txt. (i.e. define_emoji in preprocess.py)
2) Map each emoji in emoji.txt to a unique string (e.g. map 😄  to 'emoji12') and saved these unique strings to a file named emoji_lexicon.txt. (i.e. def_regular_emoji, and emoji_to_lexicon in preprocess.py)
3) Preprocessing raw tweets includes regular emoji (map each emoji to a unique string), spelling correction, acronym, special words, punctuation, symbol replacement, deleting hashtag symbols, and break contractions.

Stage 2 feature extraction:  
Extract all the features and save them in files. 
The features inculude tf-idf, bag of words, edinburgh embeddings, glove embeddings, hash tag intensities, emoji lexicon, and Affect lexicon features including AFINN, BingLiu, MPQA, NRC-EmoLex, NRC-Hash-Emo, NRC-Hash-Sent, NRC10E, Sentiment140, SentiStrength.

Stage 3 regression/classification:  
In this stage, we use the following regressors/classifiers:
Support vector machine (regressor and classifier) of sklearn. 
Multi-layer Perceptron (regressor and classifier) of sklearn. 
Gradient Boosting (regressor and classifier) of sklearn. 

We train and test our data by using 10-fold cross validation on the training dataset. 
