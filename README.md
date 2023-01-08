# Twitter-Sentiment-Analysis

## About the Project

To detect severity from tweets.

Description of experiment
This is the sentiment140 dataset. It contains 1,600,000 tweets extracted using the twitter api . The tweets have been annotated (0 = negative, 2 = neutral, 4 = positive) and they can be used to detect sentiment.
Approach For Solving The Problem 
Read twitter dataset using pandas library

Drop column which doesn’t affect the accuracy of the model
 Removed all the @ for getting the text this is done using nltk library
 
Removed all the punctuations, number and special characters using nltk library
 
Removed all the short words using nltk library
 
Normalized all the text using nltk
 
Extracted feature using count vectorizer
 
Applied RandomForestClassifier for finding the accuracy of the sentiments
 
Result
 
 Accuracy score is 0.7626306135618759 
 
 F1 score is 0.6599289133716688

Link

https://www.kaggle.com/kazanova/sentiment140
