import numpy as np
import pandas as pd
import re
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn import metrics
from sklearn import preprocessing
import nltk
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import gensim
from nltk.stem.porter import *
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)

    return input_txt

 # function to collect hashtags
def hashtag_extract(x):
    hashtags = []
    # Loop over the words in the tweet
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)

    return hashtags

def main():
    #reading data sets

    df=pd.read_csv('Twitter.csv', encoding='latin-1')

    #Preprocessing data

    df.drop(['Date','Flage','User','Id'],axis=1,inplace=True)

      #Removing @user
    df['tidy_tweet'] = np.vectorize(remove_pattern)(df['Text'], "@[\w]*")

      # Removing Punctuations, Numbers, and Special Characters
    df['tidy_tweet']=df['tidy_tweet'].str.replace("[^a-zA-Z#]", " ")

      # Removing Short Words
    df['tidy_tweet'] = df['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 3]))

      #Text Normalization
    tokenized_tweet = df['tidy_tweet'].apply(lambda x: x.split())  # tokenizing

    stemmer = PorterStemmer()

    tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])  # stemming

    for i in range(len(tokenized_tweet)):
        tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

    df['tidy_tweet'] = tokenized_tweet

    # print(tokenized_tweet.head())

    all_words = ' '.join([text for text in df['tidy_tweet']])
    wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)
    plt.figure(figsize=(10, 7))
    # plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    # plt.show()

    HT_regular = hashtag_extract(df['tidy_tweet'])

    # unnesting list
    HT_regular = sum(HT_regular, [])

    a = nltk.FreqDist(HT_regular)
    d = pd.DataFrame({'Hashtag': list(a.keys()),
                      'Count': list(a.values())})

    # selecting top 20 most frequent hashtags
    d = d.nlargest(columns="Count", n=20)
    plt.figure(figsize=(16, 5))
    ax = sns.barplot(data=d, x="Hashtag", y="Count")
    ax.set(ylabel='Count')
    # plt.show()

    bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
    bow= bow_vectorizer.fit_transform(df['tidy_tweet'])
    # df['result'] = bow_vectorizer.fit_transform(df['tidy_tweet'])
    #
    # tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
    # tfidf = tfidf_vectorizer.fit_transform(df['tidy_tweet'])

    # APPLYING  ML


    # splitting data into training and validation set
    xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(bow,df['Target'], random_state=42, test_size=0.3)

    lreg = RandomForestClassifier(n_estimators=100, max_depth=2,
                              random_state=2)
    lreg.fit(xtrain_bow, ytrain)  # training the model

    prediction = lreg.predict_proba(xvalid_bow)  # predicting on the validation set
    predicted=lreg.predict(xvalid_bow)
    prediction_int = prediction[:, 1] >= 0.5  # if prediction is greater than or equal to 0.3 than 1 else 0
    prediction_int = prediction_int.astype(np.int)

    expected = yvalid

    # print(metrics.classification_report(expected, prediction))

    # Print Accuracy
    print(' Accuracy :' + str(metrics.accuracy_score(expected, predicted)))

    score=f1_score(yvalid, prediction_int,pos_label=1, average='weighted')  # calculating f1 score

    print(score)

main()