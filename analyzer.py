"""Analyzer can analyze provided dataset and return positive, negative and neutral based on tweets."""

import file_work
import logging
import nltk
import tweet_parser
from textblob import TextBlob
from wordcloud import WordCloud as wd
from nltk import NaiveBayesClassifier
from nltk.corpus import twitter_samples
from nltk import classify
import random
from nltk.tag import pos_tag
import json

class Analyzer(object):

    def __init__(self):
        self.initializer = None
        self.pos = None
        self.neg = None

    def pos_tweet(self):
        ''' TODO: Write about pos_tweet
        This function wil
        '''
        tweet1 = file_work.FileOperations('/home/dhruval/PycharmProjects/Analysis/Files/pwords.txt')
        pos = tweet1.read()
        return pos

    def neg_tweet(self):
        ''' TODO: Write about neg_tweet '''
        tweet2 = file_work.FileOperations('/home/dhruval/PycharmProjects/Analysis/Files/nwords.txt')
        neg = tweet2.read()
        return neg

    def analyze_tweet(self, tweet):
        ''' TODO: Write about assign_sentiment'''
        pos1 = self.pos_tweet()
        neg1 = self.neg_tweet()
        count = 0
        #logging.info(tweet)
        sant = {}
        for word in str(tweet).split(' '):
            # logging.info(word)
            if word in pos1:
                count+=1
                sant[word] = 'positive'
            if word in neg1:
                count-=1
                sant[word] = 'negative'
        return sant, count

    def posneg_write(self, tweets):
        """TODO: Write about posneg_return
            This takes count as input from analyze_tweet
            Args:
                Takes tweet as input
            Returns:
                 positive negative or neutral based on the count value
             tweets = {
                'positive' : ["tweet"],
                'negetive' : [],
                'neutral' : []
             }
        """
        analyzed_tweet = {'positive':[], 'negative': [], 'neutral':[]}

        for tweet in tweets:
            sant, polarity = self.analyze_tweet(tweet)
            if polarity > 0:
                analyzed_tweet['positive'].append(sant)
            elif polarity < 0:
                analyzed_tweet['negative'].append(sant)
            else:
                analyzed_tweet['neutral'].append(sant)
        # logging.info(" Printing "+analyzed_tweet)
        return analyzed_tweet

    def nb_classifier(self):
        ''' TODO: Write about nb_classifier and what it returns '''
        analyzed_tweet = json.loads(file_work.FileOperations('analyzed.txt').read()[0])
        logging.info(analyzed_tweet)
        positive_tweets = [(tweet, 'Positive') for tweet in analyzed_tweet['positive']]
        negative_tweets = [(tweet, 'Negative') for tweet in analyzed_tweet['negative']]
        data_set = positive_tweets +negative_tweets
        random.shuffle(data_set)
        size = len(data_set)
        train_data = data_set[int(size*0.8):]
        test_data = data_set[:int(size*0.2)]
        logging.info(data_set)
        logging.info(train_data)
        logging.info(test_data)
        classifier = NaiveBayesClassifier.train(train_data)
        print('Accuracy is:', classify.accuracy(classifier, test_data))
        logging.info(classifier.show_most_informative_features(10))
        return classifier

    def get_tweet_polarity(self,tweet):
        ''' TODO: Write about get_tweet_polarity'''
        analysis = TextBlob(tweet_parser.TweetParser.clean_tweet(tweet)).sentiment.polarity
        return analysis

    def get_tweet_sentiment(self,tweet):
        ''' ToDO: Write about get_tweet_sentiment '''
        analysis = TextBlob(tweet_parser.TweetParser.clean_tweet(tweet)).sentiment.subjectivity
        return analysis
