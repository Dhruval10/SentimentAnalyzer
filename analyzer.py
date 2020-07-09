"""Analyzer can analyze provided dataset and return positive, negative and neutral based on tweets."""

import file_work
import logging
import nltk
import tweet_parser
from textblob import TextBlob
from wordcloud import WordCloud as wd
from nltk import NaiveBayesClassifier

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
        logging.info(tweet)
        for word in str(tweet).split(' '):
            # logging.info(word)
            if word in pos1:
                count+=1
                logging.info(word)
            if word in neg1:
                count-=1
                logging.info(word)
        # tweets = {}
        # tweets['Positive'] = []
        # tweets['Negative'] = []
        # tweets['Neutral'] = []
        #
        # if count > 0:
        #     tweets['Positive'].append(tweet,count)
        # elif count < 0:
        #     tweets['Negative'].append(tweet,count)
        # else:
        #     tweets['Neutral'].append(tweet,count)
        #
        # logging.info(tweets['Positive'].append(tweet,count))
        # logging.info(tweets['Negative'].append(tweet,count))
        # logging.info(tweets['Neutral'].append(tweet,count))

        return count

    def posneg_write(self, tweets):
        """TODO: Write about posneg_return
            This takes count as input from analyze_tweet
            Args:
                Takes tweet as input
            Returns:
                 positive negative or neutral based on the count value
             tweets = {
                'positive' : [{"tweet": score},],
                'negetive' : [],
                'neutral' : []
             }
        """
        analyzed_tweet = {'positive':{}, 'negative': {}, 'neutral':{}}

        for tweet in tweets:
            polarity = self.analyze_tweet(tweet)
            if polarity > 0:
                analyzed_tweet['positive'][tweet] = polarity
            elif polarity < 0:
                analyzed_tweet['negative'][tweet] = polarity
            else:
                analyzed_tweet['neutral'][tweet] = 0

        return analyzed_tweet

    def nb_classifier(self):
        ''' TODO: Write about nb_classifier and what it returns '''
        analyzed_tweet = file_work.FileOperations('analyzed.txt').read()
        classifier = NaiveBayesClassifier.train(analyzed_tweet)
        return classifier

    def get_tweet_polarity(self,tweet):
        ''' TODO: Write about get_tweet_polarity'''
        analysis = TextBlob(tweet_parser.TweetParser.clean_tweet(tweet)).sentiment.polarity
        return analysis

    def get_tweet_sentiment(self,tweet):
        ''' ToDO: Write about get_tweet_sentiment '''
        analysis = TextBlob(tweet_parser.TweetParser.clean_tweet(tweet)).sentiment.subjectivity
        return analysis
