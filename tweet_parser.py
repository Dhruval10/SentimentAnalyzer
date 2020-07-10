"""Parse all the tweets using got3 api"""

import got3
import re
import logging
import file_work
import string
# from nltk.corpus import stopwords
from langdetect import detect


class TweetParser(object):

    def __init__(self):
        self.initializer = None

    def set_tweet_criteria(self, since_date='2020-03-08', until_date='2020-07-01', handle=None, max_tweets=10000):
        """This function will return all the tweets for specified handle between since_date and until_date.

        Args:
            since_date: Beginning of the time for parsing.
            until_date: Last day parser will parse.
            handle: Tweeter handle to parse the tweets.
            max_tweets: Total Number of tweets you want.

        Returns:
            This function will return tweetCriteria object for specified params.
        """
        # return got3.manager.TweetCriteria().setSince(
        #     since=since_date).setUntil(until=until_date).setUsername(username=handle).setMaxTweets(maxTweets=max_tweets)

        return got3.manager.TweetCriteria().setSince(
                since=since_date).setUntil(until=until_date).setUsername(username=handle).setMaxTweets(maxTweets=max_tweets)

    def set_tweet_criteria_keyword(self, since_date='2020-03-08', until_date='2020-07-01', keyword=None, max_tweets=10000):
        """This function will return all the tweets for specified handle between since_date and until_date.

        Args:
            since_date: Beginning of the time for parsing.
            until_date: Last day parser will parse.
            handle: Tweeter handle to parse the tweets.
            max_tweets: Total Number of tweets you want.

        Returns:
            This function will return tweetCriteria object for specified params.
        """
        # return got3.manager.TweetCriteria().setSince(
        #     since=since_date).setUntil(until=until_date).setUsername(username=handle).setMaxTweets(maxTweets=max_tweets)

        return got3.manager.TweetCriteria().setSince(
                since=since_date).setUntil(until=until_date).setQuerySearch(querySearch=keyword).setMaxTweets(maxTweets=max_tweets)

    def get_tweets(self, tweet_criteria=None):
        """This function will return all the tweets for given criteria.

        Args:
            tweet_criteria: Tweet Criteria Object to fetch tweets. Use set_tweet_criteria for it.

        Returns:
            This function will return tweet set for specified params.
        """
        return got3.manager.TweetManager.getTweets(tweet_criteria)

    def process_all_tweets(self, tweets):
        """ This function will process all the tweets. It will call clean_tweet and return all the processed tweets

        Args:
            tweets: tweet object will have all the tweets

        Returns:
            This function will return all clean tweets
        """
        processed_tweets = []
        for tweet in tweets:
            #logging.info(tweet.text)
            processed_tweets.append(self.clean_tweet(tweet))
        return processed_tweets

    def clean_tweet(self, tweet):
        """This function will clean all the tweet. It will return clean tweet

        Args:
            tweet: tweet object will have one tweet

        Return:
            This function will return single clean tweet
        """
        # return ' '.join(re.sub('(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) |(\w+:\/\/\S+)| ((www\.[^\s]+) |(http?://[^\s]+)|(https?://[^\s]+)|(.pic.[^\s]+))', ' ', tweet.text).split())
        return ' '.join(re.sub(r'(@[A-Za-z0-9]+) | ([^0-9A-Za-z \t]) |[^\w\s]|(pic.twitter\.[^\s]+)|((www\.[^\s]+) |(http?://[^\s]+) |(https?://[^\s]+)|(.pic.[^\s]+))', ' ', tweet.text).split())

    def remove_stopwords(self):
        """This function will remove all the stopwords present in the stopwords.txt file

        Args:
            No argument needed to pass. It will

        Returns:
            It will return filtered sentence which will not have any stopword in it
        """
        # stop_words = set(stopwords.words('english'))
        stop_words_file = file_work.FileOperations('/home/dhruval/PycharmProjects/Analysis/Files/stopwords.txt')
        stop_words = stop_words_file.read()
        # logging.info(stop_words)
        new_tweet_file = file_work.FileOperations('tweets.txt')
        tweets = new_tweet_file.read()
        filtered_text_file = file_work.FileOperations('FilteredTweet.txt')
        #logging.info(tweets)
        filtered_text = []

        for tweet in tweets:
            logging.info(tweet)
            words = tweet.split()
            ##logging.info(words)
            word_list = []
            for word in words:
                logging.info(word)
                if word not in stop_words:
                    logging.info('found match!!!')
                    word_list.append(word)
            logging.info(word_list)
            filtered_text.append(' '.join(word_list))
        filtered_text_file.write(filtered_text)

    def print(self):
        """This function prints local variables."""
        print(self.initializer)
