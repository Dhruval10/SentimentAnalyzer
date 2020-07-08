import got3
import re
import langdetect
from langdetect import detect

since_date = '2020-03-08'
until_date = '2020-03-09'
##words = ['economy','GDP','stock market','Unemployment','jobless claims']
handle = 'economy'
count = 100


# def get_tweets():
#     tweet_criteria = got3.manager.TweetCriteria().setSince(since_date).setUntil(until_date).setQuerySearch(handle).setMaxTweets(count)
#     for i in range(count):
#         tweets = got3.manager.TweetManager().getTweets(tweet_criteria)[i]
#         print(tweets.text)


def main():
    dict_obj = dict()
    dict_obj.key = 'Positive'
    dict_obj.key = 'Negative'
    dict_obj.key = 'Zero'





if __name__ == '__main__':
    main()
