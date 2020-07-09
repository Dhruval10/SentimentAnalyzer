import analyzer
import tweet_parser
import file_work
import logging

handle = 'realdonaldtrump'


def main():
    logging.basicConfig(format='%(asctime)s {%(pathname)s:%(lineno)d} %(levelname)s %(message)s',
                        filename='application.log',
                        filemode='w',
                        level=logging.INFO,
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    parser = tweet_parser.TweetParser()
    tweets = parser.get_tweets(tweet_criteria=parser.set_tweet_criteria(handle=handle))
    processed_tweets = parser.process_all_tweets(tweets=tweets)
    new_tweet_file = file_work.FileOperations('New_tweets.txt')
    new_tweet_file.write(processed_tweets)
    new_tweet_file.remove_blines()
    parser.remove_stopwords()
    analyze = analyzer.Analyzer()
    new_file = file_work.FileOperations('FilteredTweet.txt').read()

    logging.info(analyze.posneg_write(new_file))

    # print(type(sentiment))
    # analyzed_tweet.write(sentiment)
    # logging.info(analyzed_tweet.write(list(sentiment)))


if __name__ == '__main__':
    main()
