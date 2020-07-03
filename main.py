import analyzer
import tweet_parser
import file_work
import logging

handle = 'BernieSanders'


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
    parser.remove_stopwords()
    nfile = file_work.FileOperations('nwords.txt')
    pfile = file_work.FileOperations('pwords.txt')


if __name__ == '__main__':
    main()
