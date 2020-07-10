import analyzer
import tweet_parser
import file_work
import logging
import json

handle = ['realdonaldtrump', 'joebiden']


def main():
    logging.basicConfig(format='%(asctime)s {%(pathname)s:%(lineno)d} %(levelname)s %(message)s',
                        filename='application.log',
                        filemode='w',
                        level=logging.INFO,
                        datefmt='%m/%d/%Y %I:%M:%S %p')

    parser = tweet_parser.TweetParser()
    for key in handle:
        tweets = parser.get_tweets(tweet_criteria=parser.set_tweet_criteria_keyword(keyword=key))
        processed_tweets = parser.process_all_tweets(tweets=tweets)
        new_tweet_file = file_work.FileOperations('tweets.txt')
        new_tweet_file.write(processed_tweets)
        new_tweet_file.remove_blines()

    parser.remove_stopwords()
    analyze = analyzer.Analyzer()
    new_file = file_work.FileOperations('FilteredTweet.txt').read()
    logging.info(analyze.posneg_write(new_file))
    analysis = file_work.FileOperations('analyzed.txt').write([json.dumps(analyze.posneg_write(new_file))])

    print(analyze.nb_classifier())


if __name__ == '__main__':
    main()
