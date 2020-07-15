import analyzer
import tweet_parser
import file_work
import logging
import json
import pickle
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier

import re, string, random

handle = ['USA Election', 'joebiden']


def main():
    analyze = analyzer.Analyzer()
    positive_tweets = twitter_samples.strings('positive_tweets.json')
    negative_tweets = twitter_samples.strings('negative_tweets.json')
    text = twitter_samples.strings('tweets.20150430-223406.json')
    tweet_tokens = twitter_samples.tokenized('positive_tweets.json')[0]

    stop_words = stopwords.words('english')

    positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
    negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

    positive_cleaned_tokens_list = []
    negative_cleaned_tokens_list = []

    for tokens in positive_tweet_tokens:
        positive_cleaned_tokens_list.append(analyze.remove_noise(tokens, stop_words))

    for tokens in negative_tweet_tokens:
        negative_cleaned_tokens_list.append(analyze.remove_noise(tokens, stop_words))

    all_pos_words = analyze.get_all_words(positive_cleaned_tokens_list)

    freq_dist_pos = FreqDist(all_pos_words)
    print(freq_dist_pos.most_common(10))

    positive_tokens_for_model = analyze.get_tweets_for_model(positive_cleaned_tokens_list)
    negative_tokens_for_model = analyze.get_tweets_for_model(negative_cleaned_tokens_list)

    positive_dataset = [(tweet_dict, "Positive")
                         for tweet_dict in positive_tokens_for_model]

    negative_dataset = [(tweet_dict, "Negative")
                         for tweet_dict in negative_tokens_for_model]

    dataset = positive_dataset + negative_dataset

    print(len(dataset))

    random.shuffle(dataset)

    train_data = dataset[:7000]
    test_data = dataset[7000:]

    classifier = NaiveBayesClassifier.train(train_data)

    print("Accuracy is:", classify.accuracy(classifier, test_data))

    print(classifier.show_most_informative_features(10))

    custom_tweet = "I am very depressed with Trump and what he said."

    custom_tokens = analyze.remove_noise(word_tokenize(custom_tweet), stop_words)

    print(custom_tweet, classifier.classify(dict([token, True] for token in custom_tokens)))

    logging.basicConfig(format='%(asctime)s {%(pathname)s:%(lineno)d} %(levelname)s %(message)s',
                        filename='application.log',
                        filemode='w',
                        level=logging.INFO,
                        datefmt='%m/%d/%Y %I:%M:%S %p')

    parser = tweet_parser.TweetParser()
    # for key in handle:
    #     tweets = parser.get_tweets(tweet_criteria=parser.set_tweet_criteria_keyword(keyword=key, max_tweets=4000))
    #     processed_tweets = parser.process_all_tweets(tweets=tweets)
    #     new_tweet_file = file_work.FileOperations('tweets.txt')
    #     new_tweet_file.write(processed_tweets)
    #     new_tweet_file.remove_blines()
    
    my_train_data, my_test_data = analyze.convert_to_trainable_dataset()

    parser.remove_stopwords()
    new_file = file_work.FileOperations('FilteredTweet.txt').read()
    logging.info(analyze.posneg_write(new_file))
    analysis = file_work.FileOperations('analyzed.txt').write([json.dumps(analyze.posneg_write(new_file))])

    my_classifier = analyze.nb_classifier(my_train_data, my_test_data)
    print('My classifier accuracy: ', analyze.test_classifier(my_classifier, my_test_data))
    print(my_classifier.show_most_informative_features(10))
    print('classifier accuracy: ', analyze.test_classifier(classifier, my_test_data))

    print(custom_tweet, my_classifier.classify(dict([token, True] for token in custom_tokens)))
    # pickel_file = 'trainedModel.pkl'
    # with open(pickel_file, 'wb') as file:
    #pickle.dump(my_classifier,open('trainedModel.pkl','wb'))
    # logging.info(classifier.classify(my_test_data[:40]))
    # logging.info(my_classifier.classify(my_test_data[:40]))


if __name__ == '__main__':
    main()
