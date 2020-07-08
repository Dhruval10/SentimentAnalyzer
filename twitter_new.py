import re
import pandas as pd
import matplotlib.pyplot as plt
import string
import got3
from nltk import NaiveBayesClassifier
from nltk.corpus import stopwords 
stopwords_english = stopwords.words('english')
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
from nltk.tokenize import TweetTokenizer
from langdetect import detect

since_date = '2020-03-08'
until_date = '2020-03-10'
##words = ['economy','GDP','stock market','Unemployment','jobless claims']
#handle = 'stock market'
count = 100

# Happy Emoticons
emoticons_happy = set([
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
    '<3'
    ])
 
# Sad Emoticons
emoticons_sad = set([
    ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '>:\\', ';('
    ])
 
# all emoticons (happy + sad)
emoticons = emoticons_happy.union(emoticons_sad)

max_error_count = 4
min_text_length = 3

# def is_in_english(quote):
#     d = SpellChecker("en_US")
#     d.set_text(quote)
#     errors = [err.word for err in d]
#     if ((len(errors) > max_error_count) or len(quote.split()) < min_text_length):
#         return False
#     else:
#         return True


# bag_of_words() will tokenize a tweet, create a dictionary and will return it
def bag_of_words(tweet):
    words = tokenize_tweet(tweet)
    words_dictionary = dict([word, True] for word in words)    
    return words_dictionary

# clean_tweet will remove hyperlink, URLs and hashtag and will return clean tweet
def clean_tweet(tweet):
    
    tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) |(\w+:\/\/\S+)| ((www\.[^\s]+)|(https?://[^\s]+)|(pic\.[^\s]+))", " ", tweet).split())
    return tweet 

# tokenize_tweet() will tokenize a tweet, remove stop words, emojis and punctuation
def tokenize_tweet(tweet):
    
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)
 
    filtered_token = []
    for word in tweet_tokens:
        if (word not in stopwords_english and #remove stopwords
            word not in emoticons and #remove emoticons
                word not in string.punctuation): #remove punctuation
            filtered_token.append(word)
    return filtered_token

# def get_tweets(keyword,count):
#
#     # fileio = open("tweets.txt",'w', encoding = 'utf-8')
#     tweetCriteria = got3.manager.TweetCriteria().setSince("2020-03-08").setUntil("2020-03-10").setQuerySearch(keyword).setMaxTweets(count)
#
#     print(len(str(got3.manager.TweetManager.getTweets(tweetCriteria))))
#     for i in range(count):
#
#         print(str(i)+ " " + keyword)
#         tweet1 = got3.manager.TweetManager.getTweets(tweetCriteria)[i]
#         # tweet_value1 = is_in_english(str(tweet1.text))
#         print("en not checked:" + str(tweet1.text))
        # if(tweet_value1 is True):
        #     print(tweet1.text)
        #     fileio.write(tweet1.text+"\n")
        # else:
        #     pass

# get_tweets() will call got3 and fetch all tweets. Detect() will filter tweets and will save only english tweets in the file
def get_tweets():
    words = ['economy','gdp','stock market','unemployment','investment']
    for handle in words:
        print(handle+" ")
        tweet_criteria = got3.manager.TweetCriteria().setSince(since_date).setUntil(until_date).setQuerySearch(handle).setMaxTweets(count)
        fileio = open("tweets.txt",'a', encoding = 'utf-8')
        for i in range(count):
            tweets = got3.manager.TweetManager().getTweets(tweet_criteria)[i]
            if(detect(tweets.text) == 'en'):
                fileio.write(tweets.text+"\n")
                print(tweets.text)
        
def max_prob(tweet):
    prob_result = classifier.prob_classify(tweet)
    max_result = prob_result.max()
    return max_result


def neg_prob(tweet):
    prob_result = classifier.prob_classify(tweet)
    negative_prob = prob_result.prob("negative")
    return negative_prob


def pos_prob(tweet):
    prob_result = classifier.prob_classify(tweet)
    positive_prob = prob_result.prob("positive")
    return positive_prob 
    

###################### main ################################################

filename1 = "PositiveWords.txt"
print("Reading Pos file ---------------------------------------------------")

with open (filename1, 'r', encoding = 'utf-8') as filehandle1:
    pos = filehandle1.read().replace('\n',', ')
    #print("length:", len(pos))
    #print("pos:", pos)
    

filename2 = "NegativeWords.txt"

print("Reading Neg file ---------------------------------------------------")

with open (filename2, 'r', encoding = 'utf-8') as filehandle2:
    neg = filehandle2.read().replace('\n',', ')
    #print("neg:", neg)

    
pos_tweets = []
pos_tweets.append((bag_of_words(pos), 'positive'))

#print("pos tweets: ", pos_tweets)
print("Length of pos tweets: ", len(pos_tweets))

neg_tweets = []
neg_tweets.append((bag_of_words(neg), 'negative'))

#print("neg tweets: ", neg_tweets)
print("Length of neg tweets: ", len(neg_tweets))


train = pos_tweets + neg_tweets
#print("train ", train)
print("Length of train:", len(train))
 
classifier = NaiveBayesClassifier.train(train)
print("Classifier..........", classifier)

################################################

print("Before get tweets....")
##words = ['stock market','Unemployment','economy','GDP']

tweets = get_tweets()

print("After get tweets.....")
#max_tweets = 10

#     for line in filehandle:
#         print(line)

filename3 = 'tweets.txt'

# Create a dataframe with a column called Tweets
df = pd.read_csv(filename3, sep=";", names=['Tweets'])
df.head()
print("\n")
print("Creating data frame .....\n")
print(df)

    
df['Tweets'] = df['Tweets'].apply(clean_tweet)
print("\n")
print("Cleaned tweets....\n")
print(df)

filename4 = "cleaned_tweets.txt"
print("--------------------------------------------------- Opening file again ---------------------------------------------------")
with open(filename4, 'w',encoding="utf-8") as filehandle:
    df.to_csv(r'cleaned_tweets.txt', header=None, index=None, sep=' ', mode='a')
    print("\n")
print(".........................Redirected cleaned tweets to the file..................")
print("\n")

    
df['Tokenized_Tweets'] = df['Tweets'].apply(bag_of_words)
print("\n")
print("Cleaned and Tokenized tweets....\n")
print(df)

filename5 = "tokenized_tweets.txt"
print("--------------------------------------------------- Opening file again ---------------------------------------------------")
with open(filename5, 'w',encoding="utf-8") as filehandle:
    df.to_csv(r'tokenized_tweets.txt', header=None, index=None, sep=' ', mode='a')
    print("\n")
print(".........................Redirected tokenized tweets to the file..................")
print("\n")

df['Positive_Prob'] = df['Tokenized_Tweets'].apply(pos_prob)
df['Negative_Prob'] = df['Tokenized_Tweets'].apply(neg_prob)


df['Sentiment'] = df['Tokenized_Tweets'].apply(max_prob)

df.loc[df.Positive_Prob == 0.5, "Sentiment"] = "neutral"
print("The output....")
print(df)


filename6 = "updated_tweets.txt"
print("--------------------------------------------------- Opening file again ---------------------------------------------------")
with open(filename6, 'w',encoding="utf-8") as filehandle:
    df.to_csv(r'updated_tweets.txt', header=None, index=None, sep=' ', mode='a')
    print("\n")
print(".........................Redirected everything to the file..................")
print("\n")
#with open(filename1, 'r',encoding="utf-8") as filehandle:
 #   print("Printing from file")
  #  for line in filehandle:
   #     print(line)


# Print the percentage of positive tweets
ptweets = df[df.Sentiment == 'positive']
ppos = round( (ptweets.shape[0] / df.shape[0]) * 100 , 1)
print("% of Positive Tweets....", ppos)

# Print the percentage of negative tweets
ntweets = df[df.Sentiment == 'negative']
pneg = round( (ntweets.shape[0] / df.shape[0]) * 100 , 1)
print("% of Negative Tweets...", pneg)

# Print the percentage of neutral tweets
neutweets = df[df.Sentiment == 'neutral']
#neutweets = neutweets['Tweets']
pneut = round( (neutweets.shape[0] / df.shape[0]) * 100 , 1)
print("% of Neutral Tweets...", pneut)

value = ppos, pneg, pneut
print("Value is:", value)

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
sentiment = ['Positive', 'Negative', 'Neutral']
percentage = [ppos, pneg, pneut]
ax.bar(sentiment,percentage)
plt.show()
