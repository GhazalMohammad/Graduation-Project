from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import numpy as np
from textblob import TextBlob
import pandas as pd
from io import BytesIO
import base64
import numpy as np
import tweepy
app = Flask(__name__)


def get_tweets(topic, count):
    consumerKey = 'kGGFOcIVBZRxExoct2nQwrlOI'
    consumerSecret = 'F66rwDS0dadWJzmuors8JGnez9q53JCpMER76nD6LCELDXkBLF'
    accessToken = '1579551912587927553-vGRMDNwQwXYC2JgBZ4sB2BBPaQUjWC'
    accessTokenSecret = 'ra1EJ5zdDZbF6sjD0rHzINb7ljKt4ioaDxRWjGwXJfMZH'

    # Creating the authentication object
    auth = tweepy.OAuthHandler(consumerKey, consumerSecret)

    # Setting your access token and secret
    auth.set_access_token(accessToken, accessTokenSecret)

    # Creating the API object while passing in auth information
    api = tweepy.API(auth)

    # Tweets list
    tweets = []

    # Using Cursor to get tweets
    for tweet in tweepy.Cursor(api.search_tweets, q=topic, tweet_mode='extended').items(count):

        # Empty dictionary to store required params of a tweet
        parsed_tweet = {}

        # Saving text of tweet
        parsed_tweet['text'] = tweet.full_text

        # Saving sentiment of tweet
        analysis = TextBlob(tweet.full_text)
        parsed_tweet['sentiment'] = 'positive' if analysis.sentiment.polarity >= 0 else 'negative'

        # Appending parsed tweet to tweets list
        tweets.append(parsed_tweet)

    # Return parsed tweets
    return tweets


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    topic = request.form['topic']
    num_tweets = int(request.form['number_of_tweets'])

    # get tweets
    tweets = get_tweets(topic, num_tweets)

    # perform sentiment analysis
    positive_tweets = 0
    negative_tweets = 0
    for tweet in tweets:
        analysis = TextBlob(tweet['text'])
        if analysis.sentiment.polarity >= 0:
            positive_tweets += 1
        else:
            negative_tweets += 1

    # calculate percentages
    total_tweets = len(tweets)
    positive_percent = round((positive_tweets / total_tweets) * 100, 2)
    negative_percent = round((negative_tweets / total_tweets) * 100, 2)

    # generate plot
    x = ['Positive', 'Negative']
    y = [positive_percent, negative_percent]
    plt.bar(x, y)
    plt.title('Sentiment Analysis')
    plt.xlabel('Sentiment')
    plt.ylabel('Percentage')

    # save plot to file
    image_name = 'plot.png'
    plt.savefig('static/' + image_name)

    # render template with predictions and plot
    prediction_text = f'Positive: {positive_percent}%, Negative: {negative_percent}%'
    return render_template('results.html', prediction_text=prediction_text, image_name=image_name, tweets=tweets)


if __name__ == '__main__':
    app.run(debug=True)
