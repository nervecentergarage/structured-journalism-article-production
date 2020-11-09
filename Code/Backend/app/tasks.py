from celery import Celery
from flask import render_template
from flask import request
from bson.json_util import dumps, RELAXED_JSON_OPTIONS
from bson.objectid import ObjectId
import json
from json import loads
from bson import json_util
from project import mongo
from flask import jsonify
from functools import wraps
import re
import newspaper
from newspaper import Article
import time
from time import mktime
from datetime import datetime
from pytz import timezone
import bs4
import feedparser as fp
import pandas as pd
import nltk
import warnings
import gensim
import numpy as np
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from gensim.utils import simple_preprocess
from nltk.stem.porter import *
from gensim import corpora, models
from nltk import tokenize
from collections import defaultdict, Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import locale
from pprint import pprint
from mpl_toolkits.mplot3d import Axes3D

from pymongo import MongoClient 

app = Celery()
app.config_from_object("celery_settings")
app.control.rate_limit('app.scrape_news', '10/m')
np.random.seed(2018)

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('vader_lexicon')

locale.getdefaultlocale()

stemmer = SnowballStemmer('english')

s = SentimentIntensityAnalyzer()

def fetch_news(url_list): #, category, collection): 
    all_news = []

    #get the latest article id in the collection. set as 1 if not found
    #try:
    #article_id = int(collection.find().skip(collection.count_documents({}) - 1)[0]['article_id']) + 1
    #except:
    article_id = 1

    for news_source in url_list:

        feed_url = fp.parse(news_source)

        source_name = news_source
        source_url = news_source
        url_feed = news_source
        article_list = []

        for article in feed_url.entries:
            article_dict = {}

            date = article.published_parsed
            artilce_url = article.link

            content = Article(artilce_url) #Newspapaer 3k's Article module to read the contents
            try: 
                content.download() #Downloading the News article
                content.parse()    #Downloading the News article
                title = content.title #Getting the Title of the article
                author = content.authors #Getting the Author of the article
                publish_date = content.publish_date  #Getting the publish date of the article 
                full_article = content.text #Getting the complete content of the article 

                # Processing the document with NLP tasks to extract 'Summary' and 'Keywords' from the article.

                content.nlp()  
            except:
                pass

            # Updating all the information to a dictionary

            article_dict.update({'article_id': article_id, 'source_name': source_name, 'source_url': url_feed, "article_url": artilce_url, 'image_url': content.top_image,'video_url': content.movies, 'publish_date':publish_date,'title':title, 'article': full_article, 'author':author, "summary": content.summary, "keywords": content.keywords}) #, "category": category})
            article_list.append(article_dict)
            article_id += 1

        all_news.extend(article_list)

    return all_news


def get_topic_json(data):  # reads raw data json
    # Function to split the artilce into paragraphs.
    def snips(article):
        new_list = []
        li = article.split("\n\n")

        for i in range(len(li)):
            if i >= 1:
                if len(li[
                           i].split()) <= 20:  # Checking the length of the para and if less than 20, joining it with the other paragraph.
                    new_list[-1].join([" ", li[i]])
                else:
                    new_list.append(li[i])
            else:
                new_list.append(li[i])

        return new_list  # returns a list of paragraphs

    # Function for snippets of the articles

    def snip_json(article_data):
        snippets = []
        k = 1
        for i in article_data:
            snips_pre = snips(i["article"])
            for j in snips_pre:
                final_snip = {}
                final_snip["type"] = "text"
                final_snip["snip_id"] = k
                final_snip["content"] = j
                # final_snip["snippet_summary"] = summarize_paragraph(j) # Summarizing the snippet

                final_snip["parent_article"] = i["title"]
                final_snip["parent_article_url"] = i["article_url"]
                # final_snip["image_url"]=i["image_url"]
                final_snip["publish_date"] = i["publish_date"]
                final_snip["source_url"] = i["source_url"]
                # final_snip["video_url"]=i['video_url']
                final_snip["publish_date"] = i["publish_date"]
                final_snip["author"] = i["author"]
                final_snip["category"] = i['category']
                final_snip["snippet_url"] = ""
                snippets.append(final_snip)
                k += 1

                if i["image_url"] != "":
                    img_snip = {}
                    img_snip["type"] = "image"
                    img_snip["snippet_url"] = i["image_url"]
                    img_snip["snip_id"] = k
                    img_snip["content"] = i["summary"]
                    img_snip["parent_article"] = i["title"]
                    img_snip["parent_article_url"] = i["article_url"]
                    img_snip["publish_date"] = i["publish_date"]
                    img_snip["source_url"] = i["source_url"]
                    img_snip["publish_date"] = i["publish_date"]
                    img_snip["author"] = i["author"]
                    img_snip["category"] = i['category']
                    snippets.append(img_snip)
                    k += 1

                elif i["video_url"] != "":                    
                    vid_snip = {}
                    vid_snip["type"] = "video"
                    vid_snip["snippet_url"] = i["video_url"]
                    vid_snip["snip_id"] = k
                    vid_snip["content"] = i["summary"]
                    vid_snip["parent_article"] = i["title"]
                    vid_snip["parent_article_url"] = i["article_url"]
                    vid_snip["publish_date"] = i["publish_date"]
                    vid_snip["source_url"] = i["source_url"]
                    vid_snip["publish_date"] = i["publish_date"]
                    vid_snip["author"] = i["author"]
                    vid_snip["category"] = i['category']
                    snippets.append(vid_snip)
                    k += 1

        return snippets

    snippets = snip_json(data)  # Storing the snippets json

    stops = set(stopwords.words("english"))  # Defining the Stop words

    # Performing the Text Cleaning

    def lemmatize_stemming(text):
        return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

    def preprocess(text):
        result = []
        for token in gensim.utils.simple_preprocess(text):
            if token not in stops and len(token) >= 2:
                result.append(lemmatize_stemming(token))
        return result

    # Topic Clustering of the articles and snippets using LDA Topic Model

    def attach_topics(snippets):

        stops = set(stopwords.words("english"))
        stops_rm = set(
            ['above', 'against', 'ain', 'any', 'aren', 'because', 'below', 'didn', 'couldn', 'doesn', 'does', 'down',
             'few', 'hadn', 'isn', "isn't", 'mightn', 'more', 'most', 'mustn', 'no', 'nor', 'not', 'now', 'off', 'out',
             'over', 'too', 'under', 'until', 'up', 've', 'very'])
        stops = stops.difference(stops_rm)

        df = pd.DataFrame(snippets)  # Converting to Dataframe
        df['processed_text_corpus'] = df['content'].map(preprocess)  # Perfroming the text cleaning
        df['processed_text'] = df['processed_text_corpus'].apply(lambda x: ' '.join(x))
        df["sentimentscores"] = df["content"].apply(lambda x: s.polarity_scores(x))
        df = pd.concat([df.drop(['sentimentscores'], axis=1), df['sentimentscores'].apply(pd.Series)], axis=1)
        df = df.loc[:, ~df.columns.duplicated()]
        processed_docs = df.processed_text_corpus.tolist()
        dicti = gensim.corpora.Dictionary(processed_docs)
        bow_corpus = [dicti.doc2bow(doc) for doc in processed_docs]  # Collecting the Bag of Words for the dictionary
        tfidf = models.TfidfModel(bow_corpus)  # Applying the Tf-IDF Model for the collection of words
        corpus_tfidf = tfidf[bow_corpus]

        lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=25, id2word=dicti, passes=2,
                                                     workers=2)  # Applying the LDA Model for Topic Formations and clustering

        tmp_list = []
        for k in bow_corpus:
            tmp_list.append(sorted(lda_model_tfidf[k], key=lambda tup: -1 * tup[1])[0][0])

        se = pd.Series(tmp_list)
        df['topic'] = se.values  # Defining the topic by the collection of Top Words from the clusters

        final_df = df[
            ["type", "snip_id", "content", "parent_article", "parent_article_url", "publish_date", "source_url",
             "author", "category", "snippet_url", "processed_text", "compound", "topic"]]
        topic_json = json.loads(final_df.to_json(orient="records"))

        return topic_json

    final_data = attach_topics(snippets)

    return final_data

@app.task
def scrape_news():
    print("Scraping news...")
    sports_list = ["https://sports.yahoo.com/rss/","https://www.huffingtonpost.com/section/sports/feed",
               "https://rss.nytimes.com/services/xml/rss/nyt/Sports.xml", "http://feeds.bbci.co.uk/sport/rss.xml"
               "http://rss.cnn.com/rss/edition_sport.rss","https://www.theguardian.com/uk/sport/rss",
               "http://rssfeeds.usatoday.com/UsatodaycomSports-TopStories"]

    '''
    politics_list = ["https://www.huffingtonpost.com/section/politics/feed", "http://feeds.foxnews.com/foxnews/politics"]

    health_list = ["https://rss.nytimes.com/services/xml/rss/nyt/Health.xml", "http://feeds.foxnews.com/foxnews/health"]

    finance_list = ["https://finance.yahoo.com/news/rssindex","https://www.huffingtonpost.com/section/business/feed",
                    "http://feeds.nytimes.com/nyt/rss/Business", "http://feeds.bbci.co.uk/news/business/rss.xml",
                    "https://www.theguardian.com/uk/business/rss", "http://rssfeeds.usatoday.com/UsatodaycomMoney-TopStories",
                    "https://www.wsj.com/xml/rss/3_7031.xml", "https://www.wsj.com/xml/rss/3_7014.xml"]

    environment_list = ["https://www.huffingtonpost.com/section/green/feed", "http://feeds.foxnews.com/foxnews/scitech",
                        "http://newsrss.bbc.co.uk/rss/newsonline_uk_edition/sci/tech/rss.xml",
                        "https://www.theguardian.com/uk/environment/rss"]

    scitech_list = ["http://feeds.nytimes.com/nyt/rss/Technology", "http://www.nytimes.com/services/xml/rss/nyt/Science.xml",
                "http://feeds.foxnews.com/foxnews/tech", "http://feeds.bbci.co.uk/news/technology/rss.xml",
                "https://www.theguardian.com/uk/technology/rss", "https://www.theguardian.com/science/rss",
                "https://www.wsj.com/xml/rss/3_7455.xml"]
    '''
    
    # define date format
    fmt = '%Y-%m-%dT-%H-%M%Z%z'
    # naive datetime
    naive_dt = datetime.now()
    start_time = naive_dt.strftime(fmt)

    print("Download started for sports_list:", start_time)

    client = MongoClient("mongodb+srv://TestAdmin:admintest@cluster0.toaff.mongodb.net/devDB?ssl=true&ssl_cert_reqs=CERT_NONE")

    db = client.news  # DB name

    sports_news = fetch_news(sports_list) #, "sports", sports_collection) # Fetching the news
    sports_collection =  db.sports_collection  # DB name
    #sports_news = fetch_news(sports_list, "sports", sports_collection) # Fetching the news
    articles = sports_collection.insert_many(sports_news) # Inserting the articles to mongodb

    #politics_collection = db.politics_collection  # DB name
    #politics_news = fetch_news(politics_list, "politics", politics_collection)  # Fetching the news
    #articles = politics_collection.insert_many(politics_news) # Inserting the articles to mongodb

    #health_collection = db.health_collection  # DB name  
    #health_news = fetch_news(health_list, "health", health_collection)  # Fetching the news
    #articles = health_collection.insert_many(health_news) # Inserting the articles to mongodb
    
    #finance_collection = db.finance_collection  # DB name
    #finance_news = fetch_news(finance_list, "finance", finance_collection)  # Fetching the news
    #articles = finance_collection.insert_many(finance_news) # Inserting the articles to mongodb

    #environment_collection = db.environment_collection  # DB name
    #environment_news = fetch_news(environment_list, "environment", environment_collection)  # Fetching the news
    #articles = environment_collection.insert_many(environment_news) # Inserting the articles to mongodb

    #scitech_collection = db.scitech_collection  # DB name
    #scietech_news = fetch_news(scitech_list, "scitech", scitech_collection) # Fetching the news
    #articles = scitech_collection.insert_many(scietech_news) # Inserting the articles to mongodb

    print("Scraping complete")

@app.task
def extract_snippets():
    client = MongoClient("mongodb+srv://TestAdmin:admintest@cluster0.toaff.mongodb.net/devDB?ssl=true&ssl_cert_reqs=CERT_NONE")
    db = client.news  # DB name

    snippet_collection = db.snippet_collection 

    news_collections = []

    sports_collection =  db.sports_collection  # DB name
    politics_collection =  db.politics_collection  # DB name
    health_collection =  db.health_collection  # DB name
    finance_collection =  db.finance_collection  # DB name
    environment_collection = db.environment_collection  # DB name
    scitech_collection = db.scitech_collection  # DB name

    news_collections.extend([sports_collection, politics_collection, health_collection, finance_collection, environment_collection, scitech_collection])

    for collection in news_collections:
        print("Extracting", collection, "snippets...")
        data = list(collection.find())
        snippets = get_topic_json(data)
        snippet_collection.insert_many(snippets)
        print("Completed", collection, "extraction")

