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
warnings.filterwarnings('ignore')
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
import ssl
from elasticsearch import Elasticsearch
from elasticsearch.connection import create_ssl_context

app = Celery()
app.config_from_object("celery_settings")
#app.control.rate_limit('app.scrape_news', '1/m')
np.random.seed(2018)

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('vader_lexicon')
nltk.download('stopwords')

locale.getdefaultlocale()

stemmer = SnowballStemmer('english')

s = SentimentIntensityAnalyzer()

def fetch_news(url_list, category, collection): 
    all_news = []
    latest_article_lower = 0
    latest_article_higher = 0

    #get the latest article id in the collection. set as 1 if not found
    try:
        article_id = int(collection.find().skip(collection.count_documents({}) - 1)[0]['article_id']) + 1
    except:
        article_id = 1
    
    latest_article_lower = article_id

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

                # Processing the document with NLP tasks to extract 'Summary' and 'Keywords' from the article.
                content.nlp()  
            except:
                pass

            # Updating all the information to a dictionary

            article_dict.update({'article_id': article_id, 'source_name': source_name, 'source_url': url_feed, "article_url": artilce_url, 'image_url': content.top_image,'video_url': content.movies, 'publish_date':content.publish_date,'title':content.title, 'article': content.text, 'author':content.authors, "summary": content.summary, "keywords": content.keywords, "category": category})
            article_list.append(article_dict)
            latest_article_higher = article_id
            article_id += 1

        all_news.extend(article_list)

    articles = collection.insert_many(all_news)
    return latest_article_lower, latest_article_higher

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
            final_snip["publish_date"] = i["publish_date"]
            final_snip["source_url"] = i["source_url"]
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

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    stops = set(stopwords.words("english"))
    stops_rm = set(
        ['above', 'against', 'ain', 'any', 'aren', 'because', 'below', 'didn', 'couldn', 'doesn', 'does', 'down',
            'few', 'hadn', 'isn', "isn't", 'mightn', 'more', 'most', 'mustn', 'no', 'nor', 'not', 'now', 'off', 'out',
            'over', 'too', 'under', 'until', 'up', 've', 'very'])
    stops = stops.difference(stops_rm)
    
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in stops and len(token) >= 2:
            result.append(lemmatize_stemming(token))
    return result

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

    #elastic search connection
    context = create_ssl_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    es = Elasticsearch(
    "https://6c7e6efaa2574715a49ff2ea9757622d.eastus2.azure.elastic-cloud.com",
    http_auth=(os.environ.get('ELASTIC_USER'), os.environ.get('PASS_ELASTIC')),
    # scheme="https",
    port=9243,
    ssl_context = context,
    )

    lda_model_tfidf = gensim.models.ldamodel.LdaModel(corpus_tfidf, num_topics=25, id2word=dicti, passes=2)  # Applying the LDA Model for Topic Formations and clustering

    topics=lda_model_tfidf.show_topics(num_topics=25,formatted=False, num_words= 50)
    topic_collection=[]
    for topic in topics:
        topic_dict={}
        topic_dict[str(topic[0])] = [i[0] for i in topic[1]]
        topic_collection.append(topic_dict)
        new_data={"topic_id": topic[0],"keywords": ' '.join(topic_dict[str(topic[0])])} 
        response = es.index(index = 'article_production',id = topic[0],body = new_data) #Storing the topic ID's and the keywords in ElasticSearch

    ## mongo new collection topics
    data_layer = {
    "connection_string": "mongodb+srv://TestAdmin:admintest@cluster0.toaff.mongodb.net/devDB?ssl=true&ssl_cert_reqs=CERT_NONE",
    "collection_name": "article_production",
    "database_name": "article_production"
    }
    db_connect = MongoClient(data_layer["connection_string"])
    database=db_connect[data_layer['database_name']]
    collection=database[data_layer['collection_name']]
    collection.insert_many(topic_collection) # Storing the data in the Interim Database

    tmp_list=[]
    tmp_list1=[]
    for k in bow_corpus:
        results=lda_model_tfidf[k]
        tmp_list.append(sorted(results, key=lambda tup: -1*tup[1])[0][0])
        tmp_list1.append(sorted(results, key=lambda tup: -1*tup[1])[0][1])
    se = pd.Series(tmp_list)
    se1 = pd.Series(tmp_list1)
    df['topic'] = se.values
    df["percentage"] = se1.values
    final_df = df[["type",	"snip_id",	"content",	"parent_article",	"parent_article_url",	"publish_date",	"source_url",	"author",	"category",	"snippet_url",		"processed_text",		"compound",	"topic", "percentage"]]
    topic_json=json.loads(final_df.to_json(orient="records"))

    return topic_json

def get_topic_json(data):  # reads raw data json
    snippets = snip_json(data)  # Storing the snippets json

    stops = set(stopwords.words("english"))  # Defining the Stop words

    final_data = attach_topics(snippets)

    return final_data

@app.task
def scrape_news():
    print("Scraping news...")
    sports_list = ["https://sports.yahoo.com/rss/","https://www.huffingtonpost.com/section/sports/feed",
               "https://rss.nytimes.com/services/xml/rss/nyt/Sports.xml", "http://feeds.bbci.co.uk/sport/rss.xml"
               "http://rss.cnn.com/rss/edition_sport.rss","https://www.theguardian.com/uk/sport/rss",
               "http://rssfeeds.usatoday.com/UsatodaycomSports-TopStories"]

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

    
    # define date format
    fmt = '%Y-%m-%dT-%H-%M%Z%z'
    # naive datetime
    naive_dt = datetime.now()
    start_time = naive_dt.strftime(fmt)

    print("Download started for all lists:", start_time)

    client = MongoClient("mongodb+srv://TestAdmin:admintest@cluster0.toaff.mongodb.net/devDB?ssl=true&ssl_cert_reqs=CERT_NONE")

    db = client.news  # DB name

    print("1/6: Collecting sports news")
    sports_collection =  db.sports_collection  # DB name
    fetch_news(sports_list, "sports", sports_collection) # Fetching the news

    print("2/6: Collecting politics news")
    politics_collection = db.politics_collection  # DB name
    fetch_news(politics_list, "politics", politics_collection)  # Fetching the news
    
    print("3/6: Collecting health news")    
    health_collection = db.health_collection  # DB name
    fetch_news(health_list, "health", health_collection)  # Fetching the news

    print("4/6: Collecting finance news")        
    finance_collection = db.finance_collection  # DB name
    fetch_news(finance_list, "finance", finance_collection)  # Fetching the news

    print("5/6: Collecting environment news")        
    environment_collection = db.environment_collection  # DB name
    fetch_news(environment_list, "environment", environment_collection)  # Fetching the news

    print("6/6: Collecting scitech news")        
    scitech_collection = db.scitech_collection  # DB name
    fetch_news(scitech_list, "scitech", scitech_collection) # Fetching the news

    print("Scraping complete")

@app.task
def extract_snippets():
    print("Starting Extract Snippets Task")
    client = MongoClient("mongodb+srv://TestAdmin:admintest@cluster0.toaff.mongodb.net/devDB?ssl=true&ssl_cert_reqs=CERT_NONE")
    db = client.news  

    snippet_collection = db.snippet_collection 

    sports_collection =  db.sports_collection  
    politics_collection =  db.politics_collection  
    health_collection =  db.health_collection  
    finance_collection =  db.finance_collection  
    environment_collection = db.environment_collection  
    scitech_collection = db.scitech_collection  

    news_collections = []
    news_collections.extend([sports_collection, politics_collection, health_collection, finance_collection, environment_collection, scitech_collection])

    n = 1
    for collection in news_collections:
        print("Extracting", str(n) + "/6 snippets...")
        data = list(collection.find())
        snippets = get_topic_json(data)
        snippet_collection.insert_many(snippets)
        n += 1

@app.task
def scrape_snip():
    scrape_news()
    extract_snippets()

@app.task
def scrape_snip_loop():
    scrape_news()
    extract_snippets()
    scrape_snip_loop.delay()

@app.task
def scrape_snip_latest():
    scrape_snip_latest_news()
    
def scrape_snip_latest_news():
    print("Scraping and Snipping latest news...")
    sports_list = ["https://sports.yahoo.com/rss/","https://www.huffingtonpost.com/section/sports/feed",
               "https://rss.nytimes.com/services/xml/rss/nyt/Sports.xml", "http://feeds.bbci.co.uk/sport/rss.xml"
               "http://rss.cnn.com/rss/edition_sport.rss","https://www.theguardian.com/uk/sport/rss",
               "http://rssfeeds.usatoday.com/UsatodaycomSports-TopStories"]

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

    
    # define date format
    fmt = '%Y-%m-%dT-%H-%M%Z%z'
    # naive datetime
    naive_dt = datetime.now()
    start_time = naive_dt.strftime(fmt)

    print("Download started for all lists:", start_time)

    client = MongoClient("mongodb+srv://TestAdmin:admintest@cluster0.toaff.mongodb.net/devDB?ssl=true&ssl_cert_reqs=CERT_NONE")

    db = client.news  # DB name
    snippet_collection = db.snippet_collection 

    start_article = 0
    end_article = 0

    print("1/6: Collecting sports news")
    sports_collection =  db.sports_collection  # DB name
    start_article, end_article = fetch_news(sports_list, "sports", sports_collection) # Fetching the news
    print("1/6: Extracting sports news snippets...")
    data = list(sports_collection.find({"article_id" : {"$gte":start_article, "$lt":end_article+1}}))
    snippets = get_topic_json(data)
    snippet_collection.insert_many(snippets)

    print("2/6: Collecting politics news")
    politics_collection = db.politics_collection  # DB name
    start_article, end_article = fetch_news(politics_list, "politics", politics_collection)  # Fetching the news
    print("2/6: Extracting politics news snippets...")
    data = list(politics_collection.find({"article_id" : {"$gte":start_article, "$lt":end_article+1}}))
    snippets = get_topic_json(data)
    snippet_collection.insert_many(snippets)
    
    print("3/6: Collecting health news")    
    health_collection = db.health_collection  # DB name
    start_article, end_article = fetch_news(health_list, "health", health_collection)  # Fetching the news
    print("3/6: Extracting health news snippets...")
    data = list(health_collection.find({"article_id" : {"$gte":start_article, "$lt":end_article+1}}))
    snippets = get_topic_json(data)
    snippet_collection.insert_many(snippets)

    print("4/6: Collecting finance news")        
    finance_collection = db.finance_collection  # DB name
    start_article, end_article = fetch_news(finance_list, "finance", finance_collection)  # Fetching the news
    print("4/6: Extracting finance news snippets...")
    data = list(finance_collection.find({"article_id" : {"$gte":start_article, "$lt":end_article+1}}))
    snippets = get_topic_json(data)
    snippet_collection.insert_many(snippets)

    print("5/6: Collecting environment news")        
    environment_collection = db.environment_collection  # DB name
    start_article, end_article = fetch_news(environment_list, "environment", environment_collection)  # Fetching the news
    print("5/6: Extracting environment news snippets...")
    data = list(environment_collection.find({"article_id" : {"$gte":start_article, "$lt":end_article+1}}))
    snippets = get_topic_json(data)
    snippet_collection.insert_many(snippets)

    print("6/6: Collecting scitech news")        
    scitech_collection = db.scitech_collection  # DB name
    start_article, end_article = fetch_news(scitech_list, "scitech", scitech_collection) # Fetching the news
    print("6/6: Extracting scitech news snippets...")
    data = list(scitech_collection.find({"article_id" : {"$gte":start_article, "$lt":end_article+1}}))
    snippets = get_topic_json(data)
    snippet_collection.insert_many(snippets)

    print("Scraping and Snipping of latest Articles complete")