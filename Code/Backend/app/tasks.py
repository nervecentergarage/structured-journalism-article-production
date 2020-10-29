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

from pymongo import MongoClient 

app = Celery()
app.config_from_object("celery_settings")

client = MongoClient("mongodb+srv://TestAdmin:admintest@cluster0.toaff.mongodb.net/devDB?ssl=true&ssl_cert_reqs=CERT_NONE")

def fetch_news(url_list): 

    all_news = []

    for news_source in url_list:

        feed_url = fp.parse(news_source)

#         feed_url = news_source
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

            article_dict.update({'source_name': source_name, 'source_url': url_feed, "article_url": artilce_url, 'image_url': content.top_image,'video_url': content.movies, 'publish_date':publish_date,'title':title, 'article': full_article, 'author':author, "summary": content.summary, "keywords": content.keywords})
            article_list.append(article_dict)

        all_news.extend(article_list)

    return all_news



@app.task
def hello():
    print("hello")

@app.task
def scrapeNews():
    news_list = ["https://sports.yahoo.com/rss/"]

    # define date format
    fmt = '%Y-%m-%dT-%H-%M%Z%z'
    # define eastern timezone
    eastern = timezone('US/Eastern')
    # naive datetime
    naive_dt = datetime.now()
    loc_dt = datetime.now(eastern)
    start_time = naive_dt.strftime(fmt)

    print("Download started for news_list:", start_time)

    news = fetch_news(news_list) # Fetching the news

    test_news_collection =  client.test_news_collection  # DB name
    test_news_collection.insert_many(news) # Inserting the articles to mongodb

    print("Complete")
