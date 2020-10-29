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

from tasks import scrapeNews
from . import data_blueprint


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


@data_blueprint.route('/getData/', methods=['GET'])
def getData():

    news_collection = mongo.db.testData #Replace testData to actual DB
    output = []
    try:
        news_articles = loads(dumps(news_collection.find({"topic": "General"}))) # make topic passable
        
        for article in news_articles:

            # structure schema here
            output.append({
                'title': article['title'],
                'author': article['author'],
                'body': article['body']
            })
        if len(output) == 0:
            output = {'code': 2, "error": "User not found"}
        else:
            output = {"count": len(output), "results": output}
    except:
        output = {'code': 2, "error": "Error fetching details from DB"}

    return output



@data_blueprint.route('/scrapeCategoryNews/', methods=['POST'])
def scrapeCategoryNews():
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
    # define eastern timezone
    eastern = timezone('US/Eastern')
    # naive datetime
    naive_dt = datetime.now()
    loc_dt = datetime.now(eastern)
    start_time = naive_dt.strftime(fmt)

    print("Download started for sports_list:", start_time)

    sports_news = fetch_news(sports_list) # Fetching the news

    sports_collection =  mongo.db.sports_collection  # DB name
    sports_collection.insert_many(sports_news) # Inserting the articles to mongodb

    politics_news = fetch_news(politics_list)  # Fetching the news

    politics_collection = mongo.db.politics_collection  # DB name
    politics_collection.insert_many(politics_news) # Inserting the articles to mongodb

    health_news = fetch_news(health_list)  # Fetching the news

    health_collection = mongo.db.health_collection  # DB name
    health_collection.insert_many(health_news) # Inserting the articles to mongodb

    finance_news = fetch_news(finance_list)  # Fetching the news

    finance_collection = mongo.db.finance_collection  # DB name
    finance_collection.insert_many(finance_news) # Inserting the articles to mongodb

    environment_news = fetch_news(environment_list)  # Fetching the news

    environment_collection = mongo.db.environment_collection  # DB name
    environment_collection.insert_many(environment_news) # Inserting the articles to mongodb

    scitech_news = fetch_news(scitech_list) # Fetching the news

    scitech_collection = mongo.db.scitech_collection  # DB name
    scitech_collection.insert_many(scitech_news) # Inserting the articles to mongodb

    print("Complete")
    return "Completed"


@data_blueprint.route('/postData/', methods=['POST'])
def postData():
    request_json = request.get_json()
    url_list = request_json['url_list']
    all_news = []

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
                content.parse()
                content.nlp()  
                

            except:
                pass

            # Updating all the information to a dictionary

            article_dict.update({'source_name': source_name, 'source_url': url_feed, "article_url": artilce_url, 'image_url': content.top_image,'video_url': content.movies, 'publish_date':content.publish_date,'title':content.title, 'article': content.text, 'author':content.authors, "summary": content.summary, "keywords": content.keywords})
            article_list.append(article_dict)

        all_news.extend(article_list)

    fmt = '%Y-%m-%dT-%H-%M%Z%z'
    # define eastern timezone
    eastern = timezone('US/Eastern')
    # naive datetime
    naive_dt = datetime.now()
    loc_dt = datetime.now(eastern)
    start_time = naive_dt.strftime(fmt)

    print("Download started:", start_time)

    news_db = mongo.db.testData
    news_db.insert_many(all_news)

    print("Complete")
    
    return "Completed"
    

@data_blueprint.route('/scrapeURL/', methods=['GET'])
def scrapeURL():
    
    scrapeNews.delay()
    return "Scrape URL called"