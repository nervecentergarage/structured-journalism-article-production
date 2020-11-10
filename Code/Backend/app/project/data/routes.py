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
import os

from tasks import scrape_news
from tasks import extract_snippets
from tasks import scrape_snip
from tasks import scrape_snip_loop
from tasks import scrape_snip_latest

import ssl
from elasticsearch import Elasticsearch
from elasticsearch.connection import create_ssl_context

import warnings
warnings.filterwarnings('ignore')

from . import data_blueprint
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
sia = SIA()

@data_blueprint.route('/getData/', methods=['GET'])
def getData(request = request):
    requested_collection = request.args.get("collection")

    news_collection = mongo.db[requested_collection] #Replace testData to actual DB
    output = []
    try:
        news_articles = loads(dumps(news_collection.find())) # make topic passable
        
        for article in news_articles:

            # structure schema here
            output.append({
                'title': article['title'],
                'summary': article['summary'],
                'source_name': article['source_name'],
                'author': article['author'],
                'article': article['article'],
                'sentiment_score': article['sentiment_score'],
                'sentiment_type': article['sentiment_type'],
            })
        if len(output) == 0:
            output = {'code': 2, "error": "User not found"}
        else:
            output = {"count": len(output), "results": output}
    except:
        output = {'code': 2, "error": "Error fetching details from DB"}

    return output

@data_blueprint.route('/scrapeURL/', methods=['GET'])
def scrapeURL():
    scrape_news.delay()
    return "Scraped News and Extracted Snippets"

@data_blueprint.route('/scrapeSnip/', methods=['GET'])
def scrapeSnip():
    scrape_snip.delay()
    return "Scraped Snip called"

@data_blueprint.route('/scrapeSnipLoop/', methods=['GET'])
def scrapeSnipLoop():
    scrape_snip_loop.delay()
    return "Scraped Snip Loop called"

@data_blueprint.route('/scrapeSnipLatest/', methods=['GET'])
def scrapeSnipLatest():
    scrape_snip_latest.delay()
    return "Scraped Snip Latest called"


@data_blueprint.route('/extractSnippet/', methods=['GET'])
def extractSnippet():
    
    extract_snippets.delay()
    return "Extract snippets called"

@data_blueprint.route('/postData/', methods=['POST'])
def postData():
    request_json = request.get_json()
    url_list = request_json['url_list']
    all_news = []

    for news_source in url_list:


        feed_url = fp.parse(news_source)

        source_name = news_source
        url_feed = news_source
        article_list = []

        for article in feed_url.entries:
            article_dict = {}

            #date = article.published_parsed
            artilce_url = article.link

            content = Article(artilce_url) #Newspapaer 3k's Article module to read the contents
            try: 
                content.download() #Downloading the News article
                content.parse()
                content.nlp()  
                

            except:
                pass

            # Updating all the information to a dictionary
            # Adding the sentiment score and sentiment type functionality
            sentiment_score = sia.polarity_scores(content.text)
            sentiment_type = max(sentiment_score, key=sentiment_score.get)
            article_dict.update({'source_name': source_name, 'source_url': url_feed, "article_url": artilce_url, 'image_url': content.top_image,'video_url': content.movies, 'publish_date':content.publish_date,'title':content.title, 'article': content.text, 'author':content.authors, "summary": content.summary, "keywords": content.keywords, "sentiment_score": sentiment_score, "sentiment_type": sentiment_type})
            article_list.append(article_dict)

        all_news.extend(article_list)

    fmt = '%Y-%m-%dT-%H-%M%Z%z'
    naive_dt = datetime.now()
    start_time = naive_dt.strftime(fmt)

    print("Download started:", start_time)

    news_db = mongo.db.testData
    news_db.insert_many(all_news)

    print("Complete")
    
    return "Completed"

@data_blueprint.route('/processTheme/', methods=['POST'])
def processTheme():
    inputThemes = request.data
    print(inputThemes)

    topic_ids = get_topics(inputThemes)
    json_results = get_articles_by_topic(topic_ids)

    print("Process theme called")

    return json_results

def get_topics(theme_words):
    print("Getting topics")
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

    ##Providing the search query
    search_param = {
    "query": {
        "multi_match": {
           "query": theme_words,
           "fields": "keywords"
           }
        }
    }
      
    response = es.search(index="article_production", body=search_param) # Response from Elasticsearch
    results= response["hits"]["hits"] 
    # Collecting the Topic ID's from the search results.

    topics=[]
    for i in results:
      topics.append(i["_id"])

    print("Completed getting topics")
    return topics

def get_articles_by_topic(topic_ids):
    snippet_collection = mongo.db["snippet_collection"] 

    topic_results = []
    for topic in topic_ids:
        topic_dict = {}
        print("Getting snippets with highest compound for topic", topic)
        snippet_data = list(snippet_collection.find({"topic": int(topic)}).sort("percentage", -1).limit(10))

        cleaned_snippet_data = []

        for snippet in snippet_data:
            temp_dict = {}
            temp_dict["snippetID"] = snippet["snip_id"]
            temp_dict["snippet_type"] = snippet["type"]
            temp_dict["snippet_url"] = snippet["snippet_url"]
            temp_dict["snippet_description"] = snippet["content"]
            temp_dict["title"] = snippet["parent_article"]
            temp_dict["article_url"] = snippet["parent_article_url"]
            temp_dict["percentage"] = snippet["percentage"]

            cleaned_snippet_data.append(temp_dict)

        topic_dict["topicID"] = topic
        topic_dict["title"] = snippet_data[0]["parent_article"]
        topic_dict["summary"] = snippet_data[0]["content"]
        topic_dict["primary_snippets"] = cleaned_snippet_data[:6] #get first 6 snippets from list
        topic_dict["secondary_snippets"] = cleaned_snippet_data[-4:] #get last 4 snippets from list

        topic_results.append(topic_dict)

    return topic_results

