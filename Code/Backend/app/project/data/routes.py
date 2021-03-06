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

from tasks import scrape_snip_latest

import ssl
from elasticsearch import Elasticsearch
from elasticsearch.connection import create_ssl_context

import warnings
warnings.filterwarnings('ignore')
from pymongo import MongoClient 
from . import data_blueprint
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
sia = SIA()

@data_blueprint.route('/hello/', methods=['GET'])
def helloWorld():
    return "Hello World"

@data_blueprint.route('/scrapeSnipLatest/', methods=['GET'])
def scrapeSnipLatest():
    scrape_snip_latest.delay()
    return "Scraped Snip Latest called"

@data_blueprint.route('/news/', methods=['GET'])
def get_by_sentimenttype(request=request):
    result = []
    sentiment_type = request.args.get('sentiment_type')
    for col in mongo.db.list_collection_names():
        collection = mongo.db[col]
        result.append(list(collection.find({"sentiment_type": sentiment_type})))
    return {"news":json.loads(json_util.dumps(result))}


@data_blueprint.route('/processTheme/', methods=['POST'])
def processTheme():
    inputThemes = request.data
    if isinstance(inputThemes, bytes):
        inputThemes = inputThemes.decode("utf-8")
    else:
        inputThemes = str(inputThemes)

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
    os.environ.get('ELASTIC_API'),
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

    topic_results = {}
    for topic in topic_ids:
        topic_dict = {}
        print("Getting snippets with highest compound for topic", topic)
        snippet_data_int = list(snippet_collection.find({"topic": int(topic)}).sort("percentage", -1).limit(10))
        snippet_data_string = list(snippet_collection.find({"topic": topic}).sort("percentage", -1).limit(10))
        
        snippet_data = []
        snippet_data.extend(snippet_data_int)
        snippet_data.extend(snippet_data_string)
        
        snippet_data.sort(key= lambda x:-x['percentage'])
        snippet_data = snippet_data[:10]
        
        cleaned_snippet_data = []

        if(snippet_data != []):
            for snippet in snippet_data:
                temp_dict = {}
                stringSnippetID = "snippet-"+ str(snippet["snip_id"])
                temp_dict["snippetID"] = stringSnippetID
                temp_dict["snippet_type"] = snippet["type"]
                temp_dict["snippet_url"] = snippet["snippet_url"]
                temp_dict["snippet_description"] = snippet["content"]
                temp_dict["title"] = snippet["parent_article"]
                
                temp_dict["sentiment_score"] = snippet["Sentiment_Score"] #Sentiment Score
                temp_dict["sentiment_type"] = snippet["Sentiment_type"] #Sentiment Type
                temp_dict["snippet_summary"] = snippet["snippet_summary"] #Snippet Summary
                temp_dict["relevance_score"] = snippet["percentage"] #Relevance Score (Importance)
                
                temp_dict["article_url"] = snippet["parent_article_url"]

                cleaned_snippet_data.append(temp_dict)

            topic_dict["topicID"] = topic
            topic_dict["title"] = snippet_data[0]["parent_article"]
            topic_dict["summary"] = snippet_data[0]["content"]
            topic_dict["primary_snippets"] = cleaned_snippet_data[0:6] #get first 6 snippets excluding the 1st snippet from list
            if (len(snippet_data) > 6):
                int_to_get = 6 - len(snippet_data)
                topic_dict["secondary_snippets"] = cleaned_snippet_data[int_to_get:] #get the remaining snippets from list
            topic_dict["tags"] = get_keywords(topic)

            topic_results[topic] = topic_dict

    return topic_results
    
def get_keywords(topic_id):
    client = MongoClient(os.environ.get('WORKER_MONGO_ARTICLES_DB'))
    topicDb = client.Topic_and_Keyword_DB
    topic_keyword_collection = topicDb.topic_keyword_collection

    document = list(topic_keyword_collection.find({ topic_id : { '$exists' : 1 } }))
    if document != []:
        keywords = document[0][topic_id]
    else:
        keywords = []

    return keywords

