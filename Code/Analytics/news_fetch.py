from pymongo import MongoClient 
import pandas as pd
import requests
import bs4
import feedparser as fp
import json
import newspaper
from newspaper import Article
import time
from time import mktime
from datetime import datetime
from pytz import timezone


def fetch_news(): 
    
    data = pd.read_csv("news_sources.csv")

    all_news = []

    for news_source in data.iterrows():
    
        feed_url = fp.parse(news_source[1]['rss_feed'])
        source_name = news_source[1]['source']
        source_url = news_source[1]['url']
        url_feed = news_source[1]['rss_feed']
        article_list = []
    
        for article in feed_url.entries:
            article_dict = {}
            try: 
                date = article.published_parsed
                artilce_url = article.link

                content = Article(artilce_url) #Newspapaer 3k's Article module to read the contents
                content.download() #Downloading the News article
                content.parse()    #Downloading the News article
                title = content.title #Getting the Title of the article
                author = content.authors #Getting the Author of the article
                publish_date = content.publish_date  #Getting the publish date of the article 
                full_article = content.text #Getting the complete content of the article 

            # Processing the document with NLP tasks to extract 'Summary' and 'Keywords' from the article.

                content.nlp()  
                summary = content.summary   
                keywords = content.keywords
                image_url = content.top_image
                video_url = content.movies 
                
            except:
                pass
            
            # Updating all the information to a dictionary
            article_dict.update({'source_name': source_name, 'source_url': url_feed, "article_url": artilce_url, 'video_url': video_url, 'image_url':image_url,'publish_date':publish_date,'title':title, 'article': full_article, 'author':author, "summary": summary, "keywords": keywords})
            article_list.append(article_dict)
            
        all_news.extend(article_list)
        
    return all_news
    
while True: 

    # define date format
    fmt = '%Y-%m-%dT-%H-%M%Z%z'
    # define eastern timezone
    eastern = timezone('US/Eastern')
    # naive datetime
    naive_dt = datetime.now()
    loc_dt = datetime.now(eastern)
    start_time = naive_dt.strftime(fmt)
    
    print("Download started:", start_time)
    
    all_news = fetch_news()  # Fetching the news

    # Storing the data to MongoDB

    try: 
        client = MongoClient("mongodb+srv://sahmed253:1qa1uXw88xx8nYPX@cluster0.gpcla.azure.mongodb.net/<dbname>?retryWrites=true&w=majority")
    #     print("Connected successfully!!!") 
    except:   
        print("Could not connect to MongoDB") 


    db = client.news  # DB name
    collection = db.news_collection # Collection name

    articles = collection.insert_many(all_news) # Inserting the articles to mongodb

    print("Complete:") 

    time.sleep(60*60)
        
        
    

    