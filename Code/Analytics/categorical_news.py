import pymongo
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
                summary = content.summary   
                keywords = content.keywords
                image_url = content.top_image 
                video_url = content.movies 
    
            except:
                pass
            
            # Updating all the information to a dictionary
            
            article_dict.update({'source_name': source_name, 'source_url': url_feed, "article_url": artilce_url, 'image_url': image_url,'video_url': video_url, 'publish_date':publish_date,'title':title, 'article': full_article, 'author':author, "summary": summary, "keywords": keywords})
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
    
    # Connecting to MongoDB

    try: 
        client = MongoClient("mongodb+srv://sahmed253:1qa1uXw88xx8nYPX@cluster0.gpcla.azure.mongodb.net/<dbname>?retryWrites=true&w=majority")
    #     print("Connected successfully!!!") 
    except:   
        print("Could not connect to MongoDB") 

    client = MongoClient("mongodb+srv://sahmed253:1qa1uXw88xx8nYPX@cluster0.gpcla.azure.mongodb.net/<dbname>?retryWrites=true&w=majority")
    

    sports_news = fetch_news(sports_list) # Fetching the news
    
    sports_db = client.news  # DB name
    sports_collection = sports_db.sports_news # Collection name
    sports_collection.insert_many(sports_news) # Inserting the articles to mongodb

    politics_news = fetch_news(politics_list)  # Fetching the news
    politics_db = client.news  # DB name
    politics_collection = sports_db.politics_news # Collection name
    politics_collection.insert_many(politics_news) # Inserting the articles to mongodb
    
    health_news = fetch_news(health_list)  # Fetching the news
    health_db = client.news  # DB name
    health_collection = sports_db.health_news # Collection name
    health_collection.insert_many(health_news) # Inserting the articles to mongodb
    
    finance_news = fetch_news(finance_list)  # Fetching the news
    finance_db = client.news  # DB name
    finance_collection = sports_db.finance_news # Collection name
    finance_collection.insert_many(finance_news) # Inserting the articles to mongodb
        
    environment_news = fetch_news(environment_list)  # Fetching the news
    environment_db = client.news  # DB name
    environment_collection = sports_db.environment_news # Collection name
    environment_collection.insert_many(environment_news) # Inserting the articles to mongodb
    
    scitech_news = fetch_news(scitech_list) # Fetching the news
    scitech_db = client.news  # DB name
    scitech_collection = sports_db.scitech_news # Collection name
    scitech_collection.insert_many(scitech_news) # Inserting the articles to mongodb
    
          
#     # Printing the data inserted 
#     cursor = collection.find() 
#     for record in cursor: 
    print("Complete")
      
    time.sleep(60*60)

    

