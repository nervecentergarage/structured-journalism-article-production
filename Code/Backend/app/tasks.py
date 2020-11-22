import re
import os
import ssl
import bs4
import json
import time
import locale
import warnings
import pandas as pd
from json import loads
from time import sleep
from celery import Celery
from pprint import pprint
from datetime import datetime
from string import punctuation

import newspaper
from newspaper import Article
from bson import json_util
from bson.json_util import dumps, RELAXED_JSON_OPTIONS
from bson.objectid import ObjectId
import feedparser as fp


import gensim
from gensim import corpora, models
from gensim.models import CoherenceModel
from gensim.summarization import summarize
from gensim.utils import simple_preprocess
import numpy as np
np.random.seed(2018)

import nltk
from nltk import tokenize
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from collections import defaultdict, Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from elasticsearch import Elasticsearch
from elasticsearch.connection import create_ssl_context
from pymongo import MongoClient 

locale.getdefaultlocale()
warnings.filterwarnings('ignore')

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('vader_lexicon')
nltk.download('stopwords')

stemmer = SnowballStemmer('english')

app = Celery()
app.config_from_object("celery_settings")

s = SentimentIntensityAnalyzer() # New sentiment Analyzer variable
#sia = SentimentIntensityAnalyzer()


# NEW CODE =========================================================

# Initialize the stopwords
stoplist = stopwords.words('english')

# Defining the stopwords

stopwords_verbs = ['say', 'get', 'go', 'know', 'may', 'need', 'like', 'make', 'see', 'want', 'come', 'take', 'use', 'would', 'can']
stopwords_other = ['one', 'mr', 'bbc', 'image', 'getty', 'de', 'en', 'caption', 'also', 'copyright', 'something']
stops_rm = set(['above','against','ain','any','aren','because','below','didn','couldn','doesn','does','down','few','hadn','isn',"isn't",'mightn','more','most','mustn','no','nor','not','now','off','out','over','too','under','until','up','ve','very'])

my_stopwords = stoplist + stopwords_verbs + stopwords_other

stops = set(my_stopwords).difference(stops_rm)

remove_special_character= re.compile('[/(){}\[\]\|@,;]')
bad_symbols = re.compile('[^0-9a-z #+_]')
stops.update(list(punctuation)) # Updating with punctuations

# END OF NEW CODE =========================================================


def snippet_summarizer(text):
  summary = summarize(text, ratio=0.3)
  return summary

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

        try:
            feed_url = fp.parse(news_source)
            paper = newspaper.build(news_source)
            source_name = paper.brand
            source_url = news_source
            url_feed = news_source
        except:
            pass
        
        article_list = []

        for article in feed_url.entries:
            article_dict = {}
            
            try: 
                date = article.published_parsed
                artilce_url = article.link
                content = Article(artilce_url) #Newspapaer 3k's Article module to read the contents
                content.download() #Downloading the News article
                content.parse()    #Downloading the News article

                # Processing the document with NLP tasks to extract 'Summary' and 'Keywords' from the article.
                content.nlp()  
            except:
                pass
            

            if content.publish_date != None:
                publish_date = content.publish_date
            else:
                publish_date = ""
            sentiment_results = s.polarity_scores(content.text)
            sentiment_score = sentiment_results['compound']
            if -0.2<= sentiment_score <= 0.2:
                sentiment_type = 'neu'
            elif sentiment_score > 0.2:
                sentiment_type = 'pos'
            else:
                sentiment_type = 'neg'
            # Updating all the information to a dictionary
            article_dict.update({'article_id': article_id, 'source_name': source_name, 'source_url': news_source, "article_url": artilce_url, 'image_url': content.top_image,'video_url': content.movies, 'publish_date': publish_date,'title':content.title, 'article': content.text, 'author':content.authors, "default_summary": content.summary, "keywords": content.keywords, "category": category, "sentiment_score": sentiment_score, "sentiment_type": sentiment_type})
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
            if len(li[i].split()) <=50:  # NEW CODE
                new_list[-1] = new_list[-1].join([" ",li[i]]) # NEW CODE
            else:
                new_list.append(li[i])
        else:
            new_list.append(li[i])
    return new_list  # returns a list of paragraphs

def snip_json(article_data, snippet_collection):
    snippets = []
    try:
        snippet_id = int(snippet_collection.find().skip(snippet_collection.count_documents({}) - 1)[0]['snip_id']) + 1
    except:
        snippet_id = 1
    for i in article_data:
        snips_pre = snips(i["article"])
        for j in snips_pre:
            final_snip = {}
            final_snip["type"] = "text"
            final_snip["snip_id"] = snippet_id
            final_snip["content"] = j

            # NEW CODE =============================================
            try:
                final_snip["snippet_summary"]=snippet_summarizer(j)
            except ValueError:
                final_snip["snippet_summary"]=j
            final_snip["parent_article_summary"]=i["default_summary"] # "default_summary" new key not found in old implemenation
            # END OF NEW CODE ======================================


            final_snip["parent_article"] = i["title"]
            final_snip["parent_article_url"] = i["article_url"]

            ####ADD
            try:
                final_snip["publish_date"]=datetime.fromtimestamp(i["publish_date"]["$date"]//1000).strftime("%m/%d/%Y, %H:%M:%S")
            except:
                final_snip["publish_date"]="none"
            #######
            final_snip["source_url"] = i["source_url"]
            final_snip["publish_date"] = i["publish_date"]
            final_snip["author"] = i["author"]
            final_snip["category"] = i['category']
            final_snip["snippet_url"] = ""
            snippets.append(final_snip)
            snippet_id += 1

        if i["image_url"] != "":
            img_snip = {}
            img_snip["type"] = "image"
            img_snip["snippet_url"] = i["image_url"]
            img_snip["snip_id"] = snippet_id

            img_snip["content"] = i["default_summary"] #New code changed this to "defaul_summary"

            img_snip["parent_article"] = i["title"]
            img_snip["parent_article_url"] = i["article_url"]

            ####ADD
            try:
                img_snip["publish_date"]=datetime.fromtimestamp(i["publish_date"]["$date"]//1000).strftime("%m/%d/%Y, %H:%M:%S")
            except:
                img_snip["publish_date"]="none"
            ###

            img_snip["source_url"] = i["source_url"]
            img_snip["publish_date"] = i["publish_date"]
            img_snip["author"] = i["author"]
            img_snip["category"] = i['category']
            snippets.append(img_snip)
            snippet_id += 1

        elif i["video_url"] != "":                    
            vid_snip = {}
            vid_snip["type"] = "video"
            vid_snip["snippet_url"] = i["video_url"]
            vid_snip["snip_id"] = snippet_id

            vid_snip["content"] = i["default_summary"] #New code changed this to "default_summary"

            vid_snip["parent_article"] = i["title"]
            vid_snip["parent_article_url"] = i["article_url"]

            #ADDD
            try:
                vid_snip["publish_date"]=datetime.fromtimestamp(i["publish_date"]["$date"]//1000).strftime("%m/%d/%Y, %H:%M:%S")
            except:
                vid_snip["publish_date"]="none"
            ######

            vid_snip["source_url"] = i["source_url"]
            vid_snip["publish_date"] = i["publish_date"]
            vid_snip["author"] = i["author"]
            vid_snip["category"] = i['category']
            snippets.append(vid_snip)
            snippet_id += 1

    return snippets

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    text = text.lower() # lowering text
    text = remove_special_character.sub('', text) # replace any special characters by space in text    
    text = bad_symbols.sub('', text) # delete symbols which are in bad symbols from text
    
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in stops and len(token) >= 2:
            result.append(lemmatize_stemming(token))
    return result

# NEW CODE =================================================
# Defining a Search Function to check the similarity
def search(tfidf_matrix, model, request):
    request_transform = model.transform([request])
    similarity = np.dot(request_transform, np.transpose(tfidf_matrix))
    x = np.array(similarity.toarray()[0])
    indices =  np.argmax(x)
    if x[indices]>0.2: 
      return (indices,x[indices])
    else:
      return (-1,0)
# END OF NEW CODE ===========================================

def attach_topics(snippets):
    ## mongo new collection topics
    data_layer = {
    "connection_string": os.environ.get('WORKER_MONGO_ARTICLES_DB'),
    "collection_name": "topic_keyword_collection",
    "database_name": "Topic_and_Keyword_DB"
    }
    db_connect = MongoClient(data_layer["connection_string"])
    database=db_connect[data_layer['database_name']]
    collection=database[data_layer['collection_name']]

    #Comparing with previous topics
    append_prev=[]
    if collection.count() != 0: # Check for empty collection

        prev_topics=[]
        for i in collection.find():
            keys=list(i.keys())
            values=list(i.values())
            prev_topics.append([keys[1]," ".join(values[1])])

        d=pd.DataFrame(prev_topics)

        text_content = d[1]
        vector = TfidfVectorizer(max_df=0.1,         # drop words that occur in more than X percent of documents
                                    #min_df=8,      # only use words that appear at least X times
                                    stop_words=None, # remove stop words
                                    lowercase=True, # Convert everything to lower case 
                                    use_idf=True,   # Use idf
                                    norm=u'l2',     # Normalization
                                    smooth_idf=True # Prevents divide-by-zero errors
                                    )
        tfidf = vector.fit_transform(text_content)


        append_prev=[]
        append_snip_ids=[]
        filtered_snippets=[]

        for req in snippets:
            request = req["content"]
            result = search(tfidf, vector, request)
            if result != (-1, 0):
                req["topic"]=d.iloc[ result[0] , 0 ]
                req["percentage"]=result[1]
                req["Sentiment_Score"]=s.polarity_scores(request)["compound"]
                req["Sentiment_type"]=("positive" if req["Sentiment_Score"] > 0.2 else ("negative" if req["Sentiment_Score"]<0.2 else "neutral"))
                append_prev.append(req)
                append_snip_ids.append(req["snip_id"])
            else:
                filtered_snippets.append(req)
        

        if(filtered_snippets != []):
            df=pd.DataFrame(filtered_snippets)
        else:
            return []


    else:
        df=pd.DataFrame(snippets)

    df['processed_text_corpus'] = df['content'].map(preprocess)
    df['processed_text'] = df['processed_text_corpus'].apply(lambda x: ' '.join(x))
    df["sentimentscores"] = df["content"].apply(lambda x : s.polarity_scores(x))
    df = pd.concat([df.drop(['sentimentscores'], axis = 1), df['sentimentscores'].apply(pd.Series)], axis = 1)
    df = df.loc[:,~df.columns.duplicated()]
    processed_docs=df.processed_text_corpus.tolist()
    dicti=gensim.corpora.Dictionary(processed_docs)
    bow_corpus = [dicti.doc2bow(doc) for doc in processed_docs]
    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]

    ##elastic
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


    # lda_model_tfidf =  gensim.models.LdaMulticore(corpus_tfidf, num_topics =25 , id2word = dicti, passes = 2, workers = 2)
    models_list = []
    coherence_list = []
    number_topics = []
    for num_topics in range(5, 30,5):
        number_topics.append(num_topics)
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus_tfidf,
                                              id2word=dicti,
                                              num_topics=num_topics, 
                                              random_state=100,
                                              eval_every=10,
                                              chunksize=2000,
                                              passes=5,
                                              per_word_topics=True
                                              )
        
        coherence = CoherenceModel(model=lda_model, 
                                   texts=processed_docs,
                                   dictionary=dicti, coherence='c_v',
                                   processes=1)
        coherence_list.append(coherence.get_coherence())
        models_list.append(lda_model)


    best_coherence=coherence_list.index(max(coherence_list))

    lda_model_tfidf=models_list[best_coherence]
    topics=lda_model_tfidf.show_topics(num_topics=number_topics[best_coherence],formatted=False, num_words= 50)
    topic_collection=[]


    client = MongoClient(os.environ.get('WEB_MONGO_SNIPPET_DB'))
    db = client.Topic_and_Keyword_DB
    topic_keyword_collection = db.topic_keyword_collection


    
    latest_topic_id = 1
    try:
        topic_cursor = topic_keyword_collection.find().sort("_id", -1).limit(1)
        topic_keys = topic_cursor[0].keys()

        for key in topic_keys:
            if key != "_id":
                latest_topic_id = int(key) + 1
    except:
        latest_topic_id = 1

    for topic in topics:
        topic_dict={}
        topic_dict[str(latest_topic_id)] = [i[0] for i in topic[1]] #Change inside the str "topic[0]" to increment according to highest topic_id found in DB
        topic_collection.append(topic_dict)
        new_data={"topic_id": latest_topic_id,"keywords": ' '.join(topic_dict[str(latest_topic_id)])} #Change "topic_id" to increment according to highest topic_id found in DB
        response = es.index(index = 'article_production',id = latest_topic_id,body = new_data) #Change id to increment according to highest topic_id found in DB
        latest_topic_id += 1

#######

    collection.insert_many(topic_collection)



    tmp_list=[]
    tmp_list1=[]
    for k in bow_corpus:
        results=lda_model_tfidf[k][0]
        tmp_list.append(sorted(results, key=lambda tup: -1*tup[1])[0][0])
        tmp_list1.append(sorted(results, key=lambda tup: -1*tup[1])[0][1])
    se = pd.Series(tmp_list)
    se1 = pd.Series(tmp_list1)
    df['topic'] = se.values
    df["percentage"] = se1.values
    final_df = df[["type",	"snip_id", "content",	"parent_article",	"parent_article_url",	"publish_date",	"source_url",	"author", "category", "snippet_url", "processed_text", "compound", "topic", "percentage"]]
    final_df = final_df.rename(columns={"compound": "Sentiment_Score"})
    final_df["Sentiment_type"] = final_df["Sentiment_Score"].apply(lambda x : "positive" if x > 0.2 else ("negative" if x<0.2 else "neutral"))

    topic_json=json.loads(final_df.to_json(orient="records"))

    topic_json.extend(append_prev)
    return topic_json

def get_topic_json(data, snippet_collection):  # reads raw data json
    snippets = snip_json(data, snippet_collection)  # Storing the snippets json

    stops = set(stopwords.words("english"))  # Defining the Stop words

    final_data = attach_topics(snippets)

    return final_data

@app.task
def scrape_news():
    print("Scraping news...")

    client = MongoClient(os.environ.get('WEB_MONGO_SNIPPET_DB'))
    db = client.news  # DB name

    # Dictionary of news categories and their information
    # category: {collection name, news list}
    news_dictionary = { "sports": {"collection": db.sports_collection, 
                                    "news_list": ["https://sports.yahoo.com/rss/","https://www.huffingtonpost.com/section/sports/feed",
                                               "https://rss.nytimes.com/services/xml/rss/nyt/Sports.xml", "http://feeds.bbci.co.uk/sport/rss.xml",
                                               "http://rss.cnn.com/rss/edition_sport.rss","https://www.theguardian.com/uk/sport/rss",
                                               "http://rssfeeds.usatoday.com/UsatodaycomSports-TopStories"]},
                        "politics": {"collection": db.politics_collection, 
                                    "news_list": ["https://www.huffingtonpost.com/section/politics/feed", "http://feeds.foxnews.com/foxnews/politics"]}, 
                        "health": {"collection": db.health_collection, 
                                    "news_list": ["https://rss.nytimes.com/services/xml/rss/nyt/Health.xml", "http://feeds.foxnews.com/foxnews/health"]},
                        "finance": {"collection": db.finance_collection, 
                                    "news_list": ["https://finance.yahoo.com/news/rssindex","https://www.huffingtonpost.com/section/business/feed",
                                                "http://feeds.nytimes.com/nyt/rss/Business", "http://feeds.bbci.co.uk/news/business/rss.xml",
                                                "https://www.theguardian.com/uk/business/rss", "http://rssfeeds.usatoday.com/UsatodaycomMoney-TopStories",
                                                "https://www.wsj.com/xml/rss/3_7031.xml", "https://www.wsj.com/xml/rss/3_7014.xml"]}, 
                        "environment": {"collection": db.environment_collection, 
                                        "news_list": ["https://www.huffingtonpost.com/section/green/feed", "http://feeds.foxnews.com/foxnews/scitech",
                                                    "http://newsrss.bbc.co.uk/rss/newsonline_uk_edition/sci/tech/rss.xml",
                                                    "https://www.theguardian.com/uk/environment/rss"]}, 
                        "scitech": {"collection": db.scitech_collection, 
                                    "news_list": ["http://feeds.nytimes.com/nyt/rss/Technology", "http://www.nytimes.com/services/xml/rss/nyt/Science.xml",
                                                "http://feeds.foxnews.com/foxnews/tech", "http://feeds.bbci.co.uk/news/technology/rss.xml",
                                                "https://www.theguardian.com/uk/technology/rss", "https://www.theguardian.com/science/rss",
                                                "https://www.wsj.com/xml/rss/3_7455.xml"]}, 
                        "general": {"collection": db.general_collection, 
                                    "news_list": ["https://www.yahoo.com/news/rss", "https://www.huffpost.com/section/front-page/feed?x=1", 
                                                "http://rss.cnn.com/rss/cnn_topstories.rss", "http://www.nytimes.com/services/xml/rss/nyt/HomePage.xml", 
                                                "http://feeds.foxnews.com/foxnews/latest", "http://www.nbcnews.com/id/3032091/device/rss/rss.xml", 
                                                "http://www.dailymail.co.uk/home/index.rss", "http://www.washingtontimes.com/rss/headlines/news/", "https://www.theguardian.com/uk/rss", 
                                                "https://www.wsj.com/xml/rss/3_7031.xml", "http://feeds.abcnews.com/abcnews/topstories", "http://newsrss.bbc.co.uk/rss/newsonline_uk_edition/front_page/rss.xml", 
                                                "http://rssfeeds.usatoday.com/UsatodaycomNation-TopStories", "https://www.latimes.com/local/rss2.0.xml"]} }

    categories = news_dictionary.keys()
    
    for c in categories:
        news_information = news_dictionary[c]

        print("Collecting", c, "news")
        fetch_news(news_information["news_list"], c, news_information["collection"]) # Fetch news by passing news list, category, and collection name

    print("Scraping complete")

@app.task
def extract_snippets():
    print("Starting Extract Snippets Task")
    client = MongoClient(os.environ.get('WORKER_MONGO_ARTICLES_DB'))
    db = client.Snippet_DB  

    snippet_collection = db.snippet_collection 

    news_collections = []
    news_collections.extend([db.sports_collection, db.politics_collection, db.health_collection, 
                            db.finance_collection, db.environment_collection, db.scitech_collection, db.general_collection])

    n = 1
    for collection in news_collections:
        print("Extracting", str(n) + "/" +  str(len(news_collections)) + " snippets...")
        data = list(collection.find())
        snippets = get_topic_json(data, snippet_collection)
        snippet_collection.insert_many(snippets)
        n += 1

@app.task
def scrape_snip():
    scrape_news()
    extract_snippets()

@app.task
def scrape_snip_latest():
    scrape_snip_latest_news()
    
def scrape_snip_latest_news():
    print("Scraping and Snipping latest news...")

    client = MongoClient(os.environ.get('WORKER_MONGO_ARTICLES_DB'))
    db = client.News_Article_DB  # DB name
    db_Snippet = client.Snippet_DB
    snippet_collection = db_Snippet.snippet_collection 

    # Dictionary of news categories and their information
    # category: {collection name, news list}
    news_dictionary = { "sports": {"collection": db.sports_collection, 
                                    "news_list": ["https://sports.yahoo.com/rss/","https://www.huffingtonpost.com/section/sports/feed",
                                               "https://rss.nytimes.com/services/xml/rss/nyt/Sports.xml", "http://feeds.bbci.co.uk/sport/rss.xml"
                                               "http://rss.cnn.com/rss/edition_sport.rss","https://www.theguardian.com/uk/sport/rss",
                                               "http://rssfeeds.usatoday.com/UsatodaycomSports-TopStories"]},
                        "politics": {"collection": db.politics_collection, 
                                    "news_list": ["https://www.huffingtonpost.com/section/politics/feed", "http://feeds.foxnews.com/foxnews/politics"]}, 
                        "health": {"collection": db.health_collection, 
                                    "news_list": ["https://rss.nytimes.com/services/xml/rss/nyt/Health.xml", "http://feeds.foxnews.com/foxnews/health"]},
                        "finance": {"collection": db.finance_collection, 
                                    "news_list": ["https://finance.yahoo.com/news/rssindex","https://www.huffingtonpost.com/section/business/feed",
                                                "http://feeds.nytimes.com/nyt/rss/Business", "http://feeds.bbci.co.uk/news/business/rss.xml",
                                                "https://www.theguardian.com/uk/business/rss", "http://rssfeeds.usatoday.com/UsatodaycomMoney-TopStories",
                                                "https://www.wsj.com/xml/rss/3_7031.xml", "https://www.wsj.com/xml/rss/3_7014.xml"]}, 
                        "environment": {"collection": db.environment_collection, 
                                        "news_list": ["https://www.huffingtonpost.com/section/green/feed", "http://feeds.foxnews.com/foxnews/scitech",
                                                    "http://newsrss.bbc.co.uk/rss/newsonline_uk_edition/sci/tech/rss.xml",
                                                    "https://www.theguardian.com/uk/environment/rss"]}, 
                        "scitech": {"collection": db.scitech_collection, 
                                    "news_list": ["http://feeds.nytimes.com/nyt/rss/Technology", "http://www.nytimes.com/services/xml/rss/nyt/Science.xml",
                                                "http://feeds.foxnews.com/foxnews/tech", "http://feeds.bbci.co.uk/news/technology/rss.xml",
                                                "https://www.theguardian.com/uk/technology/rss", "https://www.theguardian.com/science/rss",
                                                "https://www.wsj.com/xml/rss/3_7455.xml"]}, 
                        "general": {"collection": db.general_collection, 
                                    "news_list": ["https://www.yahoo.com/news/rss", "https://www.huffpost.com/section/front-page/feed?x=1", 
                                                "http://rss.cnn.com/rss/cnn_topstories.rss", "http://www.nytimes.com/services/xml/rss/nyt/HomePage.xml", 
                                                "http://feeds.foxnews.com/foxnews/latest", "http://www.nbcnews.com/id/3032091/device/rss/rss.xml", 
                                                "http://www.dailymail.co.uk/home/index.rss", "http://www.washingtontimes.com/rss/headlines/news/", "https://www.theguardian.com/uk/rss", 
                                                "https://www.wsj.com/xml/rss/3_7031.xml", "http://feeds.abcnews.com/abcnews/topstories", "http://newsrss.bbc.co.uk/rss/newsonline_uk_edition/front_page/rss.xml", 
                                                "http://rssfeeds.usatoday.com/UsatodaycomNation-TopStories", "https://www.latimes.com/local/rss2.0.xml"]} }

    start_article = 0
    end_article = 0
    categories = news_dictionary.keys()
    for c in categories:
        news_information = news_dictionary[c]

        print("Collecting", c, "news...")
        start_article, end_article = fetch_news(news_information["news_list"], c, news_information["collection"]) # Fetch news by passing news list, category, and collection name
        collection = news_information["collection"]

        print("Extracting", c, "news snippets...")
        data = list(collection.find({"article_id" : {"$gte":start_article, "$lt":end_article+1}}))
        snippets = get_topic_json(data, snippet_collection) # Snipping
        if(snippets != []):
            snippet_collection.insert_many(snippets)

    print("Scraping and Snipping of latest Articles complete")


    