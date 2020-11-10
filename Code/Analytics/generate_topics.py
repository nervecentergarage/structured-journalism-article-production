# Creating topics from the Raw Data.

%matplotlib inline

import warnings
warnings.filterwarnings('ignore')
import re
import json
import pandas as pd
import matplotlib
import gensim
# import locale
# locale.setlocale(locale.LC_ALL, 'en_US')
import numpy as np
np.random.seed(2018)
import nltk
nltk.download('wordnet')
nltk.download('vader_lexicon')
nltk.download('stopwords') # Download these stop words


from nltk.stem import WordNetLemmatizer, SnowballStemmer
stemmer = SnowballStemmer('english')
from pprint import pprint
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
from gensim.utils import simple_preprocess
from nltk.stem.porter import *
from gensim import corpora, models
from nltk import tokenize
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize   
from collections import defaultdict, Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
s = SentimentIntensityAnalyzer()
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 

from pymongo import MongoClient

import locale
locale.getdefaultlocale()

import ssl
from elasticsearch import Elasticsearch
from elasticsearch.connection import create_ssl_context

# Reading the Data file. This data needs to be collected from MongoDB - Raw database.

with open("/content/sample_data/news_data.json") as f:
    data = json.load(f)

# Function to split the articles into snippets 
def snips(article):
  new_list=[]
  li=article.split("\n\n")
  for i in range(len(li)):
    if i >=1:
      if len(li[i].split()) <=20:
        new_list[-1].join([" ",li[i]])
      else:
        new_list.append(li[i])
    else:
      new_list.append(li[i])
  return new_list

# Splitting the entire articles into Text, Image and Video snippets

def snip_json(article_data):
  snippets=[]
  k=1 #pull from mongo
  for i in article_data:
    snips_pre=snips(i["article"])
    for j in snips_pre:
      final_snip={}
      final_snip["type"]="text"
      final_snip["snip_id"]=k
      # final_snip["article_id"]=i["article_id"]
      final_snip["content"]=j
      final_snip["parent_article"]=i["title"]
      final_snip["parent_article_url"]=i["article_url"]
      final_snip["publish_date"]=i["publish_date"]
      final_snip["source_url"]=i["source_url"]
      final_snip["author"]=i["author"]
      final_snip["category"]=i['category']
      final_snip["snippet_url"] = ""
      snippets.append(final_snip)
      k+=1
    if i["image_url"] != "":
        
      img_snip={}
      img_snip["type"]="image"
      img_snip["snippet_url"]=i["image_url"]
      img_snip["snip_id"]=k
      img_snip["content"]=i["default_summary"]
      img_snip["parent_article"]=i["title"]
      img_snip["parent_article_url"]=i["article_url"]
      img_snip["publish_date"]=i["publish_date"]
      img_snip["source_url"]=i["source_url"]
      img_snip["publish_date"]=i["publish_date"]
      img_snip["author"]=i["author"]
      img_snip["category"]=i['category']
      snippets.append(img_snip)
    elif i["video_url"] !="":
      k+=1
      vid_snip={}
      vid_snip["type"]="video"
      vid_snip["snippet_url"]=i["video_url"]
      vid_snip["snip_id"]=k
      vid_snip["content"]=i["default_summary"]
      vid_snip["parent_article"]=i["title"]
      vid_snip["parent_article_url"]=i["article_url"]
      vid_snip["publish_date"]=i["publish_date"]
      vid_snip["source_url"]=i["source_url"]
      vid_snip["publish_date"]=i["publish_date"]
      vid_snip["author"]=i["author"]
      vid_snip["category"]=i['category']
      snippets.append(vid_snip)
  return snippets

snippets=snip_json(data) # Storing the data in the variable

# Text Cleaning, stemming and lemmatizing

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos = 'v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in stops and len(token) >= 2:
            result.append(lemmatize_stemming(token))
    return result



stops = set(stopwords.words("english")) # Defining the stopwords

def attach_topics(snippets):
  stops = set(stopwords.words("english"))
  stops_rm = set(['above','against','ain','any','aren','because','below','didn','couldn','doesn','does','down','few','hadn','isn',"isn't",'mightn','more','most','mustn','no','nor','not','now','off','out','over','too','under','until','up','ve','very'])
  stops = stops.difference(stops_rm)
  df=pd.DataFrame(snippets)
  df['processed_text_corpus'] = df['content'].map(preprocess)
  df['processed_text'] = df['processed_text_corpus'].apply(lambda x: ' '.join(x))
  df["sentimentscores"] = df["content"].apply(lambda x : s.polarity_scores(x))
  df = pd.concat([df.drop(['sentimentscores'], axis = 1), df['sentimentscores'].apply(pd.Series)], axis = 1)
  df = df.loc[:,~df.columns.duplicated()]
  processed_docs=df.processed_text_corpus.tolist()
  dicti=gensim.corpora.Dictionary(processed_docs)
  bow_corpus = [dicti.doc2bow(doc) for doc in processed_docs] # Collecting the Bag of Words for the dictionary
  tfidf = models.TfidfModel(bow_corpus)  # Applying the Tf-IDF Model for the collection of words
  corpus_tfidf = tfidf[bow_corpus]

  ##elastic search connection

  context = create_ssl_context()
  context.check_hostname = False
  context.verify_mode = ssl.CERT_NONE
  es = Elasticsearch(
  "https://6c7e6efaa2574715a49ff2ea9757622d.eastus2.azure.elastic-cloud.com",
  http_auth=('elastic', 'WfAE0Yfwsny5e05thp4DmGDd'),
  # scheme="https",
  port=9243,
  ssl_context = context,
  )


  lda_model_tfidf =  gensim.models.LdaMulticore(corpus_tfidf, num_topics =25 , id2word = dicti, passes = 2, workers = 2)  # Applying the LDA Model for Topic Formations and clustering
  topics=lda_model_tfidf.show_topics(num_topics=25,formatted=False, num_words= 50)
  topic_collection=[]
  for topic in topics:
      topic_dict={}
      topic_dict[str(topic[0])] = [i[0] for i in topic[1]]
      topic_collection.append(topic_dict)
      new_data={"topic_id": topic[0],"keywords": ' '.join(topic_dict[str(topic[0])])} 
      response = es.index(index = 'article_production',id = topic[0],body = new_data) #Storing the topic ID's and the keywords in ElasticSearch

  #######
  ## mongo new collection topics
  data_layer = {
    "connection_string": "mongodb://likhil:likhil@cluster0-shard-00-00-yuxtg.mongodb.net:27017,cluster0-shard-00-01-yuxtg.mongodb.net:27017,cluster0-shard-00-02-yuxtg.mongodb.net:27017/test?ssl=true&replicaSet=Cluster0-shard-0&authSource=admin&retryWrites=true&w=majority",
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


