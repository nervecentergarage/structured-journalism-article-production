# Get the topics realted to the Theme Keywords


import ssl
from elasticsearch import Elasticsearch 
from elasticsearch.connection import create_ssl_context

def get_topics(theme_words):
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
    return topics
    

    
get_topics("input your theme") # Imput the theme