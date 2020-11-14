import requests
import datetime
scrapeSnipLatest = "https://heroku-article-production.herokuapp.com/scrapeSnipLatest/"
res = requests.get(url= scrapeSnipLatest)

date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print("{} at {}".format(res.text, date_time))