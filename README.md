# Overview
The API endpoint is deployed here: https://heroku-article-production.herokuapp.com/

How can AI make it easier to assemble structured journalism articles? Machines are no substitute for the insight and creativity of a trained journalist. But perhaps AI can assist with simple tasks that remove tedium and make production easier. This team will attempt to build an AI capable of assisting a journalist in quickly producing structured journalism content.

# Key Team Skills
Collectively, the team must have the following skill:
1. MongoDB
2. Flask or Django
3. Heroku
4. GitHub CI/CD
5. Behavior trees
6. Postman

# Roles and Responsibilities
1. **(1) Content developer**-- Create Python functions that query MongoDB for the raw content needed for article production.
2. **(1) Bot developer**-- Work with structured-journalism experts to determine simple assistant tasks. Use Flask or Django to build behavior trees that can perform simple structured journalism assistant tasks. Expose the funcationality as APIs.
3. **(1) Platform and API developer**-- Automate deployment from GitHub to Heroku. Use Postman to build API test cases for the bot and to demonstrate working functionality.


# How to setup the Heroku API endpoint

## Setting up Heroku and Github Secrets
 1. **Create a Heroku account** - you will need a heroku account to setup the API endpoint through a hosted heroku app. Create your account through the heroku site: https://www.heroku.com/
 2. **Create a Heroku app** - once created take note of the following variables so that you can replace the github secrets found in the *yml* file found in the repository.
	 - **Heroku Email** - the email address used for the Heroku account used to create the app.
	 - **Heroku App name** - the name of the Heroku app you just created.
	- **Heroku App key** - the API key for the Heroku app you just created.
3. **Setup Github Secrets** - with the following variables noted above you will have to set them within your own Github repository secrets settings. The created secrets should be assigned the following names.
	 - 	**HEROKU_EMAIL** - to be set to the noted Heroku email.
	 - **HEROKU_APP_NAME** - to be set to the noted Heroku app name
	 - **HEROKU_API_KEY** - to be set to the noted Heroku API key
## Setting up Heroku Addons
4. **Get the Heroku Redis Add-on** - this will be required for your Heroku web components to enqueue tasks for your Heroku worker component to process them.
## Setting up MongoDB
5. **Create a MongoDB account** - this account will be used to create the Databases. Create and login here: https://www.mongodb.com/
6. **Create the Following Databases** 
	-  **Snippet_DB** - this database will be used in the mongo connection string for the web component in your Heroku App.
	-	**News_Article_DB** - this database will be used in the mongo connection string for the worker in your Heroku App.
7. **Create a User with Database Access** - this user will be used in the mongo connection string.

## Setting up the Heroku Config Variables
8. **Set the Redis URI** - take the Heroku Redis add-on URI link and set it in the config variables within your Heroku application.
	- **REDIS_URL** - set the Redis URI to this name within the config vars.
9.   **Get the two Mongo Connection Strings** - take note of the 2 connection strings and set them to the following variables in the setting of your Heroku application within the config vars option (Show Config Vars).
		- **WEB_MONGO_SNIPPET_DB** - set the name of this config var to the Mongo connection string that references the **Snipper_DB**.
		- **WORKER_MONGO_ARTICLES_DB** - set the name of this confic var to the Mongo conneciton string that references the **News_Article_DB**.
* *It is a good practice to hide database connection strings, and if you look into the code you will notice the two config variables being set.
10. **Setup Elasticsearch** - create an account and take note of the password. Then set another pair of config variables in your Heroku application.
	- **ELASTIC_API** - which is used the API key used to connect to your Elasticsearch function. 
	- **ELASTIC_USER** - which is the username used to access your Elasticsearch endpoint.
	- **PASS_ELASTIC** - which is the password that you set to access your Elasticsearch endpoint.
