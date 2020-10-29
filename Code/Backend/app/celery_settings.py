import os

task_serializer  = 'json'
broker_url = os.environ.get('redis://localhost:6379/0')
accept_content  = ['json']