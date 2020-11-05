import os

task_serializer  = 'json'
broker_url = os.environ.get('REDIS_URL', 'redis://127.0.0.1:6379/0')
accept_content  = ['json']