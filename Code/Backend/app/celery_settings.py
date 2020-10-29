import os

CELERY_TASK_SERIALIZER = 'json'
BROKER_URL = os.environ.get('redis://localhost:6379/0')
CELERY_ACCEPT_CONTENT = ['json']