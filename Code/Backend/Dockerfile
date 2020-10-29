FROM python:3 as web

ADD ./app /app

WORKDIR /app

RUN apt-get update

RUN pip install -r requirements.txt

EXPOSE 5000

CMD python app.py


FROM python:3 as worker

ADD ./app /app

WORKDIR /app

RUN apt-get update

RUN pip install -r requirements.txt

EXPOSE 5000

CMD [celery, -A, tasks.py, worker]