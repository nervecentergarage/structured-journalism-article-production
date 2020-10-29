FROM python:3

ADD ./ worker/

WORKDIR /worker

RUN apt-get update

RUN pip install -r requirements.txt

EXPOSE 5000

CMD celery -A tasks worker -l info