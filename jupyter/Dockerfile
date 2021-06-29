FROM python:3.7-slim

RUN mkdir /src
WORKDIR /src

COPY requirements.txt .
RUN pip install -r requirements.txt

CMD jupyter notebook --no-browser --allow-root --ip=0.0.0.0 --port=8888