FROM python:3.7-slim

RUN mkdir /src
WORKDIR /src

COPY main.py .
COPY model.joblib .
COPY requirements.txt .

RUN pip install -r requirements.txt

CMD uvicorn --reload --host=0.0.0.0 --port=8890 main:app