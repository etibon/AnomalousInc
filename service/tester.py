import pandas as pd
import requests
import json
import random


def runTest():
    df = pd.read_csv('test.csv')
    for index, row in df.iterrows():
        req_map = {'feature_vector': [row['mean'], row['sd']]}
        if random.random() < 0.25:
            req_map['score']=True

        print(json.dumps(req_map))
        r = requests.post('http://localhost:8000/predict', headers={'accept': 'application/json'}, json=req_map)
        print(r.text)
    r = requests.get('http://localhost:8000/model_information')
    print(r.text)


if __name__ == '__main__':
    runTest()