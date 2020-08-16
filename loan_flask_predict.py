# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 12:43:22 2020

@author: padma
"""

import json
import requests
import pandas as pd

"""Setting the headers to send and accept json responses
"""
header = {'Content-Type': 'application/json', \
          'Accept': 'application/json'}

"""Reading test batch
"""
#df = pd.read_csv('../data/test.csv', encoding="utf-8-sig")
df = pd.read_csv('C:\\Users\\cvsai\\Desktop\\loan_pred_test.csv', encoding="utf-8-sig")

df = df.head()  # for only 5 records from the test batch

"""Converting Pandas Dataframe to json
"""
data = df.to_json(orient='records')

#resp = requests.get("http://localhost:5003/", headers=header)
#print(resp.status_code)
"""POST <url>/predict
"""
resp = requests.post("http://localhost:5003/predict", \
                    data = json.dumps(data),\
                    headers= header, verify=False)
print('response status code:{0}'.format(resp.status_code))

"""The final response we get is as follows:
"""
print(resp.json())

