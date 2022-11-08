# -*- coding: utf-8 -*-
import time
import requests
import json


json_data = {'type': "12342345", 'content1': 'test something'}

aaa = time.time()
for i in range(0, 1):
    r = requests.post("http://localhost:5000/post_server", json=json_data)
    print(r)
    print(r.text)
    print("\n\n")
bbb = time.time()

print((bbb - aaa) / 1)
