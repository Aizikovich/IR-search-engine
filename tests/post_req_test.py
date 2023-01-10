import requests

r = requests.post('http://127.0.0.1:8080//get_pagerank', json=[17331890, 17330517, 17330527])
res = r.json()
print(res)

r = requests.post('http://127.0.0.1:8080//get_pageview', json=[17331890, 17330517, 17330527])
res = r.json()
print(res)