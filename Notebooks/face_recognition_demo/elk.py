
import requests
from elasticsearch import Elasticsearch
from datetime import datetime

class Elk():
	def __init__(self,url= 'http://localhost:9200',host='localhost',port = 9200):
		res = requests.get(url)
		self.es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
		self.index = 1
	def push_data(self,json):
		 self.es.index(index='fulcrum', doc_type='people_emotions', id=self.index, body=json)
		 self.index+=1


