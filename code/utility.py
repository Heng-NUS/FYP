import numpy as np
import pandas as pd
import json


class mapper(object):
	"""map twitter text to ILI level"""
	def __init__(self, tweet_count, user_count):
		super(mapper, self).__init__()
		self.tweet_count = tweet_count
		self.user_count = user_count
		self.coefficient = 0.15 
		# default coefficient is 0.15	
		self.normalised = False

	def normalise(self):
		# normalise the regional data based on its number of users
		mean = np.mean(list(self.user_count.values()))
		if not self.normalised:
			try:
				for state in self.tweet_count.keys():
					if state in self.user_count:
						self.tweet_count[state] /= self.user_count[state] / 100
					else:
						self.tweet_count[state] /= mean / 100
					# T(s,n) = T(s,o) * 100/ user_number(s)
			except IOError:
				pass
			self.normalised = True

	def map_level(self):
		# map the number of tweets to ILI level
		self.normalise()
		mean = np.mean(list(self.tweet_count.values()))
		std = np.std(list(self.tweet_count.values()))
		level_num = 10	
		self.level = dict()
		# initialize the level array to the maximum level

		for key in self.tweet_count.keys():
			for i in range(1,level_num):
				if self.tweet_count[key] <= mean + (i-2) * std * self.coefficient:
					self.level[key] = i
					break

		return self.level

	def update_tweet_count(self, tweet_count, user_count=None):
		self.tweet_count = tweet_count
		if user_count: self.user_count = user_count
		self.normalised = False

class Diffusion(object):
	"""predict the diffusion of diseases"""
	def __init__(self, tweet_count, map_path='./Data/usa_adjacency.json'):
		super(Diffusion, self).__init__()
		try:
			with open("./Data/usa_adjacency.json", 'r', encoding='utf-8') as file:
				self.adjacency_mat = json.load(file)
		except IOError:
			print("Can't find file:", map_path)


tweet_count = dict()
user_count = dict()
with open("./Data/test_result.json", 'r', encoding='utf-8') as file:
	tweet_count = json.load(file)
with open("./Data/test_users.json", 'r', encoding='utf-8') as file:
	user_count = json.load(file)

a = mapper(tweet_count, user_count)
a.map_level()
print(a.level)
b = Diffusion(tweet_count)