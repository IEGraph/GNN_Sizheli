import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import scipy.sparse as sp
from copy import deepcopy
import random

class UserItemRatingDataset(Dataset):
	"""
	<users, items, ratings>
	"""
	def __init__(self, user_tensor, item_tensor, rating_tensor):
		self.user_tensor = user_tensor
		self.item_tensor = item_tensor
		self.rating_tensor = rating_tensor

	def __getitem__(self, item):
		return self.user_tensor[item], self.item_tensor[item], self.rating_tensor[item]

	def __len__(self):
		return self.user_tensor.size(0)

class SampleGenerator():
	"""
	Construct dataset for GCMF, check there are corresponding column names in init ratings
	"""
	def __init__(self, ratings):
		assert 'user_id' in ratings.columns
		assert 'item_id' in ratings.columns
		assert 'ratings' in ratings.columns

		# make sets to collect
		self.user_pool = set(pd.unique(ratings['user_id']))
		self.item_pool = set(pd.unique(ratings['item_id']))

		# explicit feedback using _normalize to limit these values between 0 and 1
		# implicit feedback using _binarize to limit these values 0 or 1
		self.processed_ratings = self._normalize(ratings)
		self.train, self.test = self._split_loo(self.processed_ratings)
		self.positive = self._sample_positive(ratings)


	def _normalize(self, ratings):
		ratings = deepcopy(ratings)
		max_rating = ratings.ratings.max()
		ratings.ratings = ratings.ratings * 1.0/max_rating
		return ratings

	def _binarize(self, ratings):
		ratings = deepcopy(ratings)
		ratings.ratings[ratings.ratings > 0] = 1.0
		return ratings

	def _split_loo(self, ratings):
		ratings = deepcopy(ratings)
		ratings['rank_latest'] = ratings.groupby('user_id')['ts'].rank(method = 'first', ascending = False)
		test = ratings[ratings['rank_latest'] == 1]
		train = ratings[ratings['rank_latest'] > 1]
		return train[['user_id', 'item_id', 'ratings']], test[['user_id', 'item_id', 'ratings']]

	def _sample_positive(self, ratings):
		interact_status = ratings.groupby('user_id')['item_id'].apply(set).reset_index().rename(
			columns = {'item_id':'interacted_items'}
		)
		return interact_status[['user_id', 'interacted_items']]

	def instance_train_loader(self, batch_size = 8):
		users, items, ratings = [], [], []
		train_ratings = self.train

		for row in train_ratings.itertuples():
			users.append(int(row.user_id))
			items.append(int(row.item_id))
			ratings.append(float(row.ratings))
		dataset = UserItemRatingDataset(user_tensor = torch.LongTensor(users), item_tensor = torch.LongTensor(items),
										rating_tensor = torch.FloatTensor(ratings))
		return DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True)







