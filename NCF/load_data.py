import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import random
from copy import deepcopy
import scipy.sparse as sp
random.seed(0)

class UserItemRatingDataset(Dataset):
	"""
	Wrapper, convert <user, item, rating> Tensor into pytorch tensor
	"""
	def __init__(self, user_tensor, item_tensor, target_tensor):
		self.user_tensor = user_tensor
		self.item_tensor = item_tensor
		self.target_tensor = target_tensor

	def __getitem__(self, item):
		return self.user_tensor[item], self.item_tensor, self.target_tensor[item]

	def __len__(self):
		return self.user_tensor.size(0)

"""
Because we may sample some nodes from the users and items, and then do the element-wise production,
so sample number must be the same. And the data sampler extract the samples from the DataFrame
"""
class SampleGenerator():
	"""
	Construct dataset for NCF, check there are corresponding columns in the init ratings
	"""
	def __init__(self, ratings):
		assert 'userId' in ratings.columns
		assert 'itemId' in ratings.columns
		assert 'rating' in ratings.columns

		self.ratings = ratings
		# explicit feedback using _normalize to limit thsoe values between 0 and 1, and implicit using _binarize
		self.processing_ratings = self.binarize(self.ratings)
		# statistic all, now the user_pool has all userId, and item_pool has all itemId
		self.user_pool = set(ratings['userId'])
		self.item_pool = set(ratings['itemId'])
		# create positive item samples for NCF learning
		self.negative = self._sample_negative(ratings)
		# we now split the preprocessing data into train, test set
		self.train, self.test = self._split_loo(self.processing_ratings)
		self.instance_train_loader(num_negative = 99, batch_size = 10)
		"""
		There are two methods to pre-process those data, limit to (0, 1) or 0 and 1
		"""
	def normalize(self, ratings):
		ratings = deepcopy(ratings)
		max_rating = ratings.rating.max()
		ratings.rating = ratings.rating * 1.0 / max_rating
		return ratings
	def binarize(self, ratings):
		ratings = deepcopy(ratings)
		ratings.rating[ratings['rating'] > 0] = 1.0
		return ratings

	def _sample_negative(self, ratings):
		# here is one method to change the column "itemId" to become the "iteracted_items"
		interact_status = ratings.groupby('userId')['itemId'].apply(set).reset_index().rename(
			columns = {'itemId':'interacted_items'}
		)
		# select all negative items(no connection with all the users)
		interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: self.item_pool - x)
		interact_status['sample_negative'] = interact_status['negative_items'].apply(lambda x: random.sample(x, 99))

		return interact_status[['userId', 'negative_items', 'sample_negative']]

	def _split_loo(self, ratings):
		ratings = deepcopy(ratings)
		ratings['rank_latest'] = ratings.groupby('userId')['timestamp'].rank(method = 'first', ascending = False)
		test = ratings[ratings['rank_latest'] == 1]
		train = ratings[ratings['rank_latest'] > 1]
		return train[['userId','itemId','rating']], test[['userId','itemId','rating']]

	def instance_train_loader(self, num_negative, batch_size):
		users, items, ratings = [], [], []
		train_ratings = pd.merge(self.train, self.negative[['negative_items', 'userId']], on = 'userId')
		# num_negative is the number of negative items which we want to sample
		train_ratings['negatives'] = train_ratings['negative_items'].apply(lambda x: random.sample(x, num_negative))
		for row in train_ratings.itertuples():
			"""
			we should pay attention to that self.train, all positive items are in it
			"""
			users.append(int(row.userId))
			items.append(int(row.itemId))
			ratings.append(float(row.rating))
			for i in range(num_negative):
				users.append(int(row.userId))
				items.append(int(row.negatives[i]))
				ratings.append(float(0))# negative samples get 0 rating
		dataset = UserItemRatingDataset(user_tensor = torch.LongTensor(users), item_tensor = torch.LongTensor(items),
										target_tensor = torch.FloatTensor(ratings))
		return DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True)

	@property
	def evaluate_data(self):
		""" create evaluate data """
		test_ratings = pd.merge(self.test, self.negative[['userId', 'sample_negative']], on = 'userId')
		test_users, test_items, negative_users, negative_items = [],[],[],[]
		for row in test_ratings.itertuples():
			test_users.append(int(row.userId))
			test_items.append(int(row.itemId))
			for i in range(len(row.sample_negative)):
				negative_users.append(int(row.userId))
				negative_items.append(int(row.sample_negative[i]))
		return [torch.LongTensor(test_users), torch.LongTensor(test_items),
				torch.LongTensor(negative_users), torch.LongTensor(negative_items)]

class Data():
	def __init__(self, file_path):
		# load data
		ml1m_rating = pd.read_csv(file_path, delimiter = ',', engine = 'python')
		print("the range of userID is [{},{}]".format(ml1m_rating.userId.min(), ml1m_rating.userId.max()))
		print("the range of itemID is [{},{}]".format(ml1m_rating.itemId.min(), ml1m_rating.itemId.max()))
		self.columns = ml1m_rating.columns
		self.ml1m_rating = ml1m_rating

		self.num_users = ml1m_rating.userId.max() + 1
		self.num_items = ml1m_rating.itemId.max() + 1
		self.users_list = np.array(ml1m_rating.userId)
		self.items_list = np.array(ml1m_rating.itemId)

		self.ratings = np.array(ml1m_rating.rating)
		# adjacency matrix
		#self.norm_ratings_adj = self.create_rating_adj(self.ratings)
		# print the adjacency matrix shape
		#print("the shape of the adj matrix: ", self.norm_ratings_adj.shape)
		self.ratings_df = ml1m_rating[ml1m_rating.columns[-4:]]
		print("the columns of the ratings_df: \n", self.ratings_df.columns)
		# a new class for sample Data
		sample_generator = SampleGenerator(self.ratings_df)
		self.sample = sample_generator

	def sample_gen_evaluate(self):
		sample_evaluate = self.sample
		return sample_evaluate, sample_evaluate.evaluate_data

	def get_users_items_nums(self):
		return self.num_users, self.num_items
	def get_ratings(self):
		return self.ratings_df
	def create_rating_adj(self, ratingList):
		adj_mat = sp.dok_matrix((self.num_users, self.num_items))
		# transform the constructure of the adj_mat
		adj_mat = adj_mat.tolil()
		adj_mat[self.users_list, self.items_list] = ratingList
		adj_mat = adj_mat.tolil()
		return adj_mat



if __name__ == "__main__":
	data = Data('ratings.csv')
	"""
	print("the list of users:\n", data.users_list.shape)
	print("the list of items \n", data.items_list.shape)
	print("****: \n", data.ratings.shape)
	print("the shape of rating matrix \n", data.norm_ratings_adj.shape)
	print("get ratings: \n", data.get_ratings()[:100])
	"""

