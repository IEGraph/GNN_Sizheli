import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import random
from copy import deepcopy
import scipy.sparse as sp
from Sample import SampleGenerator

class data_loader():
	def __init__(self, file_path):
		# load data
		columns = ['user_id','item_id', 'ratings', 'ts']
		# the separation is '\t'
		ratings_df = pd.read_csv(file_path, sep = '\t', names = columns, engine = 'python')
		ratings_df['user_id'] = ratings_df['user_id'] - 1
		ratings_df['item_id'] = ratings_df['item_id'] - 1
		ratings_df['ratings'] = ratings_df['ratings'] - 1

		print("the range of userID is [{},{}]".format(ratings_df.user_id.min(), ratings_df.user_id.max()))
		print("the range of itemID is [{},{}]".format(ratings_df.item_id.min(), ratings_df.item_id.max()))

		self.num_users = ratings_df.user_id.max() + 1
		self.num_items = ratings_df.item_id.max() + 1

		# unique the ratings
		self.ratings_level = set(ratings_df['ratings'])
		print("ratings_level: ", self.ratings_level)
		rating_level_adj_list = []
		for rating in self.ratings_level:
			user_list = list(ratings_df[ratings_df['ratings'] == rating]['user_id'])
			item_list = list(ratings_df[ratings_df['ratings'] == rating]['item_id'])
			rating_level_adj_list.append([user_list] + [item_list])
		print("the length of our list: ",len(rating_level_adj_list))
		adj_matrix = []
		for element in rating_level_adj_list:
			unit = self.adj_matrix(element[0], element[1])
			print(unit.shape)
			adj_matrix.append(unit)
		self.norm_adjs = adj_matrix
		self.sample_generator = SampleGenerator(ratings_df)
	def construct_config(self):
		config = {'layers': [10], 'embed_size': 8, 'optimizer': 'adam',
				  'adam_lr': 1e-4, 'l2_regularization': 0, 'batch_size':1000,
				  'num_epoch':10,
				  'use_cuda': True if torch.cuda.is_available() else False}
		config['n_users'] = self.num_users
		config['n_items'] = self.num_items
		config['norm_adjs'] = self.norm_adjs
		return config
	def adj_matrix(self, user_list, item_list):
		adj_mat = sp.dok_matrix((self.num_users, self.num_items), dtype = np.float32)
		# transform the construction of the adj_mat
		adj_mat = adj_mat.tolil()
		adj_mat[user_list, item_list] = 1
		full_mat = sp.dok_matrix((self.num_items + self.num_users, self.num_users + self.num_items),
								 dtype = np.float32).tolil()
		full_mat[:self.num_users, self.num_users:] = adj_mat
		full_mat[self.num_users:, :self.num_users] = adj_mat.T
		def mean_adj_single(adj):
			# D^-1 * A
			row_sum = np.array(adj.sum(axis = 1))
			d_inv = np.power(row_sum, -1).flatten()
			d_inv[np.isinf(d_inv)] = 0
			d_mat_inv = sp.diags(d_inv)
			norm_adj = np.dot(d_mat_inv, adj)
			return norm_adj
		mean_adj_mat = mean_adj_single(full_mat + sp.eye(full_mat.shape[0]))
		return mean_adj_mat



if __name__ == "__main__":
	file_path = './data/ml-100k/raw/u1.base'
	data = data_loader(file_path)