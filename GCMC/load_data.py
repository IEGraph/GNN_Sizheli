import torch
from torch_geometric.data import InMemoryDataset, Data, download_url, extract_zip
from torch_scatter import scatter_add

import copy
import glob
import shutil
import pandas as pd
import numpy as np
import os

class MCDataset(InMemoryDataset):
	def __init__(self, root, name, transform = None, pre_transform = None):
		self.name = name
		super(MCDataset, self).__init__(root = root, transform = transform, pre_transform = pre_transform)
		print("location of raw_dir: ",self.raw_dir)
		print("location of root: ", self.root)
		# it shows that those raw files are stored in raw_paths
		print("raw paths: ", self.raw_paths)
		print(os.path.join(self.raw_dir, self.name))
		self.data, self.slices = torch.load(self.processed_paths[0])

	@property
	def num_relations(self):
		return self.data.edge_type.max().item() + 1

	@property
	def num_nodes(self):
		return self.data.x.shape[0]

	@property
	def raw_file_names(self):
		return ['u1.base', 'u1.test']

	@property
	def processed_file_names(self):
		return ['data.pt']

	def download(self):
		if self.name == 'ml-100k':
			url = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
		path = download_url(url, self.root)
		extract_zip(path = path, folder = self.raw_dir, log = self.name)
		os.unlink(path)
		for file in glob.glob(os.path.join(self.raw_dir, self.name, '*')):
			shutil.move(file, self.raw_dir)
		#os.rmdir(os.path.join(self.raw_dir, self.name))

	def process(self):
		train_csv, test_csv = self.raw_paths
		train_df, train_nums = self.create_df(train_csv)
		test_df, test_nums = self.create_df(test_csv)

		train_idx, train_gt = self.create_gt_idx(df = train_df, nums = train_nums)
		test_idx, test_gt = self.create_gt_idx(df = test_df, nums = test_nums)

		print("the train_idx:\n", train_idx )

	def create_df(self, csv_path):
		columns = ['user_id','item_id', 'ratings', 'ts']
		# the separation is '\t'
		df = pd.read_csv(csv_path, sep = '\t', names = columns)
		# drop the forth column
		df.drop(['ts'], axis = 1)
		num_users = df['user_id'].max()
		num_items = df['item_id'].max()

		df['user_id'] = df['user_id'] - 1
		df['item_id'] = df['item_id'] - 1
		df['ratings'] = df['ratings'] - 1

		nums = {'user':num_users, 'item':num_items, 'nodes':num_users + num_items, 'edge': len(df)}
		return df, nums

	def create_gt_idx(self, df, nums):
		df['idx'] = df['user_id'] * nums['item'] + df['item_id']
		idx = torch.tensor(df['idx'])
		gt = torch.tensor(df['ratings'])
		return idx, gt



if __name__ == "__main__":
	current_path = './data/ml-100k'
	name = 'ml-100k'
	data = MCDataset(root = current_path, name = name)
	"""
	test_path = os.path.join(current_path, 'raw', 'u1.base')
	file = pd.read_csv(test_path, sep = '\t', names = ['user_id','item_id', 'ratings', 'ts'])
	file = file.drop(['ts'], axis = 1)
	file['idx'] = file['user_id'] * (file['item_id'].max()+1) + file['item_id']
	"""






