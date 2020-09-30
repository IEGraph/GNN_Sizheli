import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from sparse_matrix import spar_matrix
from Engine import Engine

'https://github.com/tanimutomo/gcmc'

class GCMC(nn.Module):
	def __init__(self, config):
		super(GCMC, self).__init__()
		self.n_users = config['n_users']
		self.n_items = config['n_items']
		self.layers = config['layers']
		self.norm_adjs = config['norm_adjs']
		self.embed_size = config['embed_size']
		self.integrated_dimensions = config['layers'][-1] * len(config['norm_adjs'])

		self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adjs)
		self.embed_dict, self.weight_dict = self.init_weight()

		self.affine_output = nn.Linear(in_features = self.integrated_dimensions, out_features = self.embed_size)
	def init_weight(self):
		# xavier init
		initializer = nn.init.xavier_uniform_
		# construct the dictionary for the embedding_dict
		embedding_dict = nn.ParameterDict({
			'user_emb': nn.Parameter(initializer(torch.empty(self.n_users, self.embed_size))),
			'item_emb': nn.Parameter(initializer(torch.empty(self.n_items, self.embed_size)))
		})
		weight_dict = nn.ParameterDict()
		layers = [self.embed_size] + self.layers
		for k in range(len(self.norm_adjs)):
			weight_dict.update({"Weight_%d" % k: nn.Parameter(initializer(torch.empty(layers[0], layers[1])))})
			weight_dict.update({"Bias_%d" % k: nn.Parameter(initializer(torch.empty(1, layers[1])))})
		return embedding_dict, weight_dict

	# write f function to transform the sparse coordinate form to tensor
	def _convert_sp_mat_to_sp_tensor(self, X):
		norm_adjs = []
		# transform the X into the coordinate format
		for x in X:
			coo = x.tocoo()
			indices = torch.LongTensor([coo.row, coo.col])
			values = torch.from_numpy(coo.data).float()
			norm_adjs.append(torch.sparse_coo_tensor(indices = indices, values = values, size = coo.shape))
		return norm_adjs

	def rating(self, u_g_embeddings, i_g_embeddings):
		return u_g_embeddings.mm(i_g_embeddings.t())

	def forward(self, users, items):
		L_user_item = self.sparse_norm_adj
		E_embeddings = torch.cat([self.embed_dict.user_emb, self.embed_dict.item_emb], dim = 0)
		M_all = []
		for k in range(len(L_user_item)):
			# L = D^-1
			# do the one layer multiply
			Hide_state = torch.sparse.mm(L_user_item[k], E_embeddings).mm(self.weight_dict['Weight_%d' % k]) + \
						self.weight_dict['Bias_%d' % k]
			M_all.append(Hide_state)
		# concate all embeddings
		M_all = torch.cat(M_all, dim = 1)
		M_all = nn.LeakyReLU(negative_slope = 0.2)(M_all)
		# split users, items into their own embeddings
		logits = self.affine_output(M_all)
		# now split the embeddings
		u_embeddings = logits[:self.n_users, :]
		i_embeddings = logits[self.n_users:, :]

		u_embeddings = u_embeddings[users, :]
		i_embeddings = i_embeddings[items, :]

		ratings = u_embeddings * i_embeddings
		return torch.sum(ratings, dim = 1)

class GCMCEngine(Engine):
	def __init__(self, config):
		self.model = GCMC(config = config)
		if config['use_cuda']:
			self.model.cuda()
		super(GCMCEngine, self).__init__(config)

if __name__ == "__main__":
	config = {'n_users':20, 'n_items':20, 'layers':[10], 'embed_size':8, 'optimizer': 'adam',
			  'adam_lr': 1e-3, 'l2_regularization': 0,
			  'use_cuda': True if torch.cuda.is_available() else False}
	rating_level = 5
	norm_adjs = []
	for i in range(rating_level):
		norm_adjs.append(spar_matrix(users = config['n_users'], items = config['n_items'],
									 ratings = 30))
	config['norm_adjs'] = norm_adjs
	gcmc_model = GCMCEngine(config)
	users = [0,2,4,5]
	items = [2,7,8,9]
	result = gcmc_model.model(users, items)
	print(result)



