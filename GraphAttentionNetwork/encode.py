import torch.nn as nn
import torch
from torch.nn import init
from torch.autograd import Variable
from GALayer import Aggregator
import torch.nn.functional as F

class Encoder(nn.Module):
	def __init__(self, features, features_dim, emded_dim, adj_lists, aggregator, concate = True):
		"""
		:param features: mapping functions to ccorresponding features
		:param features_dim: features dimension
		:param emded_dim: the transformed feature matrix dimension
		:param adj_lists: neighbors of nodes()
		:param aggregator:
		"""
		super(Encoder, self).__init__()
		self.features = features
		self.features_dim = features_dim
		self.aggregator = aggregator
		self.emded_dim = emded_dim
		self.cat = concate
		if self.cat:
			self.weight = nn.Parameter(torch.FloatTensor(2 * self.features_dim, self.emded_dim))
		else:
			self.weight = nn.Parameter(torch.FloatTensor(self.features_dim, self.emded_dim))
		self.adj_list = adj_lists
		init.xavier_uniform_(self.weight)
	def forward(self, nodes):
		"""
		:param nodes:
		:return:
		"""
		# the dimension of neigh_feats is (node_number, origin_feature dimension)
		neigh_feats = self.aggregator.forward(nodes, [self.adj_list[int(node)] for node in nodes])
		self_feats = self.features(torch.LongTensor(nodes))
		if self.cat:
			combined = torch.cat([self_feats, neigh_feats], dim = 1)
		else:
			combined = neigh_feats
		# dimension of hidden, (node_number, origin)
		hidden_feats = combined.mm(self.weight)
		# output is the connection matrix of nodes
		output = F.leaky_relu(hidden_feats.mm(hidden_feats.t()), negative_slope = 0.2)
		output = F.softmax(output, dim = 1)
		return output

