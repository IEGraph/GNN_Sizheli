import torch.nn as nn
import torch
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F

class Encoder(nn.Module):
	def __init__(self, features, feature_dim, embed_dim, adj_lists, aggregator, num_sample = 10,
				 base_model = None, gcn = False, cuda = False):
		super(Encoder, self).__init__()
		self.features = features
		self.features_dim = feature_dim
		self.adj_lists = adj_lists
		self.aggregator = aggregator
		# the num_sample of the neighbors
		self.num_sample = num_sample
		if base_model != None:
			self.base_model = base_model
		self.gcn = gcn
		self.embed_dim = embed_dim
		self.cuda = cuda
		self.aggregator.cuda = cuda
		# we set the weight as the Parameter form,
		if self.gcn:
			self.weight = nn.Parameter(torch.FloatTensor(self.features_dim, self.embed_dim))
		else:
			self.weight = nn.Parameter(torch.FloatTensor(2 * self.features_dim, self.embed_dim))
		#init.xavier_uniform(self.weight)
	def forward(self, nodes):
		"""
		:param nodes: list of nodes
		:return: combined matrix
		"""
		neigh_feats = self.aggregator.forward(nodes, [self.adj_lists[int(node)] for node in nodes], self.num_sample)
		if not self.gcn:
			if self.cuda:
				self_feats = self.features(torch.LongTensor(nodes).cuda())
			else:
				self_feats = self.features(torch.LongTensor(nodes))
			combined = torch.cat([self_feats, neigh_feats], dim = 1)
		else:
			combined = neigh_feats
		combined = F.relu(combined.mm(self.weight))
		return combined



