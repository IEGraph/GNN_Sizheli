import torch.nn as nn
import torch
import numpy
import random
from construct_graph import construct_graph
from torch.autograd import Variable

class Aggregator(nn.Module):
	"""
	Aggregator collects a node's embedding from its neighbors.
	"""
	def __init__(self, features):
		"""
		:param features: node features
		"""
		super(Aggregator, self).__init__()
		self.features = features
	def forward(self, nodes, to_neighs):
		"""
		:param nodes:
		:param to_neighs:
		:return:
		"""
		_set = set
		samp_neighs = to_neighs
		samp_neighs = [samp_neigh.union(set([nodes[i]])) for i,samp_neigh in enumerate(samp_neighs)]
		#print(samp_neighs)
		unique_node_list = list(set.union(*samp_neighs))
		unique_nodes = {n:i for i,n in enumerate(unique_node_list)}
		mask = Variable(torch.zeros(len(samp_neighs), len(unique_node_list)))
		column_indices = []
		for samp_neigh in samp_neighs:
			for n in samp_neigh:
				column_indices.append(unique_nodes[n])
		row_indices = []
		for i in range(len(samp_neighs)):
			for j in range(len(samp_neighs[i])):
				row_indices.append(i)
		mask[row_indices, column_indices] = 1
		# now we do the mean of h
		num_neigh = mask.sum(1, keepdim = True)
		mask = mask.div(num_neigh)
		embed_matrix = self.features(torch.LongTensor(unique_node_list))
		to_feats = mask.mm(embed_matrix)

		return to_feats
