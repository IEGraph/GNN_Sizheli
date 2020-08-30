import torch
import torch.nn as nn
from torch.autograd import Variable
from sample_neighbors import construct_graph
import random
import dgl.data.citation_graph as citation
"""
Set of modules for aggregating embeddings of neighbors.
"""

class MeanAggregator(nn.Module):
	"""
	Aggregates a node's embeddings using mean of neighbors' embeddings
	"""
	def __init__(self, features, cuda = False, gcn = False):
		"""
		:param features: function mapping LongTensor of node ids to FloatTensor of feature values.
		:param cuda: whether to use GPU
		:param gcn:
		"""
		super(MeanAggregator, self).__init__()
		self.features = features
		self.cuda = cuda
		self.gcn = gcn

	def forward(self, nodes, to_neighs, num_sample = 10):
		"""
		:param nodes: list of nodes in a batch
		:param to_neighs: list of sets, each set is the set of neighbors for node in batch
		:param num_sample: number of neighbors to sample.
		:return:
		"""
		_set = set
		if not num_sample is None:
			_sample = random.sample
			samp_neighs = []
			for to_neigh in to_neighs:
				if len(to_neigh) > num_sample:
					samp_neighs.append(_set(_sample(to_neigh, num_sample)))
				else:
					samp_neighs.append(to_neigh)
		else:
			samp_neighs = to_neighs
		if self.gcn:
			samp_neighs = [samp_neigh + set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
		#print(len(samp_neighs))
		unique_nodes_list = list(set.union(*samp_neighs))
		#print("the length of node_list:", len(unique_nodes_list))
		unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}
		#print("the length of nodes dic:", len(unique_nodes))
		mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
		column_indices = []
		for samp_neigh in samp_neighs:
			for n in samp_neigh:
				column_indices.append(unique_nodes[n])
		row_indices = []
		for i in range(len(samp_neighs)):
			for j in range(len(samp_neighs[i])):
				row_indices.append(i)
		mask[row_indices, column_indices] = 1
		# if the cuda is ok, then uese cuda
		if self.cuda:
			mask = mask.cuda()
		num_neigh = mask.sum(1, keepdim = True)
		mask = mask.div(num_neigh)
		if self.cuda:
			embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
		else:
			embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
		#print(embed_matrix.dtype)
		to_feats = mask.mm(embed_matrix)
		return to_feats
if __name__ == "__main__":
	# we load the cora data
	data = citation.load_cora()
	# load the graph
	graph = data.graph
	# build the graph class
	model = construct_graph(graph)
	# compute the adj matrix of the model
	model.adjency_matrix()
	# build the instrance of the MeanAggregator
	features = nn.Embedding(2708, 1433)
	features.weight = nn.Parameter(torch.FloatTensor(data.features), requires_grad = False)

	net = MeanAggregator(features)
	nodes = model.nodes
	# find the neighbors of corresponding nodes aggregator.py:81
	to_neighs = model.index_adjmatrix()
	result = net(nodes, to_neighs)
	# now test the result of the to_feats, check the dtype and dimension of
	# matrix




