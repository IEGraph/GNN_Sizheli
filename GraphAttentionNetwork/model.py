import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import time
import random
import networkx as nx
import numpy as np
import time

from construct_graph import construct_graph
from GALayer import Aggregator
from encode import Encoder

class GAN(nn.Module):
	def __init__(self, num_classes, enc, node_num):
		super(GAN, self).__init__()
		self.enc = enc
		self.xnent = nn.CrossEntropyLoss()
		self.weight = nn.Parameter(torch.FloatTensor(node_num, num_classes), requires_grad = True)
		init.xavier_uniform_(self.weight)
	def forward(self, nodes):
		embeds = self.enc(nodes)
		result = embeds.mm(self.weight)
		return result
	def loss(self, nodes, labels):
		output = self.forward(nodes)
		return self.xnent(output, labels)


def load_data(N):
	# load the data from nx package, 100 is the number of nodes, and p is the possibility
	graph = nx.erdos_renyi_graph(N, 0.1)
	# construct the graph
	model = construct_graph(graph)
	# product the neighbor list
	neighbors = model.index_adjmatrix()
	# random access to the features
	features_model = model.construct_features(5)
	feature_size = features_model.size()
	# construct the embedding of feature
	features = nn.Embedding(feature_size[0], feature_size[1])
	features.weight = nn.Parameter(features_model, requires_grad = False)
	# we had to deliberately create some labels
	label_num = len(model.nodes)
	label = torch.randint(low = 0, high = 6, size = (label_num,))
	return neighbors, features, label

def run_data():
	neighbors, features, label = load_data(200)
	#print("label of nodes:",label)
	#print("neighbors of nodes list:", neighbors)
	agg1 = Aggregator(features)
	# select nodes from nodes

	features_dim = features.weight.size(1)
	enc1 = Encoder(features = features, features_dim = features_dim, emded_dim = 5,
				   adj_lists = neighbors, aggregator = agg1)
	graphAttention = GAN(num_classes = 6, enc = enc1, node_num = 200)
	rand_indices = np.random.permutation(len(label))
	test = list(rand_indices[:50])
	val = list(rand_indices[50:80])
	train = list(rand_indices[80:])
	all = list(range(200))
	optimizer = torch.optim.SGD(filter(lambda p:p.requires_grad, graphAttention.parameters()), lr = 0.07)

	#output = graphAttention(train)
	#print(output)

	for batch in range(200):
		batch_nodes = all
		random.shuffle(batch_nodes)
		start_time = time.time()
		optimizer.zero_grad()
		loss = graphAttention.loss(batch_nodes, Variable(torch.LongTensor(label[batch_nodes])))
		loss.backward()
		optimizer.step()
		end_time = time.time()
		during = end_time - start_time
		print(batch, loss.data.detach().item(),"  ", during)

if __name__ == "__main__":
	run_data()

