import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import time
import numpy as np
import time
import random
from sklearn.metrics import f1_score

from aggregator import MeanAggregator
from encoder import Encoder
from sample_neighbors import construct_graph
import dgl.data.citation_graph as citation

class SupervisedGraphSage(nn.Module):
	def __init__(self, num_classes, enc):
		super(SupervisedGraphSage, self).__init__()
		self.enc = enc
		self.loss = nn.CrossEntropyLoss()

		self.weight = nn.Linear(enc.embed_dim, num_classes)
		# init.xavier_uniform(self.weight)
	def forward(self, nodes):
		embeds = self.enc(nodes)
		result = self.weight(embeds)
		return result
	def ComputeLoss(self, nodes, labels):
		result = self.forward(nodes)
		return self.loss(result, labels)

def load_data():
	# load the data from package
	data = citation.load_cora()
	# load the labels of every node
	labels = data.labels
	# load the num_classes
	num_classes = data.num_labels
	# load the graph and build the model
	model = construct_graph(data.graph)
	# compute the adjency_matrix
	model.adjency_matrix()
	# build the instance of the MeanAggregator
	data_features = data.features
	features = nn.Embedding(data_features.shape[0], data_features.shape[1])
	features.weight = nn.Parameter(torch.tensor(data_features, dtype = torch.float32), requires_grad = False)
	nodes = model.nodes
	# find the neighbors of corresponding nodes aggregator.py
	adj_list = model.index_adjmatrix()

	return features, adj_list,labels

def run_cora():
	np.random.seed(100)
	random.seed(1)
	features, adj_list, labels = load_data()
	num_nodes = len(labels)
	agg1 = MeanAggregator(features, cuda = False)
	enc1 = Encoder(features, 1433, 128, adj_lists = adj_list, aggregator = agg1,
				   gcn = True, cuda = False, num_sample = 5)
	agg2 = MeanAggregator(lambda nodes: enc1(nodes), cuda = False)
	enc2 = Encoder(lambda nodes: enc1(nodes), enc1.embed_dim, 128, adj_lists = adj_list,
				   aggregator = agg2, gcn = True, cuda = False, num_sample = 5)

	train1 = list(range(500))
	graphsage = SupervisedGraphSage(7, enc2)
	result = graphsage(train1)
	# split train, val, test
	rand_indices = np.random.permutation(num_nodes)
	test = rand_indices[:1000]
	val = rand_indices[1000:1500]
	train = list(rand_indices[1500:])

	optimizer = torch.optim.SGD(filter(lambda p:p.requires_grad, graphsage.parameters()), lr = 0.7)
	times = []
	for batch in range(200):
		batch_nodes = train
		random.shuffle(train)
		start_time = time.time()
		optimizer.zero_grad()
		loss = graphsage.ComputeLoss(batch_nodes, Variable(torch.LongTensor(labels[batch_nodes])))
		loss.backward()
		optimizer.step()
		end_time = time.time()
		times.append(end_time - start_time)
		print(batch, loss.data.detach().item())


if __name__ == "__main__":
	run_cora()


