import dgl.data.citation_graph as cit_graph
from dgl import DGLGraph
import networkx as nx
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
import matplotlib.pylab as plt
import os
from scipy import expm1

def construct_graph(u, v):
	dimension = u.size + 1
	# we build up a zeros matrix dimension * dimension
	A = np.zeros((dimension, dimension))
	# make sure the connection location in adjancy matrix
	for src, dest in zip(u, v):
		A[src][dest] = 1
		A[dest][src] = 1
	return A

def renormlizeMatrix(A):
	dimension = A.shape
	A = A + np.eye(dimension[0], dimension[1])
	Degrees = np.sum(A, axis = 1)
	D = np.diag(Degrees**-0.5)
	return D, A

def Fourier_transform(u, v):
	# first compute the adjancy matrix
	A = construct_graph(u, v)
	# renormalization
	D, A = renormlizeMatrix(A)
	Z = np.dot(np.dot(D, A), D)
	return Z

class GCNlayer(nn.Module):
	def __init__(self, in_feats, out_feats):
		super(GCNlayer, self).__init__()
		self.linear = nn.Linear(in_feats, out_feats)
	def forward(self, Z, features):
		return self.linear(Z @ features)

def evaluate(model, Z, feature_matrix, targets):
	model.eval()
	logits = model(Z, features_matrix)
	_, indices = torch.max(logits, dim = 1)
	correct = torch.sum(indices == targets)
	return correct.item() * 1.0 / len(targets)

class Net(nn.Module):
	def __init__(self, in_feats, hidden_feats, out_feats):
		super(Net, self).__init__()
		self.layer1 = GCNlayer(in_feats, hidden_feats)
		self.layer2 = GCNlayer(hidden_feats, out_feats)
	def forward(self, Z, features):
		h = F.relu(self.layer1(Z, features))
		h = self.layer2(Z, h)
		return h


if __name__ == "__main__":
	os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
	# we pre-set some vertices, including sources and destinations
	u = np.zeros(49, dtype = np.int)
	v = np.arange(1,50)
	# set adjancy matrix
	Z = Fourier_transform(u, v)
	# transform Z into the torch scale
	Z = torch.FloatTensor(Z)
	# pre-define a N * C matrix, N is the number of Nodes, C is the number of channels.
	features_matrix =  torch.randn((50,100), dtype = torch.float32)
	# give the examples of in_feats, hidden_features, out_features
	in_feats = 100
	hidden_feats = 30
	out_feats = 5
	# set targets
	targets = torch.randint(0,4,(1,50)).squeeze(0)
	# build the model
	#model = GCNlayer(in_feats, hidden_feats)
	# build the net
	net = Net(in_feats, hidden_feats, out_feats)
	optimizer = torch.optim.Adam(net.parameters(), lr = 0.01)

	for epoch in range(50):
		net.train()
		logits = net(Z, features_matrix)
		loss = nn.CrossEntropyLoss()
		output = loss(logits, targets)
		optimizer.zero_grad()
		output.backward()
		optimizer.step()
		acc = evaluate(net, Z, features_matrix, targets)
		print("accuracy:", acc)
		print("print the loss value:",output.detach().item())











