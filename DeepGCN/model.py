import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import time
import random
from torch.nn import init
import dgl.data.citation_graph as citation
from construct_graph import construct_graph, TransformationMatrix

class GCNlayer(nn.Module):
	def __init__(self, in_feats, out_feats):
		super(GCNlayer, self).__init__()
		self.linear = nn.Linear(in_feats, out_feats)
	def forward(self, Z, features):
		return self.linear(Z.mm(features))

class GCNnet(nn.Module):
	def __init__(self, in_feats, hidden_feats, out_feats):
		super(GCNnet, self).__init__()
		self.layer1 = GCNlayer(in_feats, hidden_feats)
		self.layer2 = GCNlayer(hidden_feats, out_feats)
	def forward(self, Z, features):
		h = F.relu(self.layer1(Z, features))
		h = self.layer2(Z, h)
		return h

def label_list(sample_label, labels, num_classes):
	# we get our train labels:
	train_sample = labels[sample_label]
	# we combine the order and label value
	combined_sample = [(label, order) for label, order in zip(train_sample, sample_label)]
	integrated = []
	for i in range(num_classes):
		temp = []
		for element in combined_sample:
			if element[0] == i:
				temp.append(element[1])
		integrated.append(set(temp))
	numberoflabels_in_list = list(map(lambda x: len(x), integrated))
	return integrated, numberoflabels_in_list

def load_Coradata():
	# load the data from citation
	data = citation.load_cora()
	# load all labels
	labels = torch.LongTensor(data.labels)
	# load num_classes
	num_classes = data.num_labels
	graph = data.graph
	features = torch.FloatTensor(data.features)
	return graph, num_classes, labels, features

def run_origin_GCN(ratio_train):
	graph, num_classes, labels, features = load_Coradata()
	sample = list(range(ratio_train))
	# make sure the dimension of features
	model = construct_graph(graph)
	# run adj_matrix to get the adjmatrix
	model.adjent_matrix()
	# run normalized degree matrix
	model.degree_matrix()
	# finally we have the D^-0.5AD^-0.5
	Z = model.renormalized_adj()
	# in_feats, hidden_feats is the first layer linear transform
	# hidden_feats, out_feats is the second layer linear transform
	in_feats = features.size(1)
	hidden_feats = 100
	out_feats = num_classes
	features = nn.Parameter(features, requires_grad = False)
	# build the model
	net = GCNnet(in_feats = in_feats, hidden_feats = hidden_feats, out_feats = out_feats)
	# give the optimizer
	optimizer = torch.optim.Adam(net.parameters(), lr = 0.01)

	for epoch in range(100):
		net.train()
		logits = net(Z, features)
		loss = nn.CrossEntropyLoss()
		output = loss(logits[sample], labels[sample])
		optimizer.zero_grad()
		output.backward()
		optimizer.step()
		print("epoch:", epoch)
		print("loss value:", output.detach().item())
		acc = evaluate(net, Z, features, labels)
		print("accuracy value:", acc)


def run_data(ratio_train = 100):
	graph, num_classes, labels, features = load_Coradata()
	sample = list(range(ratio_train))
	label_classification, num_labels = label_list(sample_label = sample, labels = labels, num_classes = num_classes)
	# so we need to make sure the value of K. the window size of the selected elements in one loop
	# construct the transform-possibility matrix
	transform_model = TransformationMatrix(graph)
	transform_model.LaplacianMatrix()
	transform_model.TransformMatrix(alpha = 0.5)
	possibility_matrix = transform_model.TransformMatrix(0.1)
	print(possibility_matrix)

def evaluate(model, Z, feature_matrix, labels):
	model.eval()
	logits = model(Z, feature_matrix)
	_, indices = torch.max(logits, dim = 1)
	correct = torch.sum(indices == labels)
	return correct.detach().item() * 1.0 / len(labels)

if __name__ == "__main__":
	# now to load
	run_data()



