import networkx as nx
import matplotlib.pylab as plt
import torch.nn as nn
import torch
import torch.nn.functional as F
from dgl.data import MiniGCDataset
from dgl.nn.pytorch import GraphConv
import dgl
import torch.optim as optim
from torch.utils.data import DataLoader

def collate(samples):
	# The input "sample" is a list of pairs
	# (graph, label)
	graphs, labels = map(list, zip(*samples))
	batched_graph = dgl.batch(graphs)
	return batched_graph, torch.tensor(labels)

class Classifier(nn.Module):
	def __init__(self, in_dim, hidden_dim, n_classes):
		super(Classifier, self).__init__()
		self.conv1 = GraphConv(in_dim, hidden_dim)
		self.conv2 = GraphConv(hidden_dim, hidden_dim)
		self.classify = nn.Linear(hidden_dim, n_classes)
	def forward(self, g):
		# Use node degree as the initial node feature
		h = g.in_degrees().view(-1, 1).float()
		# Perform graph convolution and activation function.
		h = F.relu(self.conv1(g, h))
		h = F.relu(self.conv2(g, h))
		g.ndata["h"] = h
		# Calculate graph representation by averaging all the node repres
		hg = dgl.mean_nodes(g, "h")
		return self.classify(hg)

if __name__ == "__main__":
	# a dataset with 80 samples, each graph is of size(10, 20)
	dataset = MiniGCDataset(80, 10, 20)
	# Create training and test sets.
	trainset = MiniGCDataset(320, 10, 20)
	testset = MiniGCDataset(80, 10, 20)
	"""
	data_loader = DataLoader(trainset, batch_size = 32, shuffle = True, collate_fn = collate)
	print(trainset.num_classes, trainset.num_graphs)
	model = Classifier(1, 256, trainset.num_classes)
	loss_func = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr = 0.001)
	for epoch in range(80):
		epoch_loss = 0
		for iter, (bg, label) in enumerate(data_loader):
			predict = model(bg)
			loss = loss_func(predict, label)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			epoch_loss += loss.detach().item()
			print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
		epoch_loss /= (iter + 1)
		print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
		"""
	batch_graph, labels = collate(dataset)
	print(labels.size())
	print(batch_graph.adjacency_matrix())






