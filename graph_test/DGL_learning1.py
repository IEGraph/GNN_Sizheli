import dgl
import numpy as np
import torch
import networkx as nx
import matplotlib.pylab as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
import itertools

def build_karate_club_graph():
	# all 78 nodes are stored in two numpy arrays, One for the start nodes, and others are destination nodes
	src = np.array([1, 2, 2, 3, 3, 3, 4, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 9, 10, 10,
					10, 11, 12, 12, 13, 13, 13, 13, 16, 16, 17, 17, 19, 19, 21, 21,
					25, 25, 27, 27, 27, 28, 29, 29, 30, 30, 31, 31, 31, 31, 32, 32,
					32, 32, 32, 32, 32, 32, 32, 32, 32, 33, 33, 33, 33, 33, 33, 33,
					33, 33, 33, 33, 33, 33, 33, 33, 33, 33])
	dst = np.array([0, 0, 1, 0, 1, 2, 0, 0, 0, 4, 5, 0, 1, 2, 3, 0, 2, 2, 0, 4,
					5, 0, 0, 3, 0, 1, 2, 3, 5, 6, 0, 1, 0, 1, 0, 1, 23, 24, 2, 23,
					24, 2, 23, 26, 1, 8, 0, 24, 25, 28, 2, 8, 14, 15, 18, 20, 22, 23,
					29, 30, 31, 8, 9, 13, 14, 15, 18, 19, 20, 22, 23, 26, 27, 28, 29, 30,
					31, 32])
	# Edges are directional in DGL, so we have to make them double-directional, from the src to dst and dst to src
	u = np.concatenate([src, dst])
	v = np.concatenate([dst, src])
	# construct a dgl graph
	return dgl.DGLGraph((u,v))

# we build a Convolution GCN
class ConvGCN(nn.Module):
	def __init__(self, in_feats, hidden_size, num_classes):
		super(ConvGCN, self).__init__()
		self.conv1 = GraphConv(in_feats, hidden_size)
		self.conv2 = GraphConv(hidden_size,out_feats = num_classes)
	def forward(self, g, inputs):
		h = self.conv1(g, inputs)
		h = torch.relu(h)
		h = self.conv2(g, h)
		return h

if __name__ == "__main__":
	model = build_karate_club_graph()
	# so how to transfer it into  visualize
	nx_model = model.to_networkx().to_directed()
	position = nx.kamada_kawai_layout(nx_model)
	#nx.draw(nx_model, position, with_labels = True)
	#plt.show()
	"""
	fore work is to construct a graph, including nodes and edges
	"""
	embed = nn.Embedding(34, 5) # every node has the feature of (5,0) tensor
	"""
	Data preparation and initialization
	"""
	net = ConvGCN(5, 5, 2)
	inputs = embed.weight
	print("the inputs initial is :", inputs)
	labeled_nodes = torch.tensor([0, 33])
	labels = torch.tensor([0, 1])
	"""Train then visualize"""
	optimizer = torch.optim.Adam(itertools.chain(net.parameters(), embed.parameters()), lr = 0.01)
	# use the list to record those
	all_logits = []
	for epoch in range(50):
		logits = net(model, inputs)
		all_logits.append(logits.detach())
		logp = F.log_softmax(logits, 1)
		# we only compute the loss for labeled nodes
		loss = F.nll_loss(logp[labeled_nodes], labels)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		print("Epoch||", loss.item())



