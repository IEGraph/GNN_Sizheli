import torch
import dgl
import matplotlib.pylab as plt
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl import DGLGraph
from dgl.data import citation_graph as citegrh
import networkx as nx


# we then proceed to define the GCN module. A GCNlayer essentially performs message passing all the nodes then applies
# a fully connected layer
class GCNLayer(nn.Module):
	def __init__(self, in_feats, out_feats):
		super(GCNLayer, self).__init__()
		self.linear = nn.Linear(in_feats, out_feats)
	def forward(self, graph, feature):
		with graph.local_scope():
			graph.ndata["h"] = feature
			graph.update_all(gcn_msg, gcn_reduce)
			h = graph.ndata["h"]
			return self.linear(h)

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.layer1 = GCNLayer(1433, 16)
		self.layer2 = GCNLayer(16, 7)
	def forward(self, g, features):
		h = self.layer1(g, features)
		h = self.layer2(g, h)
		return h

def load_cora_data():
	data = citegrh.load_cora()
	features = torch.FloatTensor(data.features)
	labels = torch.LongTensor(data.labels)
	train_mask = torch.BoolTensor(data.train_mask)
	test_mask = torch.BoolTensor(data.test_mask)
	g = DGLGraph(data.graph)
	return g, features, labels, train_mask, test_mask

def evaluate(model, g, features, labels, mask):
	model.eval()
	with torch.no_grad():
		logits = model(g, features)
		logits = logits[mask]
		labels = labels[mask]
		_, indices = torch.max(logits, dim = 1)
		correct = torch.sum(indices == labels)
		return correct.item() * 1.0 / len(labels)

if __name__ == "__main__":
	# Since the aggragation on a node u only involves summing the neighbors' representations h
	gcn_msg = fn.copy_src(src = 'h', out = 'm')
	gcn_reduce = fn.sum(msg = "m", out = "h")
	net = Net()
	g, features, labels, train_mask, test_mask = load_cora_data()
	optimizer = torch.optim.Adam(net.parameters(), lr = 0.01)
	for epoch in range(50):
		net.train()
		logits = net(g, features)
		loss = nn.CrossEntropyLoss()
		output = loss(logits[train_mask], labels[train_mask])
		optimizer.zero_grad()
		output.backward()
		optimizer.step()
		acc = evaluate(net, g, features, labels, test_mask)
		print("accurate:",acc)


