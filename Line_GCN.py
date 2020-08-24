import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import dgl
from dgl import DGLGraph
from dgl.data import citation_graph
import networkx as nx
import matplotlib.pylab as plt

def aggregate_radius(radius, g, z):
	# initializing list to collect message passing result
	z_list = []
	g.ndata["z"] = z
	# pulling message from 1-hop neighborhood
	g.update_all(message_func = fn.copy_src(src = 'z', out = "m"), reduce_func = fn.sum(msg = "m", out = "z"))
	# count 1
	z_list.append(g.ndata["z"])
	for i in range(radius - 1):
		for j in range(2**i):
			# pull message from 2^j neighborhood
			g.update_all(message_func = fn.copy_src(src = 'z', out = "m"), reduce_func = fn.sum(msg = "m", out = "z"))
		z_list.append(g.ndata["z"])
	return z_list

# implement LGNN in DGL, the LGNNCore is one layer iterable
class LGNNCore(nn.Module):
	def __init__(self, in_feats, out_feats, radius):
		super(LGNNCore, self).__init__()
		self.out_feats = out_feats
		self.radius = radius

		self.linear_prev = nn.Linear(in_feats, out_feats)
		self.linear_deg = nn.Linear(in_feats, out_feats)
		self.linear_radius = nn.ModuleList([nn.Linear(in_feats, out_feats) for i in range(radius)])
		self.linear_fuse = nn.Linear(in_feats, out_feats)
		self.bn = nn.BatchNorm1d(out_feats)

	def forward(self, g, feat_a, feat_b, deg, pm_pd):
		# term project of previous
		previous_projection = self.linear_prev(feat_a)
		degree_projection = self.linear_deg(deg * feat_a)
		# term redius
		hop2j_list = aggregate_radius(self.radius, g, feat_a)
		# apply linear transformation
		hop2j_list = [linear(x) for linear, x in zip(self.linear_radius, hop2j_list)]
		# we hope to sum all the items in hop2j_list
		radius_proj = sum(hop2j_list)

		# term fuse
		fuse = self.linear_fuse(pm_pd @ feat_b)
		# sum them together
		result = previous_projection + radius_proj + fuse

		# skip connection and batch norm
		n = self.out_feats//2
		result = torch.cat([result[:,:n], F.relu(result[:,n:])], 1)
		result = self.bn(result)

		return  result

class LGNNLayer(nn.Module):
	def __init__(self, in_feats, out_feats, radius):
		super(LGNNLayer, self).__init__()
		self.g_layer = LGNNCore(in_feats, out_feats, radius)
		self.lg_layer = LGNNCore(in_feats, out_feats, radius)
	def forward(self, g, lg, x, lg_x, deg_g, deg_lg, pm_pd):
		next_x = self.g_layer(g, x, lg_x, deg_g, pm_pd)
		pm_pd_y = torch.transpose(pm_pd, 0, 1)
		next_lg_x = self.lg_layer(lg, lg_x, deg_lg, pm_pd)

		return next_x, next_lg_x
if __name__ == "__main__":
	data = citation_graph.load_cora()
	# build the picture
	G = DGLGraph(data.graph)
	# load the label
	labels = torch.tensor(data.labels)
	# find nodes equal 0
	label0_nodes = torch.nonzero(labels == 0, as_tuple = False).squeeze()
	# find all src nodes pointing to
	src, _ = G.in_edges(label0_nodes)
	print(labels.size())
	# src labels
	src_labels = labels[src]
	# find edges whose both endpoints are in class 0
	intra_src = torch.nonzero(src_labels == 0, as_tuple = False).squeeze()
	train_set = dgl.data.CoraBinary()
	G1, matrix1, label1 = train_set[1]
	"""
	nx.draw(G1.to_networkx(), node_size = 50, node_color = label1, arrows = False, with_labels = False)
	plt.show()
	"""

