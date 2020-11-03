import matplotlib.pylab as plt
import networkx as nx
import torch
import dgl
import dgl.function as fn

# set the numnber of nodes
N = 100
# set the damping factor
DAMP = 0.85
# number of interations
K = 10
g = nx.nx.erdos_renyi_graph(N, 0.1)
g = dgl.DGLGraph(g)

g.ndata["pv"] = torch.ones(N) / N
g.ndata["deg"] = g.nodes().float()



def pagerank_message_func(edges):

	return {"pv":edges.src["pv"] / edges.src["deg"]}

def pagerank_reduce_func(nodes):
	msgs = torch.sum(nodes.mailbox["pv"], dim = 1)
	pv = (1 - DAMP) / N + DAMP * msgs
	#print("the current pv is:", pv)
	return {"pv":pv}

"""
g.register_message_func(pagerank_message_func)
g.register_reduce_func(pagerank_reduce_func)
"""

def pagerank_naive(g):
	# Phase #1: send out messages along all edges.
	for u, v in zip(*g.edges()):
		g.send((u, v))

	# Phase #2: receive messages to compute new PageRank values.
	for v in g.nodes():
		g.recv(v)

def pagerand_batch(g):
	g.send(g.edges())
	g.recv(g.nodes())

def pagerank_level2(g):
	g.update_all()

def pagerank_builtin(g):
	g.ndata['pv'] = g.ndata['pv'] / g.ndata['deg']
	g.update_all(message_func=fn.copy_src(src='pv', out='out'),
				reduce_func=fn.sum(msg='out',out='m_sum'))
	g.ndata['pv'] = (1 - DAMP) / N + DAMP * g.ndata['m_sum']

print(g.ndata["pv"])
pagerank_builtin(g)
print(g.ndata["pv"])
