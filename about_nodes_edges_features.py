import dgl
import networkx as nx
import matplotlib.pylab as plt
import torch as th
import numpy as np
import scipy.sparse as spp

g_nx = nx.petersen_graph()
g_dgl = dgl.DGLGraph(g_nx)

"""
nx.draw(g_nx, with_labels = True)
nx.draw(g_dgl.to_networkx(), with_labels = True)
plt.show()
"""
# Create a star graph from a pair of arrays
u = th.tensor([0,0,0,0,0])
v = th.tensor([1,2,3,4,5])
star1 = dgl.DGLGraph((u, v))
star2 = dgl.DGLGraph((0, v))

adj = spp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))

star3 = dgl.DGLGraph(adj)

# set a g
g = dgl.DGLGraph()
g.add_nodes(10)

src = th.tensor(list(range(1,10)))
g.add_edges(src, 0)

# assigning a feature
features = th.randn(10,3)
g.ndata["x"] = features
g.edata["w"] = th.randn(9,2)
g.add_edge(1, 0)
g.edges[1].data["w"] = th.zeros(1,2)
print(g.edges())
print(g.edata["w"])
