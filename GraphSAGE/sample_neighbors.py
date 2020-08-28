import torch
import numpy as np
import networkx as nx
import random

class construct_graph():
	def __init__(self, g):
		self.nodes = g.nodes()
		self.edges = g.edges()
		self.node_number = len(self.nodes)
		self.dimension = len(self.nodes)
	def adjency_matrix(self):
		self.adj_matrix = torch.zeros((self.dimension, self.dimension))
		for index_0, index_1 in self.edges:
			self.adj_matrix[index_0, index_1] = 1
			self.adj_matrix[index_1, index_0] = 1

		return self.adj_matrix
	# this function is to aggregate the neighbors into a list
	def index_adjmatrix(self):
		self.index_neighbor = torch.nonzero(self.adj_matrix, as_tuple= False)
		to_neighbors = []
		for nodeid in self.nodes:
			temp = []
			for item in self.index_neighbor:
				if item[0] == nodeid:
					temp.append(item[1].detach().item())
			to_neighbors.append(set(temp))
		return to_neighbors


if __name__ == "__main__":
	data = nx.petersen_graph()
	model = construct_graph(data)
	adj = model.adjency_matrix()
	index = model.index_adjmatrix()
	print(adj)
