import torch
import numpy as np
import networkx as nx
import random
from dgl.data import citation_graph as cite
import networkx as nx

class construct_graph():
	def __init__(self, g):
		self.nodes = g.nodes()
		self.edges = g.edges()
		self.node_number = len(self.nodes)
		self.dimension = len(self.nodes)
	def adjent_matrix(self):
		self.adj_matrix = torch.zeros((self.dimension, self.dimension))
		for index_0, index_1 in self.edges:
			self.adj_matrix[index_0, index_1] = 1
			self.adj_matrix[index_1, index_0] = 1
		self.adj_matrix = self.adj_matrix + torch.eye(self.dimension)
	def degree_matrix(self):
		degree_list = self.adj_matrix.sum(dim = 1)
		degree_list = degree_list.pow(-0.5)
		self.D_matrix = torch.diag(degree_list)
	def renormalized_adj(self):
		self.adj_matrix = self.D_matrix.mm(self.adj_matrix)
		self.adj_matrix = self.adj_matrix.mm(self.D_matrix)
		return self.adj_matrix

class TransformationMatrix():
	def __init__(self, g):
		self.nodes = g.nodes()
		self.edges = g.edges()
		self.node_number = len(self.nodes)
	def LaplacianMatrix(self):
		self.adj_matrix = torch.zeros((self.node_number, self.node_number))
		for index_0, index_1 in self.edges:
			self.adj_matrix[index_0, index_1] = 1
			self.adj_matrix[index_1, index_0] = 1
		degree_list = self.adj_matrix.sum(dim = 1)
		self.D_matrix = torch.diag(degree_list)
		self.Laplacianmatrix = self.D_matrix - self.adj_matrix
		#return self.Laplacianmatrix
	def TransformMatrix(self, alpha = 0.1):
		eigenvalues, eigenvectors = torch.eig(self.Laplacianmatrix, eigenvectors = False)
		# access to the eigenvalues
		self.eigvaluematrix = torch.diag(eigenvalues.t()[0])
		self.transformMatrix = self.eigvaluematrix * alpha + self.Laplacianmatrix
		inv = torch.inverse(self.transformMatrix)
		inv_RowSum = inv.sum(dim = 1, keepdim = True)
		inv = inv.div(inv_RowSum)
		return inv



if __name__ == "__main__":
	data = nx.petersen_graph()
	model = TransformationMatrix(data)
	# product the adj_matrix
	model.LaplacianMatrix()
	evals = model.TransformMatrix()
	print(model.transformMatrix)

