import time
import torch
import random
import numpy as np
from torch.nn import Parameter
from signedconvolutionlayer import signedconvolutioninit, signedconvolutiondeep, ListModule
import torch.nn.functional as F
import torch.nn.init as init

class SGCN(torch.nn.Module):
	"""
	For details see: signed graph convolutional network.
	"""
	def __init__(self, device, args, node_features):
		"""
		:param device: initialization
		:param args: arguments object
		:param X: node features
		"""
		super(SGCN, self).__init__()
		self.args = args
		torch.manual_seed(self.args.seed)
		self.device = device
		self.node_features = node_features
		self.setup_layers()
	def setup_layers(self):
		"""
		Adding first layers, deep signed layers
		Assgining regression parameters if the model is not a single layer model.
		"""
		# self.nodes is the list of actors
		self.nodes = range(self.node_features.shape[0])
		# self.layers are the integration of layers(input_features, out_Features) -- dimensions
		self.layers = self.args.layers
		# self.num_layers are the number of layers
		self.num_layers = len(self.layers)
		# we should write the first layer of networks
		self.positive_firstlayer_aggregator = signedconvolutioninit(in_features = 2*self.node_features.size(1),
																	out_features = self.layers[0]).to(self.device)
		self.negative_firstlayer_aggregator = signedconvolutioninit(in_features = 2*self.node_features.size(1),
																	out_features = self.layers[0]).to(self.device)
		# construct collections of aggregators for balanced links and negative links.
		self.positive_aggregators = []
		self.negative_aggregators = []
		for i in range(1, self.num_layers):
			self.positive_aggregators.append(signedconvolutiondeep(in_features = 3*self.layers[i-1],
																   out_features = self.layers[i]).to(self.device))
			self.negative_aggregators.append(signedconvolutiondeep(in_features = 3*self.layers[i-1],
																   out_features = self.layers[i]).to(self.device))
		self.positive_aggregators = ListModule(*self.positive_aggregators)
		self.negative_aggregators = ListModule(*self.negative_aggregators)
		# we have that the regression weight matrix is 4 * layers[-1], [zi,zj]--(zi=[hi,hj]))
		self.final_regression_weight = Parameter(torch.FloatTensor(4*self.layers[-1], 2))
		init.xavier_uniform_(self.final_regression_weight)

	def calculate_positive_loss(self, z, positive_edges):
		"""
		:param z: the hidden vertex features
		:param positive_edges: this dimension is 2 * edge_number
		:return:
		"""
		self.positive_samples = [random.choice(self.nodes) for _ in range(positive_edges.shape[1])]
		self.positive_samples = torch.from_numpy(np.array(self.positive_samples, dtype = np.int64).T)
		self.positive_samples = self.positive_samples.type(torch.long).to(self.device)
		positive_edges = torch.t(positive_edges)
		self.positive_z_i = z[positive_edges[:,0],:]
		self.positive_z_j = z[positive_edges[:,1],:]
		self.positive_z_k = z[self.positive_samples,:]
		norm_i_j = torch.norm(self.positive_z_i - self.positive_z_j, p = 2, dim = 1, keepdim = True).pow(2)
		norm_i_k = torch.norm(self.positive_z_j - self.positive_z_k, p = 2, dim = 1, keepdim = True).pow(2)
		term = norm_i_j - norm_i_k
		term[term < 0] = 0
		total_loss = torch.mean(term)
		return total_loss

	def calculate_negative_loss(self, z, negative_edges):
		# negative_samples is the list of samples in negative edges
		self.negative_samples = [random.choice(self.nodes) for node in range(negative_edges.shape[1])]
		self.negative_samples = torch.from_numpy(np.array(self.negative_samples, dtype = np.int64).T)
		self.negative_samples = self.negative_samples.type(torch.long).to(self.device)
		negative_edges = torch.t(negative_edges)
		self.negative_z_i = z[negative_edges[:,0],:]
		self.negative_z_j = z[negative_edges[:,1],:]
		self.negative_z_k = z[self.negative_samples,:]
		norm_i_j = torch.norm(self.negative_z_i - self.negative_z_j, p = 2, dim = 1, keepdim = True).pow(2)
		norm_i_k = torch.norm(self.negative_z_j - self.negative_z_k, p = 2, dim = 1, keepdim = True).pow(2)
		term = norm_i_j - norm_i_k
		term[term < 0] = 0
		total_loss = torch.mean(term)
		return total_loss

	def calculate_regression_loss(self, target):
		zi = torch.cat((self.positive_z_i, self.positive_z_j), dim = 1)
		zj = torch.cat((self.negative_z_i, self.negative_z_j), dim = 1)
		features = torch.cat((zi, zj), dim = 0)
		predictions = torch.mm(features, self.final_regression_weight)
		predictions_soft = F.log_softmax(predictions, dim = 1)
		loss_term = F.nll_loss(predictions_soft, target)
		return loss_term, predictions_soft

	def compute_loss(self, z, target, positive_edges, negative_edges):

		loss_term1 = self.calculate_positive_loss(z, positive_edges)
		loss_term2 = self.calculate_negative_loss(z, negative_edges)
		regress_loss, predictions = self.calculate_regression_loss(target)
		loss_term = regress_loss + self.args.lamb * (loss_term1 + loss_term2)
		return loss_term

	def forward(self, positive_edges, negative_edges, target):
		"""
		:param positive_edges: positive edges
		:param negative_edges: negative edges
		:param target: target vectors
		:return:
		"""
		self.h_pos, self.h_neg = [], []
		self.h_pos.append(torch.tanh(self.positive_firstlayer_aggregator(self.node_features, positive_edges)))
		self.h_neg.append(torch.tanh(self.negative_firstlayer_aggregator(self.node_features, negative_edges)))
		# we here run the neural layers from the first layer to the last one
		# and we have to do a regression operation to keep the dim to be 3, in order to make the classification
		for i in range(1, self.num_layers):
			self.h_pos.append(torch.tanh(self.positive_aggregators[i-1](self.h_pos[i-1],
																		self.h_neg[i-1],
																		positive_edges,
																		negative_edges)))
			self.h_neg.append(torch.tanh(self.negative_aggregators[i-1](self.h_neg[i-1],
																		self.h_pos[i-1],
																		positive_edges,
																		negative_edges)))
		# select the last layer, here is the zi for every node, and we can see the dim of z
		self.z = torch.cat((self.h_pos[-1], self.h_neg[-1]), dim = 1)
		print("dim of the z matrix:\t", self.z.shape)
		loss = self.compute_loss(self.z, target, positive_edges = positive_edges, negative_edges = negative_edges)
		return loss, self.z





