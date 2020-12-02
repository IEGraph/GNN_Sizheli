import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.init import uniform_
from torch_scatter import scatter_add, scatter_mean
from torch_geometric.utils import remove_self_loops, add_self_loops
import math

def init_uniform(size, tensor):
	"""
	:param size: size of the tensor
	:param tensor: tensor initialized
	:return:
	"""
	stdv = 1.0 / math.sqrt(size)
	if tensor is not None:
		result = uniform_(tensor = tensor, a = - stdv, b = stdv)
	else:
		return tensor
	return result

class ListModule(torch.nn.Module):
	def __init__(self, *args):
		"""Model initializing."""
		super(ListModule, self).__init__()
		idx = 0
		for module in args:
			self.add_module(str(idx), module)
			idx += 1
	def __getitem__(self, idx):
		it = iter(self._modules.values())
		for i in range(idx):
			next(it)
		return next(it)
	def __iter__(self):
		"""Iterating on the layers."""
		return iter(self._modules.values())
	def __len__(self):
		"""number of layers."""
		return len(self._modules)

class signedconvolution(torch.nn.Module):
	def __init__(self,
				 in_features,
				 out_features,
				 norm = True,
				 norm_embed = True,
				 bias = True):
		"""
		:param in_features: in put feature number
		:param out_features: out put feature number
		:param norm: boolean
		:param norm_embed: boolean
		:param bias: boolean
		"""
		super(signedconvolution, self).__init__()

		self.in_features = in_features
		self.out_features = out_features
		self.norm = norm
		self.norm_embed = norm_embed
		self.weight = Parameter(torch.FloatTensor(in_features, out_features))
		# check do we need bias
		if bias:
			self.bias = Parameter(torch.FloatTensor(1, out_features))
		else:
			self.bias = None

	def customize_parameters(self):
		# size is tbe number of nodes
		size = self.weight.size(0)
		self.weight = init_uniform(tensor = self.weight, size = size)
		self.bias = init_uniform(tensor = self.weight, size = size)

	def __repr__(self):
		return "{},{},{}".format(self.__class__.__name__, self.in_features, self.out_features)

class signedconvolutioninit(signedconvolution):
	"""
	This is for the first year
	"""
	def forward(self, node_features, edge_index):
		"""
		:param node_features: is the matrix of nodes
		:param edge_index: is the edge index
		:return:
		"""
		# pay attention to the shape of the edge_index input (2, x)
		edge_index, _ = remove_self_loops(edge_index, None)
		# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/loop.html#remove_self_loops

		row_index, col_index = edge_index
		if self.norm:
			out = scatter_mean(node_features[col_index], row_index, dim = 0, dim_size = node_features.size(0))

		else:
			out = scatter_add(node_features[col_index], row_index, dim = 0, dim_size = node_features.size(0))

		# we concat the out and originan x on the dim 1
		output = torch.cat((out, node_features), dim = 1)
		output = output.mm(self.weight)
		if self.bias is not None:
			output = output + self.bias
		if self.norm_embed:
			output = F.normalize(output, p = 2, dim = -1)
		return output
		# this output is prepared for the positive and negative
		# so this is Node*Feature matrix, they are identical dimensional matrix

class signedconvolutiondeep(signedconvolution):
	def forward(self, nodefeatures_1, nodefeatures_2, edge_index_pos, edge_index_neg):
		"""
		:param nodefeatures_1:
		:param nodefeatures_2:
		in the first block, the nodefeatures_1 is the balanced feature matrix, the nodefeatures_2 is the unbalanced
		feature matrix.
		in the first block, the nodefeatures_1 is the unbalanced feature matrix, the nodefeatures_2 is the balanced
		feature matrix.
		:param edge_index_pos: it is the positive edge index in the before layer
		:param edge_index_neg: it is the negative edge index in the before layer
		:return:
		"""
		edge_index_pos, _ = remove_self_loops(edge_index_pos, None)
		# whether we should add some self loops to the node
		edge_index_pos, _ = add_self_loops(edge_index_pos, None)
		edge_index_neg, _ = remove_self_loops(edge_index_neg, None)
		# whether we should add some self loops to the node
		edge_index_neg, _ = add_self_loops(edge_index_neg, None)
		# edge_index_pos is the neighbors of actors'
		row_pos, col_pos = edge_index_pos
		row_neg, col_neg = edge_index_neg

		if self.norm:
			out_1 = scatter_mean(nodefeatures_1[col_pos], row_pos, dim = 0, dim_size = nodefeatures_1.size(0))
			out_2 = scatter_mean(nodefeatures_2[col_neg], row_neg, dim = 0, dim_size = nodefeatures_2.size(0))
		else:
			out_1 = scatter_add(nodefeatures_1[col_pos], row_pos, dim = 0, dim_size = nodefeatures_1.size(0))
			out_2 = scatter_add(nodefeatures_2[col_neg], row_neg, dim = 0, dim_size = nodefeatures_2.size(0))
		# now we have the dimension of the input features of node_feature * 3.
		out = torch.cat((out_1, out_2, nodefeatures_1), dim = 1)
		out = out.mm(self.weight)

		if self.bias is not None:
			out = out + self.bias
		if self.norm_embed:
			out = F.normalize(out, p = 2, dim = -1)
		return out






