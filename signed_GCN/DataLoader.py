
import numpy as np
from arguments import parameter_parser
import pandas as pd
import scipy.sparse as sps
from sklearn.decomposition import TruncatedSVD
from scipy.sparse.linalg import svds

def read_file(args):
	dataset = pd.read_csv(args.edge_path).values.tolist()
	edges = {}
	edges["positive_edges"] = [edge[0:2] for edge in dataset if edge[2] == 1]
	edges["negative_edges"] = [edge[0:2] for edge in dataset if edge[2] == -1]
	# compute the number of existing edges and all nodes (from 0)
	edges['edge_num'] = len(dataset)
	edges["node_num"] = len(set([edge[0] for edge in dataset]+[edge[1] for edge in dataset]))
	return edges

def create_spectral_features(args, positive_edges, negative_edges, node_num):
	positive_edges = positive_edges + [[edge[1],edge[0]] for edge in positive_edges]
	negative_edges = negative_edges + [[edge[1],edge[0]] for edge in negative_edges]
	total_edges = positive_edges + negative_edges
	#
	source = [edge[0] for edge in total_edges]
	destination = [edge[1] for edge in total_edges]
	value_list = [1] * len(positive_edges) + [-1] * len(negative_edges)
	signed_ = sps.coo_matrix((value_list, (source, destination)), shape = (node_num, node_num),
							 dtype = np.float32).tocsr()
	print("the shape of signed_\t", signed_.shape)
	"""
	U, sigma, V = svds(signed_, k = 10)
	print("the shape of U:\t", U.shape)
	print("the shape of sigma:\t", sigma.shape)
	print("the shape of V:\t", V.shape)
	"""
	svd = TruncatedSVD(n_components = args.reduction_dimensions,
					   n_iter = args.reduction_iterations,
					   random_state = args.seed)
	svd.fit(signed_)
	X = svd.components_.T
	print("the shape of X.T:\t",X.shape)
	signed_prediction = svd.transform(signed_)
	print("the shape of transformed signed:\t", signed_prediction.shape)
	return X

def setup_feature(args, positive_edges, negative_edges, node_num):
	if args.spectral_features:
		X = create_spectral_features(args, positive_edges, negative_edges, node_num)
	else:
		X = np.random.random((node_num, args.reduction_dimensions))
	return X

if __name__ == "__main__":
	args = parameter_parser()
	edges = read_file(args)
	print("the edge_num", edges["edge_num"])
	print("the node_num", edges["node_num"])
	X = setup_feature(args, positive_edges = edges["positive_edges"],
							 negative_edges = edges["negative_edges"],
							 node_num = edges["node_num"])
	print(X.shape)

