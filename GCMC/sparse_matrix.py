import numpy as np
import scipy.sparse as ss
import random
import torch
import torch.nn.functional as F

def spar_matrix(users, items, ratings):
	# random produce rows, columns
	rows_1 = np.arange(0, users)
	columns_1 = np.arange(0, items)
	#values = np.random.randint(low = 0, high = 5, size = (ratings))

	# use coo_matrix yield sparse matrix
	sparseMatrix = ss.dok_matrix((users, items), dtype = np.float32).tolil()
	row_index = np.random.randint(low = 0, high = np.max(rows_1) + 1, size = (ratings,))
	columns_index = np.random.randint(low = 0, high = np.max(columns_1) + 1, size = (ratings,))
	for i in range(ratings):
		sparseMatrix[row_index[i], columns_index[i]] = 1
		print(sparseMatrix.toarray())
	# now the spar is a mapping function from location to the corresponding value
	# transform the mapping into the full matrix
	fullMatrix = ss.dok_matrix((users + items, users + items), dtype = np.float32).tolil()
	fullMatrix[:users, users:] = sparseMatrix
	fullMatrix[users:, :users] = sparseMatrix.T
	return fullMatrix

def ptSparseMat():
	#target = torch.ones([10, 64], dtype = torch.float32)
	target = torch.randint(low = 0, high = 2, size = (10 , 64), dtype = torch.float32)
	output = torch.randn([10, 64], dtype = torch.float32)
	pos_weight = torch.ones([64])
	result = torch.sigmoid(output)
	result = F.binary_cross_entropy(result, target, reduction = 'sum')
	print("the output:\n", output)
	print("the BCE loss:\n",result)

if __name__ == "__main__":
	full_sparseM = spar_matrix(20, 10, 30)
	print(full_sparseM.toarray())
