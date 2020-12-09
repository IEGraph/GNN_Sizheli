import sys
import os
import numpy as np
import scipy.sparse as sps

from collections import Counter

"""
training_datafile = sys.argv[1]
testing_datafile = sys.argv[2]
validation_datafile = sys.argv[3]
topk_neg_to_use = int(sys.argv[4])
topk_pos_to_use = int(sys.argv[5])
"""

def data_write(path):
	with open(file = path, mode = 'r') as f:
		num_b, num_s, _ = [int(value) for value in f.readline().split('\t')]
		print("num buyers and sellers:", num_b, num_s)
		links = []
		signs = []
		for l in f.readlines():
			row = [int(val) for val in l.split('\t')]
			links.append((row[0], row[1]))
			signs.append(row[2])

	num_d_pos = signs.count(1)
	num_d_neg = signs.count(-1)
	print("the number of positive links:\t", num_d_pos)
	print("the bumber of negative links:\t", num_d_neg)

	# construct the bipartite network matrix
	B = sps.dok_matrix((num_b, num_s))
	for (b, s), r in zip(links, signs):
		B[b, s] = np.float(r)
	print("transaction number:\t", len(B.items()))
	B = B.tocsc()
	print("the shape of B:\t", B.shape)

	S = (B.dot(B.transpose())).dot(B).todok()
	B = B.todok()
	assumed_links = []
	pos_with_link, neg_with_link = [], []

	for (b, s), val in S.items():
		# here we mask existing links between buyers and sellers.
		if (b, s) not in B:
			if val > 0:
				pos_with_link.append((val, (b, s)))
				assumed_links.append('{}\t{}\t{}'.format(b, s, 1))
			else:
				neg_with_link.append((val, (b, s)))
				assumed_links.append('{}\t{}\t{}'.format(b, s, -1))

	print("total nonzero:\t", len(S.keys()))
	print("total not in B:\t", len(assumed_links))
	#print("assumed_links:\t", assumed_links[:20])

	with open('./extra_links_from_B_balance_theory.txt', 'w') as f:
		f.write('\n'.join(assumed_links))

	neg_with_link.sort()
	with open('./extra_neg_links_sorted_from_B_balance_theory.txt', 'w') as f:
		lines = ['{}\t{}\t{}'.format(b, s, -1) for val, (b, s) in neg_with_link]
		f.write('\n'.join(lines))

	pos_with_link.sort(reverse = True)
	with open('./extra_pos_links_sorted_from_B_balance_theory.txt', 'w') as f:
		lines = ['{}\t{}\t{}'.format(b, s, 1) for val, (b, s) in pos_with_link]
		f.write('\n'.join(lines))


if __name__ == "__main__":
	files = os.listdir()
	data_path = os.getcwd()
	for file in files:
		if file[:4] == "bona":
			data_path = file
	print(data_path)
	data_write(data_path)