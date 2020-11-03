import pandas as pd
import numpy as np
import torch
import scipy.sparse as sp

class data_loader():
	def __init__(self, file_path):
		# load data
		columns = ['user_id', 'item_id', 'ratings', 'ts']
		ratings_df = pd.read_csv(file_path, sep = '\t', names = columns, engine = 'python')
		ratings_df['user_id'] = ratings_df['user_id']-1
		ratings_df['item_id'] = ratings_df['item_id']-1
		#ratings_df['ratings'] = ratings_df['ratings']
		print("the range of userID is [{},{}]".format(ratings_df.user_id.min(), ratings_df.user_id.max()))
		print("the range of itemID is [{},{}]".format(ratings_df.item_id.min(), ratings_df.item_id.max()))

		self.user_num = ratings_df.user_id.max()+1
		self.item_num = ratings_df.item_id.max()+1
		rating_levels = set(list(ratings_df['ratings']))
		print("level display\t",rating_levels)
		level_range = [-1, 1]
		level_nums = len(level_range) * 1.0
		# we make it the average of rating levels
		maxv = max(rating_levels)
		minv = min(rating_levels)
		avev_1 = (maxv + minv)/2.0
		# we construct a list to arrange -1, 0, 1
		rating_level_adj_list = []
		user_list_neg = list(ratings_df[ratings_df['ratings'] < avev_1]['user_id'])
		item_list_neg = list(ratings_df[ratings_df['ratings'] < avev_1]['item_id'])
		rating_level_adj_list.append([user_list_neg] + [item_list_neg])
		"""
		user_list_neu = list(ratings_df[ratings_df['ratings'] > avev_1 and ratings_df['ratings'] < avev_2]['user_id'])
		item_list_neu = list(ratings_df[ratings_df['ratings'] > avev_1 and ratings_df['ratings'] < avev_2]['item_id'])
		rating_level_adj_list.append([user_list_neu] + [item_list_neu])
		"""
		user_list_pos = list(ratings_df[ratings_df['ratings'] > avev_1]['user_id'])
		item_list_pos = list(ratings_df[ratings_df['ratings'] > avev_1]['item_id'])
		rating_level_adj_list.append([user_list_pos] + [user_list_pos])
		self.rating_signed_list = rating_level_adj_list
		#print(rating_level_adj_list[0][1])
		# the level is set to be [-1, 0 ,1]

		print("check the length of the first element of the level list:\t",len(rating_level_adj_list[0]))

	def _get_negative_positive_list(self):
		negative_links = np.array(self.rating_signed_list[0]).transpose()
		negative_signs = -1 * np.ones(len(self.rating_signed_list[0][0]))
		positive_links = np.array(self.rating_signed_list[1]).transpose()
		positive_signs = 1.0 * np.ones(len(self.rating_signed_list[1][0]))
		return negative_links, negative_signs, positive_links, positive_signs

	def _get_extra_links(self):
		def adj_matrix(user_list, item_list, sign):
			adj_mat = sp.dok_matrix((self.user_num, self.item_num), dtype = np.float32)
			adj_mat[user_list, item_list] = sign
			B_adj = adj_mat.tocsc()
			return B_adj

		signs = [-1.0, 1.0]
		integrated_B = []
		for i, sign in enumerate(signs):
			integrated_B.append(adj_matrix(self.rating_signed_list[i][0], self.rating_signed_list[i][1], sign))

		B = sum(integrated_B)
		S = (B.dot(B.T)).dot(B).todok()
		B = B.todok()
		assumed_links = []
		pos_links, neg_links = [], []
		for (b, s), val in S.items():
			if (b, s) not in B:
				if val > 0:
					pos_links.append(((b, s), val))
					assumed_links.append('{}\t{}\t{}'.format(b, s, 1))
				else:
					neg_links.append(((b, s), val))
					assumed_links.append('{}\t{}\t{}'.format(b, s, -1))
		print("the total nonzero:\t", len(S.items()))
		print("the total number not in B:\t", len(assumed_links))

		with open("./extra_links_from_ml100k", mode = 'w') as f:
			f.write('\n'.join(assumed_links))
		neg_links.sort()
		with open("./extra_neg_from_ml100k", mode = 'w') as f:
			lines = ['{}\t{}\t{}'.format(b, s, -1) for (b, s), val in neg_links]
			f.write('\n'.join(lines))
		pos_links.sort(reverse = True)
		with open("./extra_pos_from_ml100k", mode = 'w') as f:
			lines = ['{}\t{}\t{}'.format(b, s, -1) for (b, s), val in pos_links]
			f.write("\n".join(lines))

if __name__ == "__main__":
	file_path = './ml-100k/raw/u1.base'
	data = data_loader(file_path)
	negative_links, negative_signs, positive_links, positive_signs = data._get_negative_positive_list()
	print("the number of negative links: \t", negative_links.shape[0])
	print("the number of positive links: \t", positive_links.shape[0])
	data._get_extra_links()


