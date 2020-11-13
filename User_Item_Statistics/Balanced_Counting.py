import sys
import os
# load the file of type
# N M
# i j s

# Ep is set of positive edges of the form (i, x)
# En is set of negative edges of the form (i, x)
# only the (buyer, seller) are in this set since we are undirected anyway.

# Nbp[i] = set of positive neighbors of buyer i
# Nbn[i] = set of negative neighbors of buyer i
# Nsp[i] = set of positive neighbors of seller x
# Nsn[x] = set of negative neighbors of seller x
# there are 7 types of "positive and negative"
mapper = {1:'++++', 2:'----', 3:'++--', 4:'+-+-', 5:'+--+', 6:'+---', 7:'+++-'}
"""
#R results e.g.,
# b1 = 1 [++++]                                                                                                         = repeated 4x
# b2 = 2 [----]                                                                                                         = repeated 4x
# b3 = 3 [++--] same as [--++]               = buyer 2pos and buyer 2 neg, sellers each have 1 pos and 1 neg            = repeated 2x
# b4 = 4 [+-+-] same as [-+-+]               = both buyer and sellers have 1 pos and 1 neg                              = repeated 2x 
# b5 = 5 [+--+] same as [-++-]               = seller 2 pos and seller 2 neg, buyers each have 1 pos and 1 neg          = repeated 2x
# b6 = 6 [+---] same as all single pos 3 neg = one buyer has +- and one buyer --, one seller has +- and one seller --   = only 1x
# b7 = 7 [+++-] same as all single neg 3 pos = one buyer has +- and one buyer ++, one seller has +- and one seller ++   = only 1x
"""

def butterfly_computer(path, path_add):
	R = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}
	with open(path, mode = 'r') as f:
		first = f.readline()
		user_list = []
		item_list = []
		signed_list = []
		for l in f.readlines():
			row = [int(val) for val in l.strip().split('\t')]
			user_list.append(row[0])
			item_list.append(row[1])
			signed_list.append(row[2])
		# calculate the number of negatives
		ne = signed_list.count(-1)
		print("number of records:\t", len(user_list))
		print("range of userID:\t", min(user_list), max(user_list))
		print("range of itemID:\t", min(item_list), max(item_list))
		n1 = max(user_list) + 1
		n2 = max(item_list) + 1
		Ep = set()
		En = set()
		Nbp = [set() for i in range(n1)]
		Nbn = [set() for i in range(n1)]
		Nsp = [set() for i in range(n2)]
		Nsn = [set() for i in range(n2)]
		for i, sign in enumerate(signed_list):
			if sign == 1:
				Ep.add((user_list[i], item_list[i]))
				Nbp[user_list[i]].add(item_list[i])
				Nsp[item_list[i]].add(user_list[i])
			else:
				En.add((user_list[i], item_list[i]))
				Nbn[user_list[i]].add(item_list[i])
				Nsn[item_list[i]].add(user_list[i])
		for counter, (i, j) in enumerate(Ep):
			if counter % 10000 == 0:
				print("at counter {} of {}".format(counter, len(Ep)))
				sys.stdout.flush()
			for jp in Nbp[i]:
				if jp == j:
					continue
				for ip in Nsp[j]:
					if ip == i:
						continue
					if (ip, jp) in Ep:
						R[1] += 1
					elif (ip, jp) in En:
						R[7] += 1
				# second seller is through +
				for ip in Nsn[j]:
					if ip == 1:
						continue
					if (ip, jp) in En:
						R[3] += 1
			for ip in Nsp[j]:
				if ip == i:
					continue
				for jp in Nbn[i]:
					if jp == j:
						continue
					if (ip, jp) in En:
						R[5] += 1
			for jp in Nbn[i]:
				if jp == j:
					continue
				for ip in Nsn[j]:
					if ip == i:
						continue
					if (ip, jp) in Ep:
						R[4] += 1  # gets counted 2x here #['+-+-']
					elif (ip, jp) in En:
						R[6] += 1  # gets counted 1x #['+---']
		for counter, (i, j) in enumerate(En):
			if counter % 10000 == 0:
				print("at counter {} of {}".format(counter, len(En)))
				sys.stdout.flush()
			for jn in Nbn[i]:
				if jn == j:
					continue
				for i_n in Nsn[j]:
					if i_n == i:
						continue
					if (i_n, jn) in En:
						R[2] += 1
		balanced_pattern = {1,2,3,4,5}
		unbalanced_pattern = {6,7}
		diction= {'balanced_num': 0, "unbalanced_num": 0}

		# calculate the balanced and unbalanced
		for key, value in R.items():
			if key in balanced_pattern:
				if key == 1 or key == 2:
					diction['balanced_num'] += int(value/4)
				else:
					diction['balanced_num'] += int(value/2)
			elif key in unbalanced_pattern:
				diction['unbalanced_num'] += value
		with open(path_add, mode = 'a') as a:
			for key, item in diction.items():
				a.write('{}\t{}\n'.format(key, item))

def ml_1m_run(dir):
	files = os.listdir(dir)
	for i in range(2,6):
		path = os.path.join(dir,'ml-1m(mediate={}).txt'.format(i))
		path_add = os.path.join(dir, 'ml-1m(statistics_value={}).txt'.format(i))
		butterfly_computer(path, path_add)

def ml_100_run(dir):
	files = os.listdir(dir)
	for i in range(2,6):
		path = os.path.join(dir,'u1base(mediate={}).txt'.format(i))
		path_add = os.path.join(dir, 'u1base(statistics_value={}).txt'.format(i))
		butterfly_computer(path, path_add)

if __name__ == "__main__":

	dir_ml1m = 'ml-1m(count)'
	dir_ml100k = 'ml-100k(count)'
	ml_1m_run(dir_ml1m)
	ml_100_run(dir_ml100k)



