import pandas as pd
import numpy as np
import os

def basefile_dataloader(path, intervalue = 3):
	data = []
	user_list= []
	item_list = []
	rating_list = []
	with open(path, mode = 'r') as f:
		for l in f.readlines():
			row = [int(value) for value in l.strip().split('\t')]
			user_list.append(row[0])
			item_list.append(row[1])
			rating_list.append(row[2])
			# print(row)
			data.append(row)

	rating_mean = float(np.mean(rating_list))
	num_ratings = len(rating_list)
	print("the rating mean is:\t", rating_mean)
	#  if we choose the 3 as the intermediate value
	#  if rating > 3, the signed value is 1
	#  if rating < 3, the signed value is -1
	signed_ratings = []
	for value in rating_list:
		if value < intervalue:
			signed_ratings.append(-1)
		else:
			signed_ratings.append(1)
	# compute the number of positive ratings
	posi_ratings = signed_ratings.count(1)
	nega_ratings = signed_ratings.count(-1)
	print("positive numbers:\t", posi_ratings)
	print("negative numbers:\t", nega_ratings)
	signed_user_item = []
	for i in range(num_ratings):
		signed_user_item.append((user_list[i], item_list[i], signed_ratings[i]))
	diction = {}
	diction["total number"] = num_ratings
	diction["positive link"] = posi_ratings
	diction["positive density"] = 1.0 * posi_ratings/num_ratings
	diction["negative link"] = nega_ratings
	diction["negative density"] = 1.0 * nega_ratings/num_ratings
	with open('./u1base(mediate_value = {}).txt'.format(intervalue), mode = 'w') as f:
		f.write("split value is:\t{}\n".format(intervalue))
		for key, item in diction.items():
			f.write('{}\t{}\n'.format(key, item))
		for user, item, rating in signed_user_item:
			f.write('{}\t{}\t{}\n'.format(user, item, rating))
	#return data

def datfile_dataloader(path, intervalue = 3):
	data = []
	user_list= []
	item_list = []
	rating_list = []
	with open(path, mode = 'r') as f:
		for l in f.readlines():
			row = [int(value) for value in l.strip().split('::')]
			user_list.append(row[0])
			item_list.append(row[1])
			rating_list.append(row[2])
			print(row)
			data.append(row)
	signed_ratings = []
	for value in rating_list:
		if value < intervalue:
			signed_ratings.append(-1)
		else:
			signed_ratings.append(1)
	# compute the number of positive ratings
	posi_ratings = signed_ratings.count(1)
	nega_ratings = signed_ratings.count(-1)
	print("positive numbers:\t", posi_ratings)
	print("negative numbers:\t", nega_ratings)
	signed_user_item = []
	num_ratings = len(rating_list)
	for i in range(num_ratings):
		signed_user_item.append((user_list[i], item_list[i], signed_ratings[i]))
	diction = {}
	diction["total number"] = num_ratings
	diction["positive link"] = posi_ratings
	diction["positive density"] = 1.0 * posi_ratings/num_ratings
	diction["negative link"] = nega_ratings
	diction["negative density"] = 1.0 * nega_ratings/num_ratings
	with open('./ml-1m(mediate_value = {}).txt'.format(intervalue), mode = 'w') as f:
		f.write("split value is:\t{}\n".format(intervalue))
		for key, item in diction.items():
			f.write('{}\t{}\n'.format(key, item))
		for user, item, rating in signed_user_item:
			f.write('{}\t{}\t{}\n'.format(user, item, rating))

def txtfile_dataloader(path, intervalue = 2):
	data = []
	user_list = []
	item_list = []
	rating_list = []
	with open(path, mode = 'r') as f:
		for l in f.readlines():
			row = [int(value) for value in l.strip().split('\t')]
			user_list.append(row[0])
			item_list.append(row[1])
			rating_list.append(row[2])
	# calculate total number of
	num_ratigns = len(rating_list)
	posi_num = rating_list.count(1)
	nega_num = rating_list.count(-1)
	signed_user_item = []
	num_ratings = len(rating_list)
	for i in range(num_ratings):
		signed_user_item.append((user_list[i], item_list[i], rating_list[i]))
	diction = {}
	diction["total number"] = num_ratings
	diction["positive link"] = posi_num
	diction["positive density"] = 1.0 * posi_num/num_ratings
	diction["negative link"] = nega_num
	diction["negative density"] = 1.0 * nega_num/num_ratings
	with open('./bonanza(statistics).txt', mode = 'w') as f:
		f.write("split value is:\t{}\n".format(intervalue))
		for key, item in diction.items():
			f.write('{}\t{}\n'.format(key, item))
		for user, item, rating in signed_user_item:
			f.write('{}\t{}\t{}\n'.format(user, item, rating))
if __name__ == "__main__":
	path = './bonanza.txt'
	txtfile_dataloader(path)
