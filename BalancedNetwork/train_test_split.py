import os
import numpy as np

def split_train_test(path):
	train_file_list = []
	test_file_list = []
	with open(path, mode = 'r') as f:
		num_b, num_s, num_links = f.readline().split('\t')
		print("the number of buyers:\t{}, the number of sellers:\t{}, the number of links:\t{}".format(
			num_b, num_s, num_links
		))
		links = []
		signs = []
		for l in f.readlines():
			rows = [int(val) for val in l.split('\t')]
			links.append((rows[0], rows[1]))
			signs.append(rows[2])
	pos_num = signs.count(1)
	neg_num = signs.count(-1)
	print("positive number:\t", pos_num)
	print("negative number:\t", neg_num)
	for i, link in enumerate(links):
		if i % 10 != 0:
			train_file_list.append('{}\t{}\t{}'.format(link[0], link[1], signs[i]))
		else:
			test_file_list.append('{}\t{}\t{}'.format(link[0], link[1], signs[i]))
	print("display the train list:\n", len(train_file_list))
	print("display the test list:\n", len(test_file_list))
	with open("./train_datafile.txt", mode = 'w') as f:
		f.write('\n'.join(train_file_list))
	with open("./test_datafile.txt", mode = 'w') as f:
		f.write('\n'.join(test_file_list))

if __name__ == "__main__":
	files = os.listdir()
	for file in files:
		if file[:4] == 'bona':
			path = file
	split_train_test(path)
