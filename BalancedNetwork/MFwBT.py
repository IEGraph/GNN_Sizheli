from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score
from math import ceil
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from configurature import args
import scipy.sparse as sps
import sys
import torch.nn as nn
import torch
#import matplotlib.pylab as plt

"https://github.com/tylersnetwork/signed_bipartite_networks/blob/master/src/MFwBT/MFwBT.py"

class MatrixFactorization(nn.Module):
	def __init__(self, n_users, n_items, n_factors = 5):
		"""
		:param n_users: the number of users
		:param n_items: the number of items
		:param n_factors: feature dinmensions
		"""
		# constructor that initialized the attributes of the subject
		super(MatrixFactorization, self).__init__()
		# embed those user and item features
		self.user_embedding = nn.Embedding(num_embeddings = n_users, embedding_dim = n_factors, sparse = True)
		self.item_embedding = nn.Embedding(num_embeddings = n_items, embedding_dim = n_factors, sparse = True)

	def forward(self, b, s):
		# b, s are the index of buyers and sellers
		b = Variable(b, requires_grad = False)
		s = Variable(s, requires_grad = False)
		return (self.user_embedding(b)*self.item_embedding(s)).sum(1).sum(1)

	def predict(self, b, s):
		return (self.user_embedding(b)*self.item_embedding(s)).sum(1)

	def save_me(self, output_file):
		torch.save(self.state_dict(), output_file)

def run_matrix_factorization(args, parameters):
	zeros = Variable(torch.zeros([args.minibatch_size]), requires_grad = False)
	ones = Variable(torch.FloatTensor([1]*args.minibatch_size), requires_grad = False)
	# below used on the final minibatch that might be of smaller size
	zeros_left_over = Variable(torch.zeros([len(parameters["links_train"]) % args.minibatch_size]), requires_grad = False)
	ones_left_over = Variable(torch.FloatTensor([1] * (len(parameters["links_train"]) % args.minibatch_size)), requires_grad = False)
	# write the loss function
	def square_hinge(real, pred):
		try:
			loss = torch.max(zeros, (ones - real * pred)) ** 2

		except:
			loss = torch.max(zeros_left_over, (ones_left_over - real * pred)) ** 2
		return torch.mean(loss)

	extra_links_tensor = torch.LongTensor(parameters["extra_links"])
	extra_signs_tensor = torch.FloatTensor(parameters["extra_signs"])
	#extra_signs_tensor = extra_signs_tensor.unsqueeze(dim = 1)
	extra_tensor = TensorDataset(extra_links_tensor, extra_signs_tensor)
	extra_tensor_loader = DataLoader(extra_tensor, shuffle = True, batch_size = args.minibatch_size, drop_last = True)
	# transform the loader into a iterator object
	extra_tensor_iterator = iter(extra_tensor_loader)

	links_tr_tensor = torch.LongTensor(parameters["links_train"])
	signs_tr_tensor = torch.FloatTensor(parameters["signs_train"])
	#signs_tr_tensor = signs_tr_tensor.unsqueeze(dim = 1)
	tr_tensor = TensorDataset(links_tr_tensor, signs_tr_tensor)
	tr_tensor_loader = DataLoader(tr_tensor, shuffle = True, batch_size = args.minibatch_size, drop_last = True)
	tr_tensor_iterator = iter(tr_tensor_loader)

	b_te_tensor = torch.LongTensor([b for b,s in parameters["links_test"]])
	s_te_tensor = torch.LongTensor([s for b,s in parameters["links_test"]])
	signs_te_tensor = torch.FloatTensor(parameters["signs_test"])

	model = MatrixFactorization(n_users = parameters["num_buyers"],
								n_items = parameters["num_sellers"], n_factors = args.dim)

	optimizer = torch.optim.SparseAdam(model.parameters(), lr = args.learning_rate)
	num_minibatches_per_epoch = int(ceil(len(parameters["links_train"])/int(args.minibatch_size)))
	extra_num_minibatcher_per_epoch = int(ceil(len(parameters["extra_links"])/int(args.minibatch_size)))
	mod_balance = args.mod_balance
	alpha = args.alpha
	loss_history = []
	for it in range(args.num_epochs):
		model.train()
		print('epoch {} of {}'.format(it, args.num_epochs))
		for i in range(num_minibatches_per_epoch):
			if (i % mod_balance) != 0:
				# regular links, directed links
				try:
					b_s, sign = next(tr_tensor_iterator)
				except:
					train_tensor_iterator = iter(tr_tensor_loader)
					b_s, sign = next(train_tensor_iterator)
				b = b_s[:, :1]
				s = b_s[:, 1:2]
				prediction = model(b, s)
				loss = square_hinge(Variable(sign), prediction)
			else:
				# extra limks, directed links
				try:
					b_s, sign = next(extra_tensor_iterator)
				except:
					extra_tensor_iterator = iter(extra_tensor_loader)
					b_s, sign = next(extra_tensor_iterator)
				b = b_s[:, :1]
				s = b_s[:, 1:2]
				prediction = model(b, s)
				loss = alpha * square_hinge(Variable(sign), prediction)
			#print("loss is:\t", loss.data.detach().item())
			#loss_history.append(loss.detach().item())
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			# printing progress

	model.eval()
	# can switch to tensor version
	# prediction = model.predict(b_te_tensor, s_te_tensor)
	# until then the below works
	predicted = []
	for (b, s), sign in zip(parameters["links_test"], parameters["signs_test"]):
		b = Variable(torch.LongTensor([int(b)]))
		s = Variable(torch.LongTensor([int(s)]))
		prediction = model.predict(b, s)
		if prediction.data[0] > 0:
			predicted.append(1)
		else:
			predicted.append(-1)
	# calculate AUC and F1
	auc = roc_auc_score(parameters["signs_test"], predicted)
	f1 = f1_score(parameters["signs_test"], predicted)
	accu = accuracy_score(parameters["signs_test"], predicted)
	prec = precision_score(parameters["signs_test"], predicted)
	return auc, f1, loss_history, accu, prec

def runs(args, original_oath):
	train_data = 'train_datafile.txt'
	test_data = 'test_datafile.txt'

	train_links = []
	sign_train = []
	with open(train_data, mode = 'r') as f:
		f.readline()
		for l in f.readlines():
			row = [int(val) for val in l.split('\t')]
			train_links.append((row[0], row[1]))
			sign_train.append(row[2])
	test_links = []
	sign_test = []
	with open(test_data, mode = 'r') as f:
		f.readline()
		for l in f.readlines():
			row = [int(val) for val in l.split('\t')]
			test_links.append((row[0], row[1]))
			sign_test.append(row[2])
	print("the number of train:\t", len(train_links))
	print("the numbe of test:\t", len(test_links))
	extra_links = []
	sign_extra = []
	with open('extra_pos_links_sorted_from_B_balance_theory.txt', mode = 'r') as f:
		for i, l in enumerate(f.readlines()):
			if i > args.extra_pos_num:
				break
			row = [int(val) for val in l.split('\t')]
			extra_links.append((row[0], row[1]))
			sign_extra.append(row[2])
	with open('extra_neg_links_sorted_from_B_balance_theory.txt', mode = 'r') as f:
		for i, l in enumerate(f.readlines()):
			if i > args.extra_neg_num:
				break
			row = [int(val) for val in l.split('\t')]
			extra_links.append((row[0], row[1]))
			sign_extra.append(row[2])
	print("the number of extra:\t", len(extra_links))
	# construct dictions for the parameters
	with open(original_oath, mode = 'r') as f:
		num_buyers, num_sellers, sign = [int(val) for val in f.readline().split('\t')]

	parameters = {}
	parameters["links_train"] = train_links
	parameters["signs_train"] = sign_train
	parameters["links_test"] = test_links
	parameters["signs_test"] = sign_test
	parameters["extra_links"] = extra_links
	parameters["extra_signs"] = sign_extra
	parameters["num_buyers"] = num_buyers
	parameters["num_sellers"] = num_sellers

	auc, f1, loss_history, accuracy, precision = run_matrix_factorization(args = args, parameters = parameters)

	print("the area under curve is:\t", auc)
	print("f1 value is:\t", f1)
	print("accuracy is:\t", accuracy)
	print("precision is:\t", precision)
	with open('metrices_forall_threshold', mode = 'a') as f:
		f.write("auc:\t{}, f1:\t{}, accuracy:\t{}, precision:\t{}".format(auc, f1, accuracy, precision))
	#return loss_history

def plot_history(records):
	plt.figure()
	plt.plot(records, linewidth = 1.5, color = 'black')
	plt.xlabel("iterations")
	plt.ylabel('loss function')
	plt.title("Loss Function Trend")
	plt.show()

if __name__ == "__main__":
	loss_history = runs(args = args)
	#plot_history(loss_history)













