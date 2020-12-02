import torch
import torch.nn.init as init
from arguments import parameter_parser
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
from DataLoader import read_file, setup_feature
from torch.nn import Parameter
from sklearn.model_selection import train_test_split
from SGCN import SGCN

import numpy as np

def run_main():
	args = parameter_parser()
	edges = read_file(args)
	trainer = train_sgcn(args = args, edges = edges)
	trainer.setup_Dataset()
	trainer.create_and_train_model()

class train_sgcn():
	def __init__(self, args, edges):
		self.args = args
		self.edges = edges
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.setup_logs()
	def setup_logs(self):
		"""
		create a dictionary.
		:return:
		"""
		self.logs = {}
		self.logs["parameters"] = vars(self.args)

	def setup_Dataset(self):
		"""
		Creating train and test split.
		:return:
		"""
		self.train_positive_edges, self.test_positive_edges = train_test_split(self.edges["positive_edges"],
																			   test_size = self.args.test_size)
		self.train_negative_edges, self.test_negative_edges = train_test_split(self.edges["negative_edges"],
																			   test_size = self.args.test_size)
		# count the number of edges for training
		self.edges_number = len(self.train_positive_edges) + len(self.train_negative_edges)
		positive_num = len(self.train_positive_edges)
		negative_num = len(self.train_negative_edges)
		print("positive_edges number is:\t", positive_num)
		print("negative_edges number is:\t", negative_num)
		# product embeddings of nodes
		self.X = setup_feature(args = self.args, positive_edges = self.train_positive_edges, negative_edges = self.train_negative_edges,
							   node_num = self.edges_number)
		self.X = torch.from_numpy(self.X).type(torch.float).to(self.device)
		# we put the source node at the first row, the destination node at the second row, so here we make a transpose.
		self.train_positive_edges = torch.from_numpy(np.array(self.train_positive_edges, dtype = np.int64).T).type(torch.long).to(self.device)
		self.train_negative_edges = torch.from_numpy(np.array(self.train_negative_edges, dtype = np.int64).T).type(torch.long).to(self.device)
		# set the target
		self.y = np.array([0 if i < positive_num else 1 for i in range(self.edges_number)])
		self.y = torch.from_numpy(self.y).type(torch.long).to(self.device)
	def create_and_train_model(self):
		"""
		Model training and scoring
		:return:
		"""
		self.model = SGCN(device = self.device, args = self.args, node_features = self.X).to(self.device)
		self.optimizer = torch.optim.Adam(self.model.parameters(),
										  lr = self.args.learning_rate,
										  weight_decay = self.args.weight_decay)
		self.model.train()
		self.epochs = self.args.epochs
		for epoch in range(self.epochs):
			self.optimizer.zero_grad()
			loss, _ = self.model(self.train_positive_edges, self.train_negative_edges, self.y)
			loss.backward()
			self.optimizer.step()
			print("the loss value is:\t", loss.detach().item())
		self.evaluate_model()
	def evaluate_model(self):
		self.model.eval()
		_, self.train_z = self.model(self.train_positive_edges, self.train_negative_edges, self.y)
		score_positive_edges = torch.from_numpy(np.array(self.test_positive_edges, dtype = np.int64).T).type(
			torch.long).to(self.device)
		score_negative_edges = torch.from_numpy(np.array(self.test_negative_edges, dtype = np.int64).T).type(
			torch.long).to(self.device)
		test_positive_z = torch.cat(
			(self.train_z[score_positive_edges[0, :], :], self.train_z[score_positive_edges[1, :], :]), 1)
		test_negative_z = torch.cat(
			(self.train_z[score_negative_edges[0, :], :], self.train_z[score_negative_edges[1, :], :]), 1)
		scores = torch.mm(torch.cat((test_positive_z, test_negative_z), 0),
						  self.model.final_regression_weight)
		probability_score = F.softmax(scores, dim = 1)
		predictions = np.argmax(probability_score.cpu().detach().numpy(), axis = 1)
		print("the predictions is:\n", predictions)
		targets = [0] * len(self.test_positive_edges) + [1] * len(self.test_negative_edges)

		accuracy = accuracy_score(targets, predictions)

		print("accuracy is:\t", accuracy)



if __name__ == "__main__":
	run_main()






