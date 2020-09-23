import torch
import torch.nn as nn
from torch.autograd import Variable
import time

def use_optimizer(network, params):
	if params['optimizer'] == 'sgd':
		optimizer = torch.optim.SGD(network.parameters(),
									lr=params['sgd_lr'],
									momentum=params['sgd_momentum'],
									weight_decay=params['l2_regularization'])
	elif params['optimizer'] == 'adam':
		optimizer = torch.optim.Adam(network.parameters(),
									 lr=params['adam_lr'],
									 weight_decay=params['l2_regularization'])
	elif params['optimizer'] == 'rmsprop':
		optimizer = torch.optim.RMSprop(network.parameters(),
										lr=params['rmsprop_lr'],
										alpha=params['rmsprop_alpha'],
										momentum=params['rmsprop_momentum'])
	return optimizer

class Train():
	def __init__(self, config):
		self.config = config # model configuration
		self.opt = use_optimizer(self.model, config)
		self.crit = nn.BCELoss()
	def train_single_epoch(self, users, items, ratings):
		assert hasattr(self, 'model')
		self.opt.zero_grad()
		ratings_pred = self.model(users, items)
		loss = self.crit(ratings_pred.view(-1), ratings)
		loss.backward()
		self.opt.step()
		return loss.item()

	def train_an_epoch(self, train_loader, epoch_id):
		assert hasattr(self, 'model')
		self.model.train()
		total_loss = 0
		for batch_id, batch in enumerate(train_loader):
			users, items, ratings = batch[0], batch[1], batch[2]
			ratings = ratings.float()
			loss = self.train_single_epoch(users = users, items = items, ratings = ratings)
			print('[Training Epoch {}] Batch {}, Loss {}'.format(epoch_id, batch_id, loss))
			total_loss += loss

