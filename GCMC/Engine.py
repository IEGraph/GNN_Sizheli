import numpy as np
import torch
import torch.nn as nn

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

class Engine():
	def __init__(self, config):
		self.config = config # model configuration
		self.opt = use_optimizer(self.model, config)
		self.crit = nn.MSELoss()

	def train_single_epoch(self, users, items, ratings):
		assert hasattr(self, 'model')
		if self.config['use_cuda']:
			users, items, ratings = users.cuda(), items.cuda(), ratings.cuda()
		ratings_pred = self.model(users, items)
		loss = self.crit(ratings_pred.view(-1), ratings)
		self.opt.zero_grad()
		loss.backward()
		self.opt.step()
		loss = loss.item()
		return loss

	def train_an_epoch(self, train_loader, epoch_id):
		assert hasattr(self, 'model')
		self.model.train()
		total_loss = 0
		for batch_id, batch_data in enumerate(train_loader):
			assert isinstance(batch_data[0], torch.LongTensor)
			users = batch_data[0]
			items = batch_data[1]
			ratings = batch_data[2].float()
			loss = self.train_single_epoch(users = users, items = items, ratings = ratings)
			print('[Training Epoch {}] Batch {}, Loss {}'.format(epoch_id, batch_id, loss))
			total_loss += loss

