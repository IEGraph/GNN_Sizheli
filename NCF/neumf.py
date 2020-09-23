import torch
from layers import GMF, MLP
import torch.nn as nn
import random
from train import Train

class NeuMF(torch.nn.Module):
	def __init__(self, config):
		super(NeuMF, self).__init__()
		self.config = config
		self.num_users = config['num_users']
		self.num_items = config['num_items']
		self.layers = config['layers']

		self.latent_dim_mf = config['latent_dim_mf']
		self.latent_dim_mlp = config['latent_dim_mlp']

		self.embedding_user_mlp = nn.Embedding(num_embeddings = self.num_users, embedding_dim = self.latent_dim_mlp)
		self.embedding_item_mlp = nn.Embedding(num_embeddings = self.num_items, embedding_dim = self.latent_dim_mlp)
		self.embedding_user_mf = nn.Embedding(num_embeddings = self.num_users, embedding_dim = self.latent_dim_mf)
		self.embedding_item_mf = nn.Embedding(num_embeddings = self.num_items, embedding_dim = self.latent_dim_mf)

		self.fc_layers = nn.ModuleList()
		"""
		Now we should pay attention to the fc layers is that, 
		"""
		for idx, (in_size, out_size) in enumerate(zip(config['layers'][:-1], config['layers'][1:])):
			self.fc_layers.append(nn.Linear(in_size, out_size))

		self.affine_output = nn.Linear(in_features = config['layers'][-1] + config['latent_dim_mf'], out_features = 1)
		self.logistic = nn.Sigmoid()

	def forward(self, users, items):
		user_embedding_mlp = self.embedding_user_mlp(users)
		item_embedding_mlp = self.embedding_item_mlp(items)
		user_embedding_mf = self.embedding_user_mf(users)
		item_embedding_mf = self.embedding_item_mf(items)

		mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim = 1)
		mf_vector = torch.mul(user_embedding_mf, item_embedding_mf)

		for i in range(len(self.fc_layers)):
			mlp_vector = self.fc_layers[i](mlp_vector)
			mlp_vector = nn.ReLU()(mlp_vector)

		vector = torch.cat([mlp_vector, mf_vector], dim = 1)
		logits = self.affine_output(vector)
		rating = self.logistic(logits)

		return rating

	def init_weight(self):
		pass

	def load_pretrain_weight(self):
		"""
		loading weights from pre-trained MLP and GMF model
		:return:
		"""
		config = self.config
		config['latent_dim'] = config['latent_dim_mlp']
		mlp_model = MLP(config)
		if config['use_cuda']:
			mlp_model.cuda()

		self.embedding_user_mlp.weight.data = mlp_model.embedding_user.weight.data
		self.embedding_item_mlp.weight.data = mlp_model.embedding_item.weight.data
		for i in range(len(self.fc_layers)):
			self.fc_layers[i].weight.data = mlp_model.fc_layers[i].weight.data

		config['latent_dim'] = config['latent_dim_mf']
		gmf_model = GMF(config)
		if config['use_cuda']:
			gmf_model.cuda()
		self.embedding_user_mf.weight.data = gmf_model.embedding_user.weight.data
		self.embedding_item_mf.weight.data = gmf_model.embedding_item.weight.data

		self.affine_output.weight.data = torch.cat([config['alpha'] * mlp_model.affine_output.weight.data,
													(1 - config['alpha']) * gmf_model.affine_output.weight.data],
													dim = 0)
		self.affine_output.bias.data = config['alpha'] * mlp_model.affine_output.bias.data + (1 - config['alpha']) \
									   * gmf_model.affine_out.bias.data

class NeuEngine(Train):
	def __init__(self, config):
		self.model = NeuMF(config)
		super(NeuEngine, self).__init__(config)
if __name__ == '__main__':
	config = {'num_users':100, 'num_items':120, 'latent_dim_mf':8, 'latent_dim_mlp':5, 'layers':[10, 30, 10], 'use_cuda':False, 'alpha': 0.5}
	model_GMF = NeuMF(config)
	for i in range(len(model_GMF.fc_layers)):
		print(model_GMF.fc_layers[i].weight.data.size())
	# set users and items
	user_indices = torch.LongTensor(random.sample(range(config['num_users']), 10))
	item_indices = torch.LongTensor(random.sample(range(config['num_items']), 10))
	print(user_indices)
	print(item_indices)
	ratings = model_GMF(user_indices, item_indices)
	print(ratings)

