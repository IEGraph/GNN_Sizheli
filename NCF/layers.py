'https://github.com/yihong-chen/neural-collaborative-filtering'
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from train import Train
# now the config
class GMF(torch.nn.Module):
	def __init__(self, config):
		super(GMF, self).__init__()
		self.confit = config
		self.num_users = config['num_users']
		self.num_items = config['num_items']
		self.latent_dim = config['latent_dim']

		self.embedding_user = nn.Embedding(num_embeddings = self.num_users, embedding_dim = self.latent_dim)
		self.embedding_item = nn.Embedding(num_embeddings = self.num_items, embedding_dim = self.latent_dim)

		self.affine_output = nn.Linear(in_features = self.latent_dim, out_features = 1)
		self.logistic = nn.Sigmoid()

	def forward(self, users, items):
		# the length of users equals the length of items
		user_embeddings = self.embedding_user(users)
		item_embeddings = self.embedding_item(items)
		# the element_wise product is p-user mul q-item
		element_product = torch.mul(user_embeddings, item_embeddings)
		ratings = self.affine_output(element_product)
		return self.logistic(ratings)

	def init_weight(self):
		pass
	
class GMFEngine(Train):
	def __init__(self, config):
		self.model = GMF(config)
		super(GMFEngine, self).__init__(config)
		
class MLP(nn.Module):
	def __init__(self, config):
		super(MLP, self).__init__()
		self.num_users = config['num_users']
		self.num_items = config['num_items']
		self.latent_dim = config['latent_dim']
		self.layers = config['layers']

		self.embedding_user = nn.Embedding(num_embeddings = self.num_users, embedding_dim = self.latent_dim)
		self.embedding_item = nn.Embedding(num_embeddings = self.num_items, embedding_dim = self.latent_dim)

		# construct a module collection to take all the layers in such one package
		self.fc_layers = torch.nn.ModuleList()
		for i in range(len(self.layers) - 1):
			self.fc_layers.append(nn.Linear(in_features = config['layers'][i], out_features = config['layers'][i+1]))
			print(config['layers'][i])

		self.affine_output = nn.Linear(in_features = config['layers'][-1], out_features = 1)
		self.logistic = nn.Sigmoid()

	def forward(self, users, items):
		# the length of users equals the length of items
		user_embeddings = self.embedding_user(users)
		item_embeddings = self.embedding_item(items)
		# stack the user_embeddings and item_embeddings
		vector = torch.cat([user_embeddings, item_embeddings], dim = 1)
		for i in range(len(self.fc_layers)):
			vector = self.fc_layers[i](vector)
			vector = nn.ReLU()(vector)

		logits = self.affine_output(vector)
		rating = self.logistic(logits)
		return rating

	def init_weight(self):
		pass

	def load_pretrain_weight(self):
		"""
		At the class initialization, we construct the embeddings of users and items
		:return:
		"""
		gmf_model = GMF(config)
		if self.config['use_cuda']:
			gmf_model.cuda()
		self.embedding_user.weight.data = gmf_model.embedding_user.weight.data
		self.embedding_item.weight.data = gmf_model.embedding_item.weight.data

class MLPEngine(Train):
	def __init__(self, config):
		self.model = MLP(config)
		super(MLPEngine, self).__init__(config)

if __name__ == "__main__":
	config = {'num_users':100, 'num_items':120, 'latent_dim':10, 'layers':[20, 30, 10], 'use_cuda':False}
	model_GMF = MLP(config)
	k1 = torch.randint(0,5,(3,5))
	k2 = torch.randint(0,5,(4,5))
	#result = torch.mul(k1, k2)
	k = torch.cat([k1, k2], dim = 0)
	for i in range(len(model_GMF.fc_layers)):
		print(model_GMF.fc_layers[i].weight.data.size())
	# set users and items
	user_indices = torch.LongTensor(random.sample(range(config['num_users']), 10))
	item_indices = torch.LongTensor(random.sample(range(config['num_items']), 10))
	print(user_indices)
	print(item_indices)
	ratings = model_GMF(user_indices, item_indices)
	print(ratings)