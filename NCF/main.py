import torch
import torch.nn as nn
from load_data import Data, UserItemRatingDataset
from neumf import NeuMF
from layers import GMFEngine, MLPEngine
import time

gmf_config = {'alias': 'gmf_factor8neg4-implict',
              'num_epoch': 200,
              'batch_size': 1024,
              # 'optimizer': 'sgd',
              # 'sgd_lr': 1e-3,
              # 'sgd_momentum': 0.9,
              # 'optimizer': 'rmsprop',
              # 'rmsprop_lr': 1e-3,
              # 'rmsprop_alpha': 0.99,
              # 'rmsprop_momentum': 0,
              'optimizer': 'adam',
              'adam_lr': 1e-3,
              'num_users': 6040,
              'num_items': 3706,
              'latent_dim': 8,
              'num_negative': 4,
              'l2_regularization': 0, # 0.01
              'use_cuda': True,
              'device_id': 0,
              'model_dir':'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'}


if __name__ == "__main__":
	path = 'ratings.csv'
	data = Data(path)
	sample_generator, evaluate_data = data.sample_gen_evaluate()
	# specify the exact model
	config = gmf_config
	model = GMFEngine(config = config)
	print("train set:\n", sample_generator.train)
	print("test set:\n", sample_generator.test)
	for epoch in range(config['num_epoch']):
		print("Epoch {} starts !".format(epoch))
		print('-' * 50)
		train_loader = sample_generator.instance_train_loader(num_negative = config['num_negative'],
															  batch_size = config['batch_size'])
		model.train_an_epoch(train_loader, epoch_id = epoch)

