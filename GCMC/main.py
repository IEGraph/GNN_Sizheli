from DataLoader import data_loader
from Sample import SampleGenerator
from GCMC import GCMCEngine
from Engine import use_optimizer
import torch
import torch.nn as nn
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

if __name__ == "__main__":
	file_path = './data/ml-100k/raw/u1.base'
	data = data_loader(file_path)
	sample = data.sample_generator
	# specify the exact model
	config = data.construct_config()
	model = GCMCEngine(config)
	print("train set:\n", sample.train)
	print("test set:\n", sample.test)
	for epoch in range(config['num_epoch']):
		train_loader = sample.instance_train_loader(batch_size = config['batch_size'])

		model.train_an_epoch(train_loader, epoch_id = epoch)

