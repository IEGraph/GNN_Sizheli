
class config():
	def __init__(self):
		self.extra_neg_num = 10000
		self.extra_pos_num = 10000

		self.dim = 10
		self.random_seed = 123
		self.learning_rate = 0.015
		self.tuning = True
		self.num_epochs = 3
		self.minibatch_size = 100
		self.alpha = 0.01
		self.reg = 0.001
		self.mod_balance = 10

args = config()

