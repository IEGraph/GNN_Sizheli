import argparse

def parameter_parser():
	"""
	all arguments are in the args
	:return:
	"""
	parser = argparse.ArgumentParser(description = "SGCN")
	parser.add_argument("--edge-path",
						nargs = "?",
						default = "./bitcoin_otc.csv",
						help = "Edge list csv.")
	parser.add_argument("--features-path",
						nargs = "?",
						default = "./bitcoin_otc.csv",
						help = "Target embedding csv.")
	parser.add_argument("--test-size",
						type = float,
						default = 0.3,
						help = "this rate is the split proportion of train and test.")
	parser.add_argument("--epochs",
						type = int,
						default = 100,
						help = "Number of training epochs. Default is 100.")
	parser.add_argument("--reduction-iteration",
						type = int,
						default = 30,
						help = "Number of SVD iterations. Default is 30.")
	parser.add_argument("--learning-rate",
						type = float,
						default = 0.2,
						help = "Test dataset size. Default is 0.2.")
	parser.add_argument("--weight-decay",
						type = float,
						default = 10**-5,
						help = "default is 10**-5.")
	parser.add_argument("--seed",
						type = int,
						default = 42,
						help = "random seed for pre-training.")
	parser.add_argument("--lamb",
						type = int,
						default = 1.0,
						help = "Embedding regularization parameter, default is 1.0")
	parser.add_argument("--reduction-iterations",
						type = int,
						default = 30,
						help = "Number of SVD interations, here is 30.")
	parser.add_argument("--reduction-dimensions",
						type = int,
						default = 64,
						help = "Number of SVD feature extraction dimensions. Default is 64.")
	parser.add_argument("--spectral-features",
						dest = "spectral_features",
						action = "store_true")
	parser.add_argument("--general-features",
						dest = "general_features",
						action = "store_false")
	parser.add_argument("--layers",
						nargs = "+",
						type = int,
						help = "Layer dimensions separated by space.")
	parser.set_defaults(spectral_features = True)
	parser.set_defaults(layers = [32,32])

	return parser.parse_args()