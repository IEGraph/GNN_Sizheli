from MFwBT import runs
from get_extra_links import data_write
from train_test_split import split_train_test
from configurature import args

def main(value = 2):
	split_train_test('ml1m(statistics)/ratings(mediate={}).txt'.format(value))
	data_write('train_datafile.txt')
	original_path = 'ml1m(statistics)/ratings(mediate={}),txt'.format(value)
	runs(args = args, original_oath = original_path)

def main_1(path = 'bonanza_cikm2019_balance_in_signed_bipartite_networks.txt'):
	split_train_test(path)
	data_write('train_datafile.txt')
	runs(args = args, original_oath = path)

if __name__ == "__main__":
	main_1()
