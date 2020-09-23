import pandas as pd

if __name__ == "__main__":
	data = pd.read_csv("ratings.csv", delimiter = ',', engine = 'python')
	print(data[:100])
	# split data into different group by the userId, but reset_index is to name index and columns
	train = data.groupby('userId')['itemId'].apply(set).reset_index().rename(columns = {'itemId':"positive_ratings"})
	print(train.columns)