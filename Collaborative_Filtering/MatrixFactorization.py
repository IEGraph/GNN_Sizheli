import pandas as pd
import numpy as np
import os
import scipy.sparse as sps
from scipy.sparse.linalg import svds

ratings_list = []
movie_list = []
user_list = []
with open('./ml-1m/ratings.dat', mode = 'r') as f:
	for line in f.readlines():
		row = line.strip().split("::")
		ratings_list.append(list(map(lambda x: int(x), row)))

with open('./ml-1m/users.dat', mode = 'r') as f:
	for lin in f.readlines():
		row = line.strip().split("::")
		user_list.append(row)

ratings = np.array(ratings_list)
users = np.array(user_list)

print("the size of ratings:\t", ratings.shape)
print("the size of users:\t", users.shape)

ratings_df = pd.DataFrame(ratings_list, columns = ["UserID","MovieID","Rating","TimeStamp"], dtype = np.int)

#R_df = ratings_df.pivot(index = "UserID", columns = "MovieID", values = "Rating").fillna(0)
num_users = ratings_df["UserID"].max()
num_movies = ratings_df["MovieID"].max()

user_index = list(ratings_df["UserID"]-1)
movie_index = list(ratings_df["MovieID"]-1)
rating_record = list(ratings_df["Rating"])

# product a sparse matrix
R_Matrix = sps.dok_matrix((num_users, num_movies), dtype = np.float)
for i in range(len(user_index)):
	R_Matrix[user_index[i], movie_index[i]] = rating_record[i]

R_Matrix = R_Matrix.toarray()
user_ratings_mean = np.mean(R_Matrix, axis = 1, keepdims = True)
print("mean_matrix_shape\t",user_ratings_mean.shape)
R_demeaned = R_Matrix  - user_ratings_mean

# Singular Value Decomposition
U, sigma, Vt = svds(R_demeaned, k = 100)
print("the shape of U:\t", U.shape)
print("the shape of sigma:\t",sigma.shape)
print("the shape of Vt:\t", Vt.shape)

sigma = np.diag(sigma)
all_user_prediction_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean
prediction_ratings = all_user_prediction_ratings[user_index, movie_index]

"""
print("display the 10:\n", prediction_ratings[:10])
print("display the 19:\n", rating_record[:10])
"""

preds_df = pd.DataFrame(all_user_prediction_ratings, columns = list(range(1,num_movies+1)))
print(preds_df.loc[1])


def recommend_movies(predictions_df, userID, movies_df, original_ratings_df, num_recommendations=5):
	# Get and sort the user's predictions
	user_row_number = userID - 1  # UserID starts at 1, not 0
	sorted_user_predictions = preds_df.iloc[user_row_number].sort_values(ascending = False)  # UserID starts at 1
	# Get the user's data and merge in the movie information.
	user_data = original_ratings_df[original_ratings_df.UserID == (userID)]
	user_full = (user_data.merge(movies_df, how = 'left', left_on = 'MovieID', right_on = 'MovieID').
			 sort_values(['Rating'], ascending = False)
			 )
	print('User {0} has already rated {1} movies.'.format(userID, user_full.shape[0]))
	print('Recommending highest {0} predicted ratings movies not already rated.'.format(num_recommendations))

	# Recommend the highest predicted rating movies that the user hasn't seen yet.
	recommendations = (movies_df[~movies_df['MovieID'].isin(user_full['MovieID'])].
					   merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',
							 left_on = 'MovieID',
							 right_on = 'MovieID').
					   rename(columns = {user_row_number: 'Predictions'}).
					   sort_values('Predictions', ascending = False).
					   iloc[:num_recommendations, :-1]
					   )

	return user_full, recommendations