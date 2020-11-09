import os
import numpy as np
import torch.nn as nn
from data import dataset
from math import sqrt

def data_load(dataset):
	for person in dataset:
		print("the user:\t",person)


def similarity_score(person1, person2, dataset):
	# return ratio Euclidean distance score of person1 and person2
	both_viewed = {}
	for item in dataset[person1]:
		if item in dataset[person2]:
			both_viewed = 1
		# Conditions to check they both have an common rating items
	if len(both_viewed) == 0:
		return 0
	# finding Euclidean distance
	sum_of_euclidean_distance = []
	for item in dataset[person1]:
		if item in dataset[person2]:
			sum_of_euclidean_distance.append(pow((dataset[person1][item] - dataset[person2][item]),2))
	sum_of_euclidean_distance = sum(sum_of_euclidean_distance)
	return  1/(1+sqrt(sum_of_euclidean_distance))

def pearson_correlation(person1, person2):
	# To get both rated items
	both_rated = {}
	for item in dataset[person1]:
		if item in dataset[person2]:
			both_rated[item] = 1
	number_of_ratings = len(both_rated)

	# checking for that
	if number_of_ratings == 0:
		return 0

	# Add up all the preferences of each user
	person1_preference_sum = sum([dataset[person1][item] for item in both_rated])
	person2_preference_sum = sum([dataset[person2][item] for item in both_rated])

	# Add up all squares of preferences of each user
	person1_square_preference_sum = sum([pow(dataset[person1][item], 2) for item in both_rated])
	person2_square_preference_sum = sum([pow(dataset[person2][item], 2) for item in both_rated])
	product_sum_of_both_users = sum([dataset[person1][item] * dataset[person2][item] for item in both_rated])

	# Calculate the pearson score
	numerator_value = product_sum_of_both_users - (person1_preference_sum * person2_preference_sum/number_of_ratings)
	denominator_value = sqrt((person1_square_preference_sum - pow(person1_preference_sum, 2) / number_of_ratings) * (
				person2_square_preference_sum - pow(person2_preference_sum, 2) / number_of_ratings))
	if denominator_value == 0:
		return 0
	else:
		r = numerator_value/denominator_value
		return r

def most_similiar_users(person, number_of_user):
	scores = [(pearson_correlation(person, other_person), other_person) for other_person in dataset if
			  other_person != person]
	scores.sort(reverse = True)
	return scores[:number_of_user]

def user_recommendations(person):
	# Gets recommendations for a person by using a weighted average of every other users' rankings
	totals = {}
	simSums = {}
	rankings_list = []
	for other in dataset.keys():
		if other == person:
			continue
		similarity = pearson_correlation(person, other)
		if similarity <= 0:
			continue
		for item in dataset[other]:
			if item not in dataset[person] or dataset[person][item] == 0:

				# similarity * score
				totals.setdefault(item, 0)
				totals[item] += dataset[other][item] * similarity
				# sum of similarities
				simSums.setdefault(item,0)
				simSums[item] += similarity
		# Create the normalized list
		rankings = [(total/simSums[item], item)for item, total in totals.items()]

		rankings.sort(key = lambda t:t[0], reverse = True)
		recommender_list = [item for score,item in rankings]

		return recommender_list

if __name__ == "__main__":
	print(user_recommendations('Toby'))