import numpy as np
import scipy.sparse as sps
import os
import random

class MarkovChain():
	def __init__(self, N = 9):
		self.P = sps.dok_matrix((N, N), dtype = float)
		self.N = N
		row_col = []
		for i in range(N):
			row_col.append((i, i))
			if i < (N-1):
				row_col.append((i, i+1))
			else:
				row_col.append((i,0))
		print(row_col)
		for i, item in enumerate(row_col):
			self.P[item[0], item[1]] = 0.5
		print("the producted P matrix:\n",self.P.toarray())
		self.initial_vector = self._setInitial()
		self.P = self.P.toarray()

	def _getP(self):
		return self.P

	def iterations(self, iterations = 100):
		vector1 = self.initial_vector
		P_matrix = self.P
		for i in range(iterations):
			vector1 = vector1.dot(P_matrix)
			print("state is :\n",vector1)

	def _setInitial(self, n = 3):
		total = list(range(self.N))
		initial_state = random.sample(total, n)
		initial_vector = np.zeros(self.N)
		initial_vector[initial_state] = [0.25,0.25,0.5]
		print("the initial state vector:\t",initial_vector)
		return initial_vector.reshape(1,-1)


if __name__ == "__main__":
	model = MarkovChain()
	model._setInitial()
	model.iterations()
