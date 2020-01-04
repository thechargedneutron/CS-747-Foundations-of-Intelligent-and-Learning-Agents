import sys
import random
import numpy as np


def TD0_iteration(optimal_value_function, state, action, n_actions, reward, discount_factor):
	N = len(reward)
	for t in range(N):
		optimal_value_function[state[t]] = optimal_value_function[state[t]] + 0.00001*(reward[t] + discount_factor*optimal_value_function[state[t+1]] - optimal_value_function[state[t]])
	return optimal_value_function

def mse(list1, list2):
	list1 = np.array(list1)
	list2 = np.array(list2)
	return (np.square(list1 - list2)).mean(axis=None)

def main():
	filename = sys.argv[1]
	with open(filename) as f:
		data = [line.rstrip() for line in f]
	
	n_states = int(data[0])
	#print(n_states)

	n_actions = int(data[1])
	#print(n_actions)

	discount_factor = float(data[2])
	#print(discount_factor)

	data = [line.split() for line in data]

	state = [int(x[0]) for x in data[3:]]
	action = [int(y[1]) for y in data[3:-1]]
	reward = [float(z[2]) for z in data[3:-1]]

	#Random Initialization
	optimal_value_function = random.sample(xrange(50), n_states)
	optimized = False
	while(not optimized):
		optimal_value_function_store = optimal_value_function[:]
		optimal_value_function = TD0_iteration(optimal_value_function, state, action, n_states, reward, discount_factor)
		if (mse(optimal_value_function_store, optimal_value_function) < 1e-7):
			optimized = True
	#Comparing mse
	'''
	with open(soln_filename) as f2:
		soln = [line.rstrip() for line in f2]
		soln = [float(x) for x in soln]
		print("MSE: ", mse(soln, optimal_value_function))
	'''
	#Printing Value Function
	for it in range(n_states):
		print(optimal_value_function[it])

if __name__ == '__main__':
	main()