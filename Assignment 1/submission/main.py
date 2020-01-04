#Main script

import sys
import numpy as np

def check_and_update_params(bandit_instance_path, random_seed, epsilon, horizon):
	#Bandit Instance check
	try:
		np.loadtxt(bandit_instance_path)
	except IOError:
		sys.exit("Invalid File Path")
	#Random Seed Check
	try:
		random_seed = int(random_seed)
	except ValueError:
		sys.exit("Random Seed should be convertible to an integer")
	#Epsilon Check
	try:
		epsilon = float(epsilon)
	except ValueError:
		sys.exit("Epsilon should be convertibe to a float")
	#Horizon Check
	try:
		horizon = int(horizon)
	except ValueError:
		sys.exit("Horizon should be convertibe to an integer")

	return random_seed, epsilon, horizon 

def load_bandit_instances(bandit_instance_path):
	return np.loadtxt(bandit_instance_path)

def round_robin(bandit_instances, number_of_arms, random_seed, horizon):
	reward = 0
	np.random.seed(random_seed)
	for iter in range(horizon):
		index = iter%number_of_arms
		p = bandit_instances[index]
		s = np.asscalar(np.random.binomial(1, p, 1))
		reward = reward + s
	return reward

def epsilon_greedy(bandit_instances, number_of_arms, epsilon, random_seed, horizon):
	empirical_mean = np.zeros(number_of_arms)
	frequency_of_arm = np.zeros(number_of_arms)
	reward = 0
	np.random.seed(random_seed)
	for iter in range(horizon):
		choice =  np.asscalar(np.random.binomial(1, epsilon, 1))		#Choice=0 for exploration Choice=1 for exploitation
		if choice == 1:
			index = np.random.randint(number_of_arms)
		else:
			index = np.argmax(empirical_mean)
		
		p = bandit_instances[index]
		s = np.asscalar(np.random.binomial(1, p, 1))
		empirical_mean[index] = ((empirical_mean[index] * frequency_of_arm[index] * 1.0) + s)/(frequency_of_arm[index] + 1.0)
		frequency_of_arm[index] = frequency_of_arm[index] + 1
		reward = reward + s
	return reward

def ucb(bandit_instances, number_of_arms, random_seed, horizon):
	empirical_mean = np.zeros(number_of_arms)
	frequency_of_arm = np.zeros(number_of_arms)
	reward = 0
	np.random.seed(random_seed)
	for iter in range(number_of_arms):
		p = bandit_instances[iter]
		s = np.asscalar(np.random.binomial(1, p, 1))
		frequency_of_arm[iter] = frequency_of_arm[iter] + 1
		empirical_mean[iter] = s
		reward = reward + s
	for iter in range(number_of_arms, horizon):
		ucb = empirical_mean + np.sqrt((2/frequency_of_arm)*np.log(iter))
		index = np.argmax(ucb)
		p = bandit_instances[index]
		s = np.asscalar(np.random.binomial(1, p, 1))
		empirical_mean[index] = ((empirical_mean[index] * frequency_of_arm[index] * 1.0) + s)/(frequency_of_arm[index] + 1.0)
		frequency_of_arm[index] = frequency_of_arm[index] + 1
		reward = reward + s
	return reward

def kl_divergence(p, q):
	return p*np.log(p/q) + (1.0-p)*np.log((1.0-p)/(1.0-q))

def binary_search(p, bound_val, error):
	low = p
	high = 1.0
	if(p == 0):
		p = p+error #To avoid error in KL divergence calculation
	while(high-low > error):
		mid = (high + low)/2
		if kl_divergence(p, mid) <= bound_val:
			low = mid
		else:
			high = mid
	return low

def calculate_ucb(empirical_mean, bound, number_of_arms):
	optimal_p = np.zeros(number_of_arms)
	for iter in range(number_of_arms):
		error = 0.005 #Error tolerance for optimal_p
		optimal_p[iter] = binary_search(empirical_mean[iter], bound[iter], error)
	return optimal_p

def kl_ucb(bandit_instances, number_of_arms, random_seed, horizon):
	empirical_mean = np.zeros(number_of_arms)
	frequency_of_arm = np.zeros(number_of_arms)
	reward = 0
	np.random.seed(random_seed)
	for iter in range(number_of_arms):
		p = bandit_instances[iter]
		s = np.asscalar(np.random.binomial(1, p, 1))
		frequency_of_arm[iter] = frequency_of_arm[iter] + 1
		empirical_mean[iter] = s
		reward = reward + s
	for iter in range(number_of_arms, horizon):
		bound = (np.log(iter) + 3*np.log(np.log(iter)))/frequency_of_arm
		ucb = calculate_ucb(empirical_mean, bound, number_of_arms)
		index = np.argmax(ucb)
		p = bandit_instances[index]
		s = np.asscalar(np.random.binomial(1, p, 1))
		empirical_mean[index] = ((empirical_mean[index] * frequency_of_arm[index] * 1.0) + s)/(frequency_of_arm[index] + 1.0)
		frequency_of_arm[index] = frequency_of_arm[index] + 1
		reward = reward + s
	return reward

def thompson_sampling(bandit_instances, number_of_arms, random_seed, horizon):
	number_of_success = np.zeros(number_of_arms)
	number_of_failures = np.zeros(number_of_arms)
	reward = 0
	np.random.seed(random_seed)
	for iter in range(horizon):
		probability = np.random.beta(number_of_success + 1, number_of_failures + 1)
		index = np.argmax(probability)
		p = bandit_instances[index]
		s = np.asscalar(np.random.binomial(1, p, 1))
		number_of_success[index] = number_of_success[index] + s
		number_of_failures[index] = number_of_failures[index] + (1.0-s)
		reward = reward + s
	return reward

def main():
	bandit_instance_path = sys.argv[1]
	algorithm = sys.argv[2]
	random_seed = sys.argv[3]
	epsilon = sys.argv[4]
	horizon = sys.argv[5]
	
	algorithm_list = ['round-robin', 'epsilon-greedy', 'ucb', 'kl-ucb', 'thompson-sampling']

	random_seed, epsilon, horizon = check_and_update_params(bandit_instance_path, random_seed, epsilon, horizon)
	##Load Bandit Instances
	bandit_instances = load_bandit_instances(bandit_instance_path)
	number_of_arms = np.size(bandit_instances)
	## Maximum expected reward
	max_expected_reward = max(bandit_instances) * horizon
	## Running the algorithm
	if algorithm == 'round-robin':
		reward = round_robin(bandit_instances, number_of_arms, random_seed, horizon)
	elif algorithm == 'epsilon-greedy':
		reward = epsilon_greedy(bandit_instances, number_of_arms, epsilon, random_seed, horizon)
	elif algorithm == 'ucb':
		reward = ucb(bandit_instances, number_of_arms, random_seed, horizon)
	elif algorithm == 'kl-ucb':
		reward = kl_ucb(bandit_instances, number_of_arms, random_seed, horizon)
	elif algorithm == 'thompson-sampling':
		reward = thompson_sampling(bandit_instances, number_of_arms, random_seed, horizon)
	else:
		sys.exit("Invalid Algorithm\n")

	regret = max_expected_reward - reward
	print(bandit_instance_path + ", " + algorithm + ", " + str(random_seed) + ", " + str(epsilon) + ", " + str(horizon) + ", " + str(regret))

if __name__ == '__main__':
	main()