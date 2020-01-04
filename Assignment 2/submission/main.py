import sys
import numpy as np
import pulp

def interpret_input_file(file_path):
	with open(file_path) as f:
		data = f.readlines()
	data = [x.strip() for x in data] #Cite: https://stackoverflow.com/questions/3277503/how-to-read-a-file-line-by-line-into-a-list
	num_states = int(data[0])
	num_actions = int(data[1])
	discount_factor = float(data[-2])
	type_mdp = data[-1]
	reward_function = data[2:(2+num_states*num_actions)]
	reward_function = [x.split() for x in reward_function]
	reward_function = [[float(x) for x in y] for y in reward_function]
	transition_function = data[(2+num_states*num_actions):(2+2*num_states*num_actions)]
	transition_function = [x.split() for x in transition_function]
	transition_function = [[float(x) for x in y] for y in transition_function]
	return num_states, num_actions, reward_function, transition_function, discount_factor, type_mdp

def calculate_value_function(policy, num_states, num_actions, transition_function, reward_function, discount_factor):
	coeff_matrix = np.zeros((num_states, num_states))
	rhs_vector = np.zeros((num_states, 1))
	value_function = np.zeros((num_states, 1))
	for row in range(num_states):
		rhs_val = 0
		for col in range(num_states):
			# LHS matrix
			if row == col:
				coeff_matrix[row][col] = 1 - discount_factor*transition_function[num_actions*row+policy[row]][col]
			else:
				coeff_matrix[row][col] = -1*discount_factor*transition_function[num_actions*row+policy[row]][col]
			# RHS Vector
			rhs_val = rhs_val + transition_function[num_actions*row+policy[row]][col]*reward_function[num_actions*row+policy[row]][col]
		rhs_vector[row] = rhs_val
	if (discount_factor == 1.0):
		coeff_matrix_truncated = coeff_matrix[:-1, :-1]
		value_function[:-1] = np.matmul(np.linalg.inv(coeff_matrix_truncated), rhs_vector[:-1])
		value_function[-1] = 0
	else:
		value_function = np.matmul(np.linalg.inv(coeff_matrix), rhs_vector)
	return value_function

def calculate_Q_matrix(num_states, num_actions, reward_function, transition_function, value_function, discount_factor, type_mdp):
	Q_matrix = np.zeros((num_states, num_actions))
	for iter_state in range(num_states):
		for iter_action in range(num_actions):
			sum_val = 0
			for iter_state_prime in range(num_states):
				sum_val = sum_val + transition_function[num_actions*iter_state+iter_action][iter_state_prime]*(reward_function[num_actions*iter_state+iter_action][iter_state_prime] + discount_factor*value_function[iter_state_prime])
			Q_matrix[iter_state, iter_action] = sum_val
	return Q_matrix	

def solve_hpi(num_states, num_actions, reward_function, transition_function, discount_factor, type_mdp):
	#Choose random Policy to begin the iteration
	optimized = False
	policy = [0] * num_states #Initialization
	while(optimized == False):
		value_function = calculate_value_function(policy, num_states, num_actions, transition_function, reward_function, discount_factor)
		Q_matrix = calculate_Q_matrix(num_states, num_actions, reward_function, transition_function, value_function, discount_factor, type_mdp)
		next_policy = list(Q_matrix.argmax(axis=1))
		if policy==next_policy:
			optimized=True
		else:
			policy=next_policy
	#Printing Final Answer
	for iter in range(num_states):
		print(str(np.asscalar(value_function[iter])) + " " + str(policy[iter]))

def solve_lp(num_states, num_actions, reward_function, transition_function, discount_factor, type_mdp):
	# Cite: http://benalexkeen.com/linear-programming-with-python-and-pulp-part-2/
	# Cite: http://benalexkeen.com/linear-programming-with-python-and-pulp-part-4/
	my_lp_problem = pulp.LpProblem("MDP Solver", pulp.LpMinimize)
	#Defining Variables
	V = []
	for iter in range(num_states):
		V.append("V{}".format(iter))
	val_function_dict = pulp.LpVariable.dicts("Value Function", ((i) for i in V), cat='Continuous')
	# Objective Function
	my_lp_problem += (
		pulp.lpSum([val_function_dict[(i)] for i in V])
		)
	# Constraints
	for iter_state, state_name in enumerate(V):
		for iter_action in range(num_actions):
			my_lp_problem += val_function_dict[(state_name)] >= pulp.lpSum([transition_function[num_actions*iter_state+iter_action][i]*(reward_function[num_actions*iter_state+iter_action][i] + discount_factor * val_function_dict[(i_name)])] for i, i_name in enumerate(V))
	if (discount_factor == 1):
		my_lp_problem += val_function_dict[(V[-1])] == 0
	# Solving LP
	my_lp_problem.solve()
	#print(pulp.LpStatus[my_lp_problem.status])
	optimal_value = []
	for var in val_function_dict:
		var_value = val_function_dict[var].varValue
		optimal_value.append(var_value)
		#print(var_value)
	# Finding optimal action
	optimal_action = []
	for iter_state in range(num_states):
		difference_vector = []
		for iter_action in range(num_actions):
			total_rhs = 0
			for iter_state_prime in range(num_states):
				total_rhs = total_rhs + transition_function[num_actions*iter_state+iter_action][iter_state_prime]*(reward_function[num_actions*iter_state+iter_action][iter_state_prime] + discount_factor * optimal_value[iter_state_prime])
			difference_vector.append(optimal_value[iter_state] - total_rhs)
		optimal_action.append(difference_vector.index(min(difference_vector)))
	# Printing Final Answer
	for iter in range(num_states):
		print(str(optimal_value[iter])+" "+str(optimal_action[iter]))

def main():
	file_path = sys.argv[1]
	algorithm = sys.argv[2]
	num_states, num_actions, reward_function, transition_function, discount_factor, type_mdp = interpret_input_file(file_path)
	if algorithm == "lp":
		solve_lp(num_states, num_actions, reward_function, transition_function, discount_factor, type_mdp)
	elif algorithm == "hpi":
		solve_hpi(num_states, num_actions, reward_function, transition_function, discount_factor, type_mdp)
	else:
		print("Invalid Algorithm")

if __name__ == '__main__':
	main()