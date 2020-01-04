import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

def next_state_decoder(num_rows, num_cols, wind_direction, curr_state, action, random_seed, stochastic):
    '''
    Function to decode the next state given the current state and action.

    The structure of the board is same as that given in the reference book. In
    addition to this, the wind direction and strength is also same as that in
    the book. State and Action numbering is described in the report.
    '''
    if stochastic == True:
        offset = np.random.randint(-1, 2, num_cols) #Values can be -1, 0, 1
        wind_direction = wind_direction + offset
    curr_col = curr_state % num_cols
    curr_row = curr_state / num_cols
    if action == 0: #UP
        new_col = curr_col
        increment = 1 + wind_direction[curr_col]
        new_row = curr_row - increment
    elif action == 1: #DOWN
        new_col = curr_col
        increment = 1 - wind_direction[curr_col]
        new_row = curr_row + increment
    elif action == 2: #RIGHT
        new_row = curr_row - wind_direction[curr_col]
        new_col = curr_col + 1
    elif action == 3: #LEFT
        new_row = curr_row - wind_direction[curr_col]
        new_col = curr_col - 1
    elif action == 4: #UP-RIGHT
        new_col = curr_col + 1
        increment = 1 + wind_direction[curr_col]
        new_row = curr_row - increment
    elif action == 5: #UP-LEFT
        new_col = curr_col - 1
        increment = 1 + wind_direction[curr_col]
        new_row = curr_row - increment
    elif action == 6: #DOWN-RIGHT
        new_col = curr_col + 1
        increment = 1 - wind_direction[curr_col]
        new_row = curr_row + increment
    elif action == 7: #DOWN-LEFT
        new_col = curr_col - 1
        increment = 1 - wind_direction[curr_col]
        new_row = curr_row + increment
    else:
        print("INVALID ACTION")
    
    #Adusting Boundary Cases
    if new_col < 0:
        new_col = 0
    if new_col >= num_cols:
        new_col = num_cols-1
    if new_row < 0:
        new_row = 0
    if new_row >= num_rows:
        new_row = num_rows-1

    return (num_cols*new_row + new_col)

def evaluate_sarsa_0(Q_matrix, num_rows, num_cols, wind_direction, start_state, end_state, num_states, num_actions, num_steps, alpha, gamma, epsilon, random_seed, stochastic=False):
    step_count = 0
    num_episodes = 0
    episode_list = []
    np.random.seed(random_seed)
    while(step_count < num_steps):
        curr_state = start_state
        choice =  np.asscalar(np.random.binomial(1, epsilon, 1))		#Choice=0 for exploration Choice=1 for exploitation
        if choice == 1:
            action = np.random.randint(num_actions)
        else:
            action = np.argmax(Q_matrix[curr_state][:])
        while(curr_state != end_state and step_count < num_steps):
            next_state = next_state_decoder(num_rows, num_cols, wind_direction, curr_state, action, random_seed, stochastic=stochastic)
            if next_state == end_state:
                reward = 0
            else:
                reward = -1
            choice = np.asscalar(np.random.binomial(1, epsilon, 1))
            if choice == 1:
                next_action = np.random.randint(num_actions)
            else:
                next_action = np.argmax(Q_matrix[next_state][:])
            Q_matrix[curr_state][action] = Q_matrix[curr_state][action] + alpha*(reward + gamma * Q_matrix[next_state][next_action] - Q_matrix[curr_state][action])
            step_count = step_count + 1
            #print(step_count)
            if next_state == end_state:
                num_episodes = num_episodes+1
            episode_list.append(num_episodes)
            curr_state = next_state
            action = next_action
    return episode_list

def main(filename, subpart, alpha, gamma, epsilon, num_steps, random_seed):
    with open(args.filename) as f:
        board_parameters = [line.rstrip() for line in f]
    num_rows = int(board_parameters[0])
    num_cols = int(board_parameters[1])
    start_state = int(board_parameters[2])
    end_state = int(board_parameters[3])
    wind_direction = [int(it) for it in board_parameters[4].split()]
    num_states = num_rows*num_cols
    if subpart == "A":
        num_actions = 4
        stochastic = False
    elif subpart == "B":
        num_actions = 8
        stochastic = False
    elif subpart == "C":
        num_actions = 4
        stochastic = True
    else:
        raise ValueError
    #Initialization
    Q_matrix = np.zeros((num_states, num_actions))
    num_episodes = evaluate_sarsa_0(Q_matrix, num_rows, num_cols, wind_direction, start_state, end_state, num_states, num_actions, num_steps, alpha, gamma, epsilon, random_seed, stochastic=stochastic)
    num_episodes = np.asarray(num_episodes)
    return num_episodes


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--board", dest="filename")
    parser.add_argument("--part", dest="subpart")
    parser.add_argument("--alpha", dest="alpha")
    parser.add_argument("--gamma", dest="gamma")
    parser.add_argument("--epsilon", dest="epsilon")
    parser.add_argument("--num_steps", dest="num_steps")
    args = parser.parse_args()
    alpha = float(args.alpha)
    gamma = float(args.gamma)
    epsilon = float(args.epsilon)
    num_steps = int(args.num_steps)

    averaged_num_episodes = np.zeros((num_steps))

    for random_seed_iter in range(10):
        num_episodes_iter = main(args.filename, args.subpart, alpha, gamma, epsilon, num_steps, random_seed_iter)
        averaged_num_episodes = averaged_num_episodes + num_episodes_iter
    
    averaged_num_episodes = averaged_num_episodes/10
    plt.plot(averaged_num_episodes)
    plt.xlabel("Time Steps")
    plt.ylabel("Number of Episodes")
    plt.show()