import numpy as np
import sys
import os 
import pandas as pd 
import scipy 
import cvxpy as cp 
import random 
import time 
from sklearn.neighbors import NearestNeighbors
import ast  # Import the ast module for literal_eval

# to denote s numerical value for terminal state 
TERMINAL_STATE_VALUE = 10000
# size of state space (+1 to accommodate for terminal state)
STATE_SPACE = 10001
# size of action space 
ACTION_SPACE = 4
# gamma value 
GAMMA = 0.9
# learning rate
LEARNING_RATE = 0.1
# number of passings 
NUMBER_OF_PASSES = 1

class QLearningMDP():
    # Action space A: {Pass, Run, kickFG, Punt}
    def __init__(self, action_space, state_space, gamma, Q, 
                 TrackingTable, learning_rate, data, curdown_togo_fp_table):
        # action space 
        self.action_space = action_space
        # state space 
        self.state_space = state_space
        # discount rate 
        self.gamma = gamma
        # action value function Q / Q table
        self.Q = Q
        # track which s,a pairs we have explored 
        self.TrackingTable = TrackingTable
        # learning rate
        self.learning_rate = learning_rate  
        # data 
        self.data = data 
        # ith row represents state i's curdown, togo, fp
        self.curdown_togo_fp_table = curdown_togo_fp_table

    def update(self, s, a, r, s_prime):
        self.Q[s, a] += self.learning_rate * (r + (self.gamma * np.max(self.Q[s_prime])) - self.Q[s, a])    

    """
    QLearning algorithm starts off with traversing all explored data/transitions
    from files to update Q values. Then, it uses different approaches for each filetype:

    Returns the argmax of Q[s] for each state s, where the 
    Q-value of unobserved state, action pairs were approximated using 
    K-Nearest Neighbors (k=5). 
    """
    def QLearning(self):
        # iterate through every row in the data 
        for row_data in self.data:
            # get transition data
            (s, a, r, s_prime) = row_data
            # fill the Q-table
            self.update(s, a, r, s_prime)
            # update the tracking table 
            self.TrackingTable[s, a] = 1

        # Approximation: Nearest Neighbour
        k = 10
        neighbors_model = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(self.curdown_togo_fp_table)
        # calculate nearest neighbors for all states 
        d, ind = neighbors_model.kneighbors(self.curdown_togo_fp_table)  # ith row is ith state's [curdown, togo, fp]
        # Approximate the Q-value for any s,a pair that didn't appear 
        for s in range(self.state_space):  
            for a in range(self.action_space):  
                if self.TrackingTable[s][a] == 0: 
                    for i in range(1, k):
                        neighbor_index = ind[s][i]
                        self.Q[s,a] += self.Q[neighbor_index, a]
                    self.Q[s,a] = self.Q[s,a] / (len(ind[s]) - 1)

        # compute optimal policy
        policy = np.zeros(self.state_space)
        for s in range(self.state_space): 
            policy[s] = np.argmax(self.Q[s]) + 1  # one-index 

        return policy 

def write_policy_file(filename, policy):
    with open(filename, 'w') as f:
        for i in range(len(policy)):
            f.write(str(int(policy[i])) + "\n")

def process_data(inputfilepath):
    # check if path exists
    if not os.path.exists(inputfilepath):
        return None 

    # load data into numpy array
    df = pd.read_csv(inputfilepath, sep=";")
    df["State"] = df["State"].apply(ast.literal_eval)
    df["Next_State"] = df["Next_State"].apply(ast.literal_eval)

    # Change states to numerical state 
    for i in range(len(df["State"])):
        state_list, state_list_prime = df["State"][i], df["Next_State"][i]

        # State: convert to numerical state value 
        curDown, toGo, fp = int(state_list[0]), int(state_list[1]), int(state_list[2])
        # handle edge case, where toGo is greater than 25, curDown is 5+, fp is 99+
        toGo = min(25, toGo)
        curDown = min(4, curDown)
        fp = min(99, fp)
        # update 
        df.loc[i, "State"] = 100 * (toGo - 1) + 2500 * (curDown - 1) + (fp - 1)

        # State prime: account for terminal case 
        if state_list_prime[0] == "T":
            df.loc[i, "Next_State"] = TERMINAL_STATE_VALUE
        else:
            curDown, toGo, fp = int(state_list_prime[0]), int(state_list_prime[1]), int(state_list_prime[2])
            # handle edge case, where toGo is greater than 25, curDown is 5+, fp is 99+
            toGo = min(25, toGo)
            curDown = min(4, curDown)
            fp = min(99, fp)
            # update 
            df.loc[i, "Next_State"] = 100 * (toGo - 1) + 2500 * (curDown - 1) + (fp - 1)

    # zero index the data 
    df["State"] -= 1
    df["Next_State"] -= 1
    df["Action"] -= 1

    # convert to numpy for efficiency 
    data = df.to_numpy()

    return data

def main():

    # Step 1: INITIALIZE 
    gamma, action_space, state_space, rate = GAMMA, ACTION_SPACE, STATE_SPACE, LEARNING_RATE
    # initialize Q table
    Q = np.zeros((state_space, action_space))
    # initialize table to track which (s,a) has been explored
    TrackingTable = np.zeros((state_space, action_space))
    # store the optimal policy 
    optimal_policy = None
    # initialize table for nearest neighbour 
    curdown_togo_fp = []
    for num_state in range(state_space):
        curDown = num_state // 2500 + 1
        toGo = (num_state % 2500) // 100 + 1
        fp = (num_state % 2500) % 100 + 1
        curdown_togo_fp.append([curDown, toGo, fp])
    curdown_togo_fp_table = np.array(curdown_togo_fp)
    start = time.time()

    # Step 2: ITERATE DATA & UPDATE Q TABLE
    # iterate through 2023 data
    for dir in ["cleaned_2023_data/23_24_", "cleaned_2022_data/22_23_"]:
        for week_num in range(1, 22): 
            # inputfilepath = "data_cleaned/cleaned_2023_data/23_24_week_" + str(week_num) + ".csv"
            inputfilepath = 'data_cleaned/' + dir + 'week_' + str(week_num) + ".csv"
            outputfilename = "/Users/elychen/CS238/cs238_Final_Project/results/trained_w_2023.csv"

            data = process_data(inputfilepath)

            if data is None: continue
            
            # Step 3: train & obtain optimal policy 
            for n in range(NUMBER_OF_PASSES):
                QLearningInstance = QLearningMDP(action_space, state_space, gamma, Q, 
                        TrackingTable, rate, data, curdown_togo_fp_table)
                optimal_policy = QLearningInstance.QLearning()
            end = time.time()
            print(end - start, "seconds")
            print(inputfilepath)

    # Write Policy File 
    write_policy_file(outputfilename, optimal_policy)

if __name__ == '__main__':
    main()