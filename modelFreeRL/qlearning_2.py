import numpy as np
import sys
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
GAMMA = 0.95
# learning rate
LEARNING_RATE = 0.1
# number of passings 
NUMBER_OF_PASSES = 10


class QLearningMDP():
    # Action space A: {kickFG, Pass, Run, Punt}
    def __init__(self, action_space, state_space, gamma, Q, 
                 TrackingTable, learning_rate, data):
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
        
        # compute optimal policy
        policy = np.zeros(self.state_space)
        for s in range(self.state_space): 
            policy[s] = np.argmax(self.Q[s]) + 1  # one-index 

        return policy 

def write_policy_file(filename, policy):
    with open(filename, 'w') as f:
        for i in range(len(policy)):
            f.write(str(int(policy[i])) + "\n")

def main():
    
    inputfilepath = "test.csv"

    # load data into numpy array
    df = pd.read_csv("/Users/elychen/CS238/cs238_Final_Project/modelFreeRL/test.csv", sep=";")
    df["State"] = df["State"].apply(ast.literal_eval)
    df["Next_State"] = df["Next_State"].apply(ast.literal_eval)
    

    # Change states to numerical state 
    for i in range(len(df["State"])):
        state_list, state_list_prime = df["State"][i], df["Next_State"][i]

        # State: convert to numerical state value 
        curDown, toGo, fp = int(state_list[0]), int(state_list[1]), int(state_list[2])
        df.loc[i, "State"] = 100 * (toGo - 1) + 2500 * (curDown - 1) + (fp - 1)

        # State prime: account for terminal case 
        if state_list_prime[0] == "T":
            df.loc[i, "Next_State"] = TERMINAL_STATE_VALUE
        else:
            curDown, toGo, fp = int(state_list_prime[0]), int(state_list_prime[1]), int(state_list_prime[2])
            df.loc[i, "Next_State"] = 100 * (toGo - 1) + 2500 * (curDown - 1) + (fp - 1)

    df["State"] -= 1
    df["Next_State"] -= 1
    df["Action"] -= 1

    # conver to numpy for efficiency 
    data = df.to_numpy()
    # print(data)
    # initiate key vars
    gamma, action_space, state_space, rate = GAMMA, ACTION_SPACE, STATE_SPACE, LEARNING_RATE
    # initialize Q table
    Q = np.zeros((state_space, action_space))
    # initialize table to track which (s,a) has been explored
    print(Q)
    TrackingTable = np.zeros((state_space, action_space))

    start = time.time()

    for n in range(NUMBER_OF_PASSES):
        QLearningInstance = QLearningMDP(action_space, state_space, gamma, Q, 
                 TrackingTable, rate, data)
        policy = QLearningInstance.QLearning()
    end = time.time()
    outputfilename = "/Users/elychen/CS238/cs238_Final_Project/results/output_test2.csv"
    write_policy_file(outputfilename, policy)




if __name__ == '__main__':
    main()