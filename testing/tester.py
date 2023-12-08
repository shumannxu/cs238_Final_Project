import os
import numpy as np
import sys
import pandas as pd 
import cvxpy as cp 
import random 
import ast  # Import the ast module for literal_eval

# size of state space (+1 to accommodate for terminal state)
STATE_SPACE = 10001
# size of action space 
ACTION_SPACE = 4

class policy_tester():
    def __init__(self, payout_table):
        # payout table is a numpy array, where (s,a) = expected payout of state, action pair
        self.payout_table = payout_table

    """
    policy table: policy pi to test
    states_to_test: numerical states to feed to policy table & test

    ASSUMES ZERO-INDEXING FOR STATES AND ACTIONS 

    returns average score (average EPA-EPB)
    """
    def test_policy(self, policy_table, states_to_test):
        cumulative_score = 0
        # test each numerical state
        for s in states_to_test:
            # generate policy's action given the state
            a = policy_table[s]
            # generate score for this specific (s,a)
            score = self.payout_table[s][a]
            cumulative_score += score 

        expected_payout = cumulative_score / len(states_to_test)
        return expected_payout

    """
    Returns the expected payout result of 5 baseline strategies (in the following order):
    (1) Play Random 
    (2) Always Play Action 1
    (3) Always Play Action 2
    (4) Always Play Action 3
    (5) Always Play Action 4
    """
    def test_baseline_policy(self, states_to_test):
        cumulative_scores = np.array([0,0,0,0,0])
        for s in states_to_test:
            # (1) play random 
            random_action = random.randint(0, ACTION_SPACE - 1)
            cumulative_scores[0] += self.payout_table[s][random_action]
            # (2)(3)(4)(5)
            for i in range(ACTION_SPACE):
                a = i + 1
                cumulative_scores[i+1] = self.payout_table[s][a]
        return cumulative_scores / len(states_to_test)

def main():
    # Get Q learning PolicyTable
    policy_table_fp = "results/output_test2.csv"
    df = pd.read_csv(policy_table_fp)
    # zero-index action
    df -= 1
    policy_table = np.array(df)
    print(policy_table)

    # WIP

if __name__ == '__main__':
    main()


    

