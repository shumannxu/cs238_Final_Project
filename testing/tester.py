import os
import numpy as np
import sys
import pandas as pd 
import cvxpy as cp 
import random 
import ast  # Import the ast module for literal_eval

# to denote s numerical value for terminal state 
TERMINAL_STATE_VALUE = 10000
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
            cumulative_scores[0] += self.payout_table[s, random_action]
            # (2)(3)(4)(5)
            for i in range(ACTION_SPACE):
                a = i  # 0,1,2,3
                cumulative_scores[i+1] = self.payout_table[s, a]
        return cumulative_scores / len(states_to_test)
    
"""
Returns None if inputfilepath does not exist
Returns numpy array with State, Action, Rewards, Next State columns 
Zero-indexes states, action, new states
"""
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

    ### Step 1: Create Expected Value/Payout Table via MLE ###

    # Create a cum payout table (divide to get EV later)
    cumulative_rewards_table = np.full((STATE_SPACE, ACTION_SPACE), 0) 
    # Count the occurrences of each state (start with the count 4 for laplace)
    visit_count_table = np.full((STATE_SPACE, ACTION_SPACE), 1)
    # count state appearances 
    visit_count_only_states = np.full((STATE_SPACE, 1), 1)

    # Count occurrences of each (s,a) pair for all files (zero-indexed)
    for dir in ["cleaned_2023_data/23_24_", "cleaned_2022_data/22_23_"]:
        for week_num in range(1, 22): 
            inputfilepath = 'data_cleaned/' + dir + 'week_' + str(week_num) + ".csv"

            # get data from each csv file 
            data = process_data(inputfilepath)
            # check if data is not none 
            if data is None:  continue 
            # traverse through each row in the file (s,a,r,s')
            for row_data in data: 
                (s,a,r,s_prime) = row_data
                # add rewards to the cumulative payout table
                # R(s,a)
                cumulative_rewards_table[s][a] += r 
                # update count of (s,a)
                # N(s,a) 
                visit_count_table[s][a] += 1
                # update visit count for the state
                visit_count_only_states[s] += 1
    
    # expected value = R(s,a) / N(s,a)  (with laplace)
    expected_value_table = cumulative_rewards_table / visit_count_table
    print(expected_value_table)
    print(np.count_nonzero(cumulative_rewards_table), "non-zero values")

    ### Step 3: Generate 100 states we hope to test (most frequently occurring states) ###
    # top_100_idx = np.argsort(visit_count_only_states)[-100:]
    top_200_idx = np.argpartition(-visit_count_only_states[:, 0], kth=200)[:200]
    state_to_test = top_200_idx
    print(top_200_idx)

    ### Step 4: Test the scores ###

    ### Get Q learning PolicyTable ###

    policy_table_fp = "results/trained_w_2023.csv"
    df = pd.read_csv(policy_table_fp, header=None)
    df = df.apply(lambda x: x - 1)
    policy_table = np.array(df)
    print(len(policy_table))

    # Q_learning
    Q_learning_win = 0
    total_played = 0
    result_list = []
    # play 30 times 
    for i in range(30):
        TestingInstance = policy_tester(expected_value_table)
        baseline_score = TestingInstance.test_baseline_policy(state_to_test)
        model_free_score = TestingInstance.test_policy(policy_table, state_to_test)
        # print("Baseline score:", baseline_score)
        # print("Q-Learning Score:", model_free_score)
        if model_free_score > baseline_score[0]:
            Q_learning_win += 1
        total_played += 1
        result_list.append([model_free_score])
    print("Result:", Q_learning_win, "times won by Q-learning,", total_played-Q_learning_win, "times won by random")
    print("Average Score by Q_learning:", np.average(result_list))






    







    
    

            





                


    




if __name__ == '__main__':
    main()


    

