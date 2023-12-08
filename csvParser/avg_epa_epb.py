import os
import numpy as np
import pandas as pd 
import random 
import ast  # Import the ast module for literal_eval

# number of states to test (Assumption: 1 game = 100 states)
NUMBER_OF_STATES_TO_TEST = 100 
# number of games to test 
NUMBER_OF_GAMES_TO_PLAY = 10000
# to denote s numerical value for terminal state 
TERMINAL_STATE_VALUE = 10000
# size of state space (+1 to accommodate for terminal state)
STATE_SPACE = 10001
# size of action space 
ACTION_SPACE = 4

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

# Computes the Average EPA-EPB value of SF 49ers in the past 2 seasons 
def main():
    avg_epa_minus_epb = 0
    data_count = 0
    # Count occurrences of each (s,a) pair for all files (zero-indexed)
    for dir in ["cleaned_2023_data/23_24_", "cleaned_2022_data/22_23_"]:
        for week_num in range(1, 22): 
            inputfilepath = 'data_cleaned/' + dir + 'week_' + str(week_num) + ".csv"

            # get data from each csv file 
            data = process_data(inputfilepath)

            if data is None:  continue 

            for row_data in data:
                (s,a,r,s_prime) = row_data
                avg_epa_minus_epb += r
                data_count += 1

    print("Data Count:", data_count)
    print("Average EPA - EPB:", avg_epa_minus_epb / data_count)

if __name__ == '__main__':
    main()

