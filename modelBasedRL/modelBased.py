import os
import numpy as np
import pandas as pd
import ast
import random

TERMINAL_STATE_VALUE = 10000
STATE_SPACE = 10001
ACTION_SPACE = 4
GAMMA = 0.95
VALUE_ITERATION_THRESHOLD = 0.1


def parse_data(inputfilepath):
    df = pd.read_csv(inputfilepath, sep=";")
    df["State"] = df["State"].apply(ast.literal_eval)
    df["Next_State"] = df["Next_State"].apply(ast.literal_eval)

    # Convert states to numerical values
    for i in range(len(df)):
        state = df.loc[i, "State"]
        next_state = df.loc[i, "Next_State"]

        # Handle terminal state separately
        if next_state[0] == "T":
            df.loc[i, "Next_State"] = TERMINAL_STATE_VALUE
        else:
            df.loc[i, "Next_State"] = state_to_numeric(next_state)

        df.loc[i, "State"] = state_to_numeric(state)

    df["State"] -= 1
    df["Next_State"] -= 1
    df["Action"] -= 1

    return df


def state_to_numeric(state):
    curDown, toGo, fp = (
        min(int(state[0]), 4),
        min(int(state[1]), 25),
        min(int(state[2]), 100),
    )
    assert curDown <= 4 and toGo <= 25 and fp <= 100
    return 100 * (toGo - 1) + 2500 * (curDown - 1) + (fp - 1)


def estimate_transition_and_reward_matrices(data, laplace_constant=1):
    transition_matrix = np.zeros((STATE_SPACE, ACTION_SPACE, STATE_SPACE))
    reward_matrix = np.zeros((STATE_SPACE, ACTION_SPACE))
    visit_count = np.zeros((STATE_SPACE, ACTION_SPACE))

    for state, action, reward, next_state in data:
        transition_matrix[state, action, next_state] += 1
        reward_matrix[state, action] += reward
        visit_count[state, action] += 1

    # Apply Laplace smoothing
    transition_matrix += laplace_constant
    visit_count += (
        STATE_SPACE * laplace_constant
    )  # Adjusted for each possible next state

    # Normalize to get probabilities and average rewards
    for s in range(STATE_SPACE):
        for a in range(ACTION_SPACE):
            transition_matrix[s, a, :] /= visit_count[s, a]
            reward_matrix[s, a] /= (
                visit_count[s, a] - (STATE_SPACE - 1) * laplace_constant
            )  # Adjusting the divisor for added constants

    return transition_matrix, reward_matrix


def value_iteration(transition_matrix, reward_matrix):
    value_table = np.zeros(STATE_SPACE)
    policy = np.zeros(STATE_SPACE)

    while True:
        value_table_prev = np.copy(value_table)
        for s in range(STATE_SPACE):
            U_value = reward_matrix[s, :] + GAMMA * np.sum(
                transition_matrix[s, :, :] * (value_table),
                axis=1,
            )
            value_table[s] = np.max(U_value)
            policy[s] = np.argmax(U_value) + 1

        print(np.max(np.abs(value_table - value_table_prev)))
        if np.max(np.abs(value_table - value_table_prev)) < VALUE_ITERATION_THRESHOLD:
            break

    return policy


def read_data_from_folders(folders):
    all_data = []
    count = 0
    for folder in folders:
        for filename in os.listdir(folder):
            if filename.endswith(".csv"):
                filepath = os.path.join(folder, filename)
                file_data = parse_data(filepath)
                all_data.append(file_data)
                count += 1
    return pd.concat(all_data, ignore_index=True).to_numpy()


def write_policy_file(filename, policy):
    with open(filename, "w") as f:
        for i in range(len(policy)):
            f.write(str(int(policy[i])) + "\n")


def main():
    outputfilename = "../results/output_test3.csv"

    # Folders containing the data
    folder1 = "../data_cleaned/cleaned_2022_data"
    folder2 = "../data_cleaned/cleaned_2023_data"

    # Read and combine data from both folders
    combined_data = read_data_from_folders([folder1, folder2])

    transition_matrix, reward_matrix = estimate_transition_and_reward_matrices(
        combined_data
    )
    optimal_policy = value_iteration(transition_matrix, reward_matrix)
    print("policy")
    # You can write this policy to a file or use it as needed
    write_policy_file(outputfilename, optimal_policy)


if __name__ == "__main__":
    main()
