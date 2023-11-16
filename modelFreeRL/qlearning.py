import numpy as np

class QLearning:
    def __init__(self, num_states, num_actions, data, discount=.99):
        self.data = data
        self.num_states = num_states
        self.num_actions = num_actions
        self.Q = np.zeros((self.num_states, self.num_actions))
        self.policy = np.zeros(self.num_states)
        self.discount = discount

    def Q_learning(self):
        with open(self.data) as f:
            self._read_data_and_update_Q(f)
            self._update_policy()

    def _read_data_and_update_Q(self, file):
        file.readline()
        for row in file:
            s, a, r, sp = map(int, row.split(','))
            s -= 1
            a -= 1
            sp -= 1
            
            self.Q[s, a] += 0.1 * (r + self.discount * self.Q[sp, a] - self.Q[s, a])

    def _update_policy(self):
        for i in range(self.num_states):
            if not np.any(self.Q[i]):
                self.policy[i] = np.random.randint(1, self.num_actions + 1)
            else:
                self.policy[i] = np.argmax(self.Q[i]) + 1
