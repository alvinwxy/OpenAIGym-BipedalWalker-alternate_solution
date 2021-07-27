import matplotlib.pyplot as plt
import numpy as np


def plot(x, y, filename=None):
    fig = plt.figure()
    ax = fig.add_subplot()

    N = len(y)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(y[max(0, t - 100):(t + 1)])   # mean reward for 100 episode

    ax.plot(x, y, color="C0")
    ax.set_xlabel("Episode", color="C0")
    ax.set_ylabel("Episode reward", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    if filename is not None:
        plt.savefig(filename)

    plt.show()


class ReplayBuffer:
    def __init__(self, max_size=5e5):
        self.buffer = []
        self.max_size = int(max_size)
        self.size = 0

    def add(self, transition):
        self.size += 1
        # transition is tuple of (state, action, reward, next_state, done)
        self.buffer.append(transition)

    def sample(self, batch_size):
        # delete 1/5th of the buffer when full
        if self.size > self.max_size:
            del self.buffer[0:int(self.size / 5)]
            self.size = len(self.buffer)

        index = np.random.randint(0, len(self.buffer), size=batch_size)
        state, action, reward, state_, done = [], [], [], [], []

        for i in index:
            s, a, r, s_, d = self.buffer[i]
            state.append(np.array(s, copy=False))
            action.append(np.array(a, copy=False))
            reward.append(np.array(r, copy=False))
            state_.append(np.array(s_, copy=False))
            done.append(np.array(d, copy=False))

        return np.array(state), np.array(action), np.array(reward), np.array(state_), np.array(done)
