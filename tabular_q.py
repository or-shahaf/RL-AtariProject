import gym
import numpy as np

# Load environment
env = gym.make('FrozenLake-v0')

# Implement Q-Table learning algorithm
# Initialize table with all zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])
# Set learning parameters
lr = .8
y = .95
eps = 0.47
num_episodes = 2000
# create lists to contain total rewards and steps per episode
# jList = []
rList = []
for i in range(num_episodes):

    # Reset environment and get first new observation
    s = env.reset()
    rAll = 0  # Total reward during current episode
    d = False
    j = 0
    # The Q-Table learning algorithm
    while j < 99:
        j += 1
        # TODO: Implement Q-Learning
        # 1. Choose an action by greedily (with noise) picking from Q table
        if np.random.rand(1)[0] < eps ** (int(i / 100)):
            a = np.random.choice(range(len(Q[s])))
        else:
            a = np.argmax(Q[s])
        # 2. Get new state and reward from environment
        next_s, r, done, _ = env.step(a)
        # 3. Update Q-Table with new knowledge
        Q[s][a] = Q[s][a] + lr * (r + y * max(Q[next_s]) - Q[s][a])
        # 4. Update total reward
        rAll += r
        # 5. Update episode if we reached the Goal State
        s = next_s
        if done:
            break

    rList.append(rAll)

# Reports
print("Score over time: " + str(sum(rList) / num_episodes))
print("Final Q-Table Values")
print(Q)
