import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable


# Load environment
env = gym.make('FrozenLake-v0')

# Define the neural network mapping 16x1 one hot vector to a vector of 4 Q values
# and training loss
# TODO: define network, loss and optimiser(use learning rate of 0.1).


class Net(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        return out


# Net, Loss and Optimizer
net = Net(env.observation_space.n, env.action_space.n)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

# Implement Q-Network learning algorithm

# Set learning parameters
y = .99
e = 0.1
num_episodes = 2000
# create lists to contain total rewards and steps per episode
jList = []
rList = []

for i in range(num_episodes):
    # Reset environment and get first new observation
    s = env.reset()
    rAll = 0
    d = False
    j = 0
    # The Q-Network
    while j < 99:
        j += 1
        # 1. Choose an action greedily from the Q-network
        #    (run the network for current state and choose the action with the maxQ)
        # TODO: Implement Step 1
        Q = net(Variable(torch.eye(env.observation_space.n)[s]))
        a = int(Q.max(0)[1])

        # 2. A chance of e to perform random action
        if np.random.rand(1) < e:
            a = env.action_space.sample()

        # 3. Get new state(mark as s1) and reward(mark as r) from environment
        s1, r, d, _ = env.step(a)

        # 4. Obtain the Q'(mark as Q1) values by feeding the new state through our network
        # TODO: Implement Step 4
        Q1 = net(Variable(torch.eye(env.observation_space.n)[s1]))

        # 5. Obtain maxQ' and set our target value for chosen action using the bellman equation.
        # TODO: Implement Step 5
        if d:
            maxQ1 = 0
        else:
            maxQ1 = float(Q1.max())
        Qtarget = Variable(Q.clone().data)
        Qtarget[a] = r + y * maxQ1
        # 6. Train the network using target and predicted Q values (model.zero(), forward, backward, optim.step)
        # TODO: Implement Step 6
        optimizer.zero_grad()
        loss = criterion(Q, Qtarget)
        loss.backward()
        optimizer.step()

        rAll += r
        s = s1
        if d:
            # Reduce chance of random action as we train the model.
            e = 1. / ((i / 50) + 10)
            break
    jList.append(j)
    rList.append(rAll)

# Reports
print("Score over time: " + str(sum(rList)/num_episodes))
