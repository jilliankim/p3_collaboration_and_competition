### Learning Algorithm

In this project I trained two agents to play table tennis. For this environment I used Multi-Agent Deep Deterministic Policy Gradient (MADDPG) algorithm and the neural network for both Actor and Critic has two hidden layers with 256 nodes each. Here I trained each agent seperately using the same Replay Buffer so that each agent learns from other agent's experiences. When an epsilon decay of 0.99 was used the environment solved in 896 episodes but with a learning rate decay of 0.9, the environment solves in 304 episodes.

Also changed training and updating of the target network only after adding the last agent's environment variables to memory. This way we don't run into the risk of updating the target network frequently.

DDPG was described in one of the papers in the resources in the curriculum, and the pseudo code is in the image below:

![Alt text](/DDPG.png?raw=true "Plot of Results per Episode")

Hyperparmeters used:

```
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-2              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
```

### Plot of Rewards

Below are the results and a plot of rewards per episode is included to illustrate that the agents get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents).

![Alt text](/p3_output.png?raw=true "Plot of Results per Episode")

There were few attempts where I included batch normalization and were all unsuccesful, thus removing batch normalization was the key to success.

### Ideas for Future Work

I would like to apply the same algorithm to train agents to play another sports, more complicated with more players (hockey, soccer, etc) because both agents are able to learn its own reward functions, this could be possible.
