# gym_communication_wrapper
This wrapper adds an "inner state" created by the agent which is propagated to the next step.

## What does this communication concept solve?
Instead of using RNN (LSTM, GRU, etc...), the agent sends a message to its 1-step future self, similar to the hidden states in RNNs.

## How does it work conceptually?
The action is split into 2 categories:
1. The physical action- the original action in the environment.
2. The communication action - an action that will be a part of the next observation.
The physical action is being send to the environment, while the communication action, unaltered, is propagated to the next obsevation.

## Under the hood
### Spaces
The wrapper splits the observation_space and the action_space into a dictionary with the 2 categories from above:
```
  self.observation_space = Dict({
      "env": self.env.observation_space,
      "com": self.com_space
  })
  self.action_space = Dict({
      "env": env.action_space,
      "com": self.com_space
  })
```
### 

## How to use the communication wrapper?
In every environment we have observation_space and action_space. Now we are adding a communication space (com_space) that can be continuous or discrete.
### Examples:
Continuous communication space with 8 channels (can be seen as 8 words per sentence):
```
com_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(com_channels,), dtype=np.float32)
```
Discrete communication space with 8 channels and bandwidth of 5 (can be seen as 8 words in sentence with a vocabulary of 5 words in total):
```
com_space = gym.spaces.MultiDiscrete(8*[5])
```
Now that we have our com_space, we can wrap our environment to contain and use it:
```

