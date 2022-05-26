# gym_communication_wrapper
This wrapper adds an "inner state" created by the agent which is propagated to the next step.

## What does this communication concept solve?
Instead of using RNN (LSTM, GRU, etc...), the agent sends a message to its 1-step future self, similar to the hidden states in RNNs.

## How does it work conceptually?
The action is split into 2 categories:
1. The physical action- the original action in the environment.
2. The communication action (a message) - an action that will be a part of the next observation.
The physical action is being send to the environment, while the communication action, unaltered, is propagated to the next obsevation.

## Under the hood
The wrapper changes the structure of the observation and the action to a dictionary with the 2 keys/categories as mentioned above:
```
observation = {"env": env_observation, "com": last_com_action}
action = {"env": env_action, "com": new_com_action}
```
You should take this into account when dealing with this new spaces.

## How to use the communication wrapper?
In every environment we have observation_space and action_space. Now we are adding a communication space (com_space) that can be continuous or discrete.
### Example:
Continuous communication space with 8 channels (can be seen as 8 words per sentence):
```
com_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(com_channels,), dtype=np.float32)
```
Discrete communication space with 8 channels and bandwidth of 5 (can be seen as 8 words in sentence with a vocabulary of 5 words in total):
```
com_space = gym.spaces.MultiDiscrete(8*[5])
```
Now that we have our com_space from above, we can wrap our environment (env) to contain and use it:
```
env = SelfCommunicationWrapper(gym.make('CartPole-v1'), com_space)
```
That's it! Now your agent can have a "line of thought" and send messages to its future self!

## Future work
Since communication has a huge role in multi-agent reinforcement learning, this wrapper will be extended to cross-agents message communication. Communication between agents couple them together and hopefully a new language will emerge.
