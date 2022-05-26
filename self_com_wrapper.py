import gym


class SelfCommunicationWrapper(gym.Wrapper):
    def __init__(self, env, com_space):
        super(SelfCommunicationWrapper, self).__init__(env)
        self.com_space = com_space
        self.observation_space = gym.spaces.Dict({
            "env": self.env.observation_space,
            "com": self.com_space
        })
        self.action_space = gym.spaces.Dict({
            "env": env.action_space,
            "com": self.com_space
        })

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        com_action = 0 * self.com_space.sample()
        return self.observation(observation, com_action)

    def step(self, action: dict):
        obs, reward, done, info = self.env.step(action["env"])
        return self.observation(obs, action["com"]), reward, done, info

    def observation(self, obs, com_action):
        return {"env": obs, "com": com_action}