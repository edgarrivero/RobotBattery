# import sys
# import gym_environments
# import gym
# from agent import TwoArmedBandit

# num_iterations = 100 if len(sys.argv) < 2 else int(sys.argv[1])
# version = "v0" if len(sys.argv) < 3 else sys.argv[2]

# env = gym.make(f"TwoArmedBandit-{version}")
# agent = TwoArmedBandit(0.1)

# env.reset(options={'delay': 1})

# for iteration in range(num_iterations):
#     action = agent.get_action("random")
#     _, reward, _, _, _ = env.step(action)
#     agent.update(action, reward)
#     agent.render()

# env.close()

import gym
import gym_environments
from agent import QLearning

#RobotBattery-v0, Taxi-v3, FrozenLake-v1, RobotMaze-v0
ENVIRONMENT = 'RobotBattery-v0'

def train(env, agent, episodes):
    for _ in range(episodes):
        observation, _ = env.reset()
        terminated, truncated = False, False
        while not (terminated or truncated):
            action = agent.get_action(observation, 'epsilon-greedy')
            new_observation, reward, terminated, truncated, _ = env.step(
                action)
            agent.update(
                observation,
                action,
                new_observation,
                reward,
                terminated)
            observation = new_observation

def play(env, agent):
    observation, _ = env.reset()
    terminated, truncated = False, False
    while not (terminated or truncated):
        action = agent.get_action(observation, 'greedy')
        observation, _, terminated, truncated, _ = env.step(action)
        env.render()

if __name__ == "__main__":

    env = gym.make(ENVIRONMENT)
    agent = QLearning(
        env.observation_space.n, 
        env.action_space.n, 
        alpha=0.1, 
        gamma=0.9, 
        epsilon=0.1)

    train(env, agent, episodes=10)
    agent.render()

    env = gym.make(ENVIRONMENT, render_mode="human")
    play(env, agent)

env.close()