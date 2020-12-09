import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import logger as gymlogger
gymlogger.set_level(40) # error only

from q_model import QAgent
from dqn_model import DQNAgent
from fpe import ForwardPredictionErrorModel

def main():
    env = gym.make('CartPole-v0')

    # nbins = (2, 2, 10, 10)
    # scores = QAgent(nbins, env.observation_space, env.action_space).run(env)
    
    fmdl = ForwardPredictionErrorModel(env.observation_space, env.action_space)

    scores = DQNAgent(env.observation_space, env.action_space).run(env)
    plt.plot(scores)
    plt.show()

if __name__ == '__main__':
    main()
