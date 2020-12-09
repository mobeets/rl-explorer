import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class DQNAgent():
    def __init__(self, state_space, action_space, epsilon=0.1, gamma=0.9, alpha=0.01, alpha_decay=0.01, batch_size=64):
        self.epsilon = epsilon # exploration probability
        self.gamma = gamma # discount factor
        self.alpha = alpha # learning rate
        self.alpha_decay = alpha_decay # learning rate decay (adam)
        self.batch_size = batch_size # for training DQN
        self.state_space = state_space
        self.action_space = action_space
        self.init_model()
        self.memory = []

    def init_model(self):
        self.model = Sequential()
        self.model.add(Dense(24, input_dim=self.state_space.shape[0], activation='tanh'))
        self.model.add(Dense(48, activation='tanh'))
        self.model.add(Dense(self.action_space.n, activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam(lr=self.alpha, decay=self.alpha_decay))

    def get_epsilon(self, i):
        return self.epsilon

    def sample_action(self, state, eps=0):
        if np.random.uniform(0,1) < eps:
            return np.random.choice(self.action_space.n)
        return np.argmax(self.get_q_value(state))

    def preprocess_state(self, state):
        return np.array(state)[None,:]

    def get_q_value(self, state):
        return np.array(self.model(state)[0])
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        x_batch, y_batch = [], []
        cur_batch_size = min(len(self.memory), batch_size)
        minibatch = random.sample(self.memory, cur_batch_size)

        for state, action, reward, next_state, done in minibatch:
            q_target = self.get_q_value(state)
            if done:
                q_target[action] = reward
            else:
                q_target[action] = reward + self.gamma*np.max(self.get_q_value(next_state))
            x_batch.append(state)
            y_batch.append(q_target)

        self.model.fit(np.vstack(x_batch), np.array(y_batch), batch_size=cur_batch_size, verbose=0)

    def run(self, env, nepisodes=200, fpemdl=None):
        scores = np.zeros((nepisodes,))
        for i in range(nepisodes):
            done = False
            state = self.preprocess_state(env.reset())
            while not done:
                action = self.sample_action(state, self.get_epsilon(i))
                next_state, reward, done, info = env.step(action)
                next_state = self.preprocess_state(next_state)
                if fpemdl is not None:
                    reward = fpemdl.get_reward(state, action, next_state)
                self.remember(state, action, reward, next_state, done)
                if fpemdl is not None:
                    fpemdl.remember(state, action, next_state)
                
                state = next_state
                scores[i] += reward
            self.replay(self.batch_size)
            if fpemdl is not None:
                fpemdl.replay(self.batch_size)
        return scores
