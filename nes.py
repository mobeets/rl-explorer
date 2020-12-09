import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam

class NESAgent():
    def __init__(self, state_space, action_space, npop=10, sigma=0.1, gamma=0.9, alpha=0.01, alpha_decay=0.01):
        self.npop = npop # population size
        self.sigma = sigma # variance of search noise
        self.gamma = gamma # discount factor
        self.alpha = alpha # learning rate
        self.alpha_decay = alpha_decay # learning rate decay (adam)
        self.state_space = state_space
        self.action_space = action_space
        self.init_model()

    def init_model_flat(self):
        self.model = Sequential()
        self.model.add(Dense(24,
            input_dim=self.state_space.shape[0], activation='tanh'))
        self.model.add(Dense(48, activation='tanh'))
        self.model.add(Dense(self.action_space.n, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy',
            optimizer=Adam(lr=self.alpha, decay=self.alpha_decay))

    def init_model_2D(self):
        self.model = Sequential()
        self.model.add(Conv2D(32,
            input_shape=self.state_space.shape[:2],
            kernel_size=3, activation='relu'))
        self.model.add(MaxPooling2D(pool_size=2))
        self.model.add(Conv2D(64,
            kernel_size=3, activation='relu'))
        self.model.add(MaxPooling2D(pool_size=2))
        self.model.add(Flatten())
        self.model.add(Dense(self.action_space.n, activation='linear'))
        self.model.compile(loss='categorical_crossentropy',
            optimizer=Adam(lr=self.alpha, decay=self.alpha_decay))

    def init_model(self):
        # get model weights, count n per layer, then flatten
        # self.init_model_flat()
        self.init_model_2D()
        self.mu = self.model.get_weights()
        self.mu_shapes = [ws.shape for ws in self.mu]
        self.mu = np.hstack([ws.flatten() for ws in self.mu])
        self.nparams = self.mu.size

    def preprocess_state(self, state):
        if len(np.array(state).shape) == 3:
            return state[:,:,0][None,None,:]
        return np.array(state)[None,:]

    def set_model_weights(self, mu):
        c = 0
        ws = []
        for dims in self.mu_shapes:
            n = np.prod(dims)
            ws.append(mu[c:(c+n)].reshape(dims))
            c += n
        self.model.set_weights(ws)

    def get_sigma(self, t):
        return self.sigma

    def sample_action(self, state):
        return np.argmax(self.model(state)[0])

    def update(self, fitnesses, noise, t):
        # if no variance in fitness, do nothing
        if np.sum((np.std(fitnesses) - 0.0)**2) < 1e-10:
            print("No population variance in epoch {}. Skipping.".format(t))
            return
        
        # anneal learning rate
        self.alpha *= (1. / (1. + self.alpha_decay * t))

        # take step towards weighted mean of noise
        A = (fitnesses - np.mean(fitnesses)) / np.std(fitnesses)
        self.mu += self.alpha/(self.npop*self.get_sigma(t)) * np.dot(noise.T, A)
        self.set_model_weights(self.mu)

    def run(self, env, nepisodes=200):
        scores = np.zeros((nepisodes,))
        noise = np.random.randn(self.npop, self.nparams)
        fitnesses = np.zeros(self.npop)
        c = 0
        ngens = 0
        for i in range(nepisodes):
            done = False

            # set current weights
            z = self.mu + self.get_sigma(ngens)*noise[c]
            self.set_model_weights(z)

            state = self.preprocess_state(env.reset())
            while not done:
                action = self.sample_action(state)
                state, reward, done, info = env.step(action)
                state = self.preprocess_state(state)
                scores[i] += reward

            # fitness of noise is total reward at end of episode
            fitnesses[c] = scores[i]
            c += 1

            # have a complete population, so update params
            if i % self.npop == self.npop-1:
                self.update(fitnesses, noise, ngens)
                noise = np.random.randn(self.npop, self.nparams)
                fitnesses = np.zeros(self.npop)
                c = 0
                ngens += 1
        return scores
