import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class ForwardPredictionErrorModel():
    def __init__(self, state_space, action_space, alpha=0.01, alpha_decay=0.01, batch_size=64):
        self.state_space = state_space
        self.action_space = action_space
        self.alpha = alpha # learning rate
        self.alpha_decay = alpha_decay # learning rate decay (adam)
        self.batch_size = batch_size # for training
        self.memory = []
        self.init_model()

    def init_model(self):
        """
        given (s_t, a_t), predict s_t+1
        """
        self.model = Sequential()
        self.model.add(Dense(24,
            input_dim=self.state_space.shape[0]+1,
            activation='tanh'))
        self.model.add(Dense(48, activation='tanh'))
        self.model.add(Dense(self.state_space.shape[0], activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam(lr=self.alpha, decay=self.alpha_decay))

    def get_model_input(self, state, action):
        return np.hstack([state, np.array([action])[None,:]])

    def loss_function(self, y, yhat):
        return np.sum((y - yhat)**2, axis=1)
    
    def get_reward(self, state, action, next_state):
        next_state_hat = self.model(self.get_model_input(state, action))[0]
        return -self.loss_function(next_state, next_state_hat)

    def remember(self, state, action, next_state):
        self.memory.append((state, action, next_state))

    def replay(self, batch_size):
        x_batch, y_batch = [], []
        cur_batch_size = min(len(self.memory), batch_size)
        minibatch = random.sample(self.memory, cur_batch_size)

        for state, action, next_state in minibatch:
            x_batch.append(self.get_model_input(state, action))
            y_batch.append(action)

        self.model.fit(np.vstack(x_batch), np.array(y_batch), batch_size=cur_batch_size, verbose=0)

    def preprocess_state(self, state):
        return np.array(state)[None,:]

    def score(self):
        states = []
        actions = []
        next_states = []
        for i, (state, action, next_state) in enumerate(self.memory):
            states.append(state[0,:])
            actions.append(action)
            next_states.append(next_state[0,:])
        X = np.hstack([np.array(states), np.array(actions)[:,None]])
        y = np.array(next_states)
        yhat = self.model(X)
        return self.loss_function(y, yhat).mean()

    def run(self, env, nepisodes=200):
        scores = np.zeros((nepisodes,))
        for i in range(nepisodes):
            done = False
            state = self.preprocess_state(env.reset())
            while not done:
                action = random.choice(range(self.action_space.n))
                next_state, reward, done, info = env.step(action)
                next_state = self.preprocess_state(next_state)
                self.remember(state, action, next_state)
                
                state = next_state
            self.replay(self.batch_size)
            scores[i] = self.score()
        return scores
