
class QAgent():
    def __init__(self, nbins, state_space, action_space, epsilon=0.1, gamma=0.9, alpha=0.1):
        self.epsilon = epsilon # exploration probability
        self.gamma = gamma # discount factor
        self.alpha = alpha # learning rate
        self.nbins = nbins # for discretizing state space
        self.state_space = state_space
        self.action_space = action_space
        self.q_table = {}

    def preprocess_state(self, state):
        env_low = self.state_space.low
        env_high = self.state_space.high

        # truncate bins where range is too large
        env_low[env_low < -4] = -4
        env_high[env_high > 4] = 4

        states = []
        for i in range(len(state)):
            env_dx = (env_high[i] - env_low[i])/self.nbins[i]
            x = int((state[i] - env_low[i])/env_dx)
            if x >= self.nbins[i]:
                x -= 1
            states.append(x)
        return tuple(states)

    def get_epsilon(self, i):
        return self.epsilon

    def sample_action(self, state, eps=0):
        if np.random.uniform(0,1) < eps:
            return np.random.choice(self.action_space.n)
        return np.argmax(self.get_q_value(state))

    def get_q_value(self, state, action=None):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space.n)
        if action is None:
            return self.q_table[state]
        else:
            return self.q_table[state][action]
    
    def update(self, state, action, reward, next_state, done):
        q_delta = reward - self.get_q_value(state, action)
        if not done:
            q_delta += self.gamma*np.max(self.get_q_value(next_state))
        self.q_table[state][action] += self.alpha*q_delta

    def run(self, env, nepisodes=200):
        scores = np.zeros((nepisodes,))
        for i in range(nepisodes):
            done = False
            state = self.preprocess_state(env.reset())
            while not done:
                action = self.sample_action(state, self.get_epsilon(i))
                next_state, reward, done, info = env.step(action) 
                next_state = self.preprocess_state(next_state)
                self.update(state, action, reward, next_state, done)
                
                state = next_state
                scores[i] += reward
        return scores
