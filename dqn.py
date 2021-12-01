import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, deque
import torch.optim as optim
import copy
import random
import numpy as np

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT
from tqdm import tqdm
import mario_env_wrapper


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done')) # From CS885 A3Q1
# assumption: named tuples contain tensors

# PPO implementation heavily follows:
# https://github.com/seungeunrho/minimalRL and
# https://github.com/uvipen/Super-mario-bros-PPO-pytorch

class DQN(nn.Module):

    def __init__(self, n_inputs, n_actions, lr=1e-3, discount=0.95, start_epsilon=1.0, epsilon_max_step=20000, epsilon_min=0.01, buffer_size=10000, batch_size=64, 
                    target_update_period=100, device='cpu'):
        super(DQN, self).__init__()
        # Both Actor and Critic Network will share neural network parameters:
        # this requires that the loss function be written using both the
        # policy and value function
        self.n_inputs = n_inputs
        self.n_actions = n_actions
        
        # configuration of CNN from https://github.com/uvipen/Super-mario-bros-PPO-pytorch
        self.qnet = nn.Sequential(
            nn.Conv2d(n_inputs, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 6 * 6, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        
        self.target = copy.deepcopy(self.qnet)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        # for storing experiences
        self.experiences = deque(maxlen=buffer_size)
        self.batch_size = batch_size

        self.discount = discount

        # learning rate
        self.lr = lr

        # epsilon for epsilon greedy exploration
        self.epsilon = start_epsilon
        #self.decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.epsilon_steps = epsilon_max_step    
        self.target_update_period = target_update_period

        self.device = device
        self.to(device)

    def forward(self, x):
        return self.qnet(x)    


    def store_transition(self, state, action, reward, next_state, done):
        trans = Transition(state, action, reward, next_state, done)
        self.experiences.append(trans)

    def get_batch(self):
        
        minibatch = random.sample(self.experiences, self.batch_size)

        states, actions, rewards, next_states, done = [],[],[],[],[]
        for exp in minibatch:
            states.append(torch.tensor(exp.state, dtype=torch.float))
            actions.append(exp.action)
            rewards.append(exp.reward)
            next_states.append(torch.tensor(exp.next_state, dtype=torch.float))
            done.append(1 if exp.done else 0)

        # convert lists to pytorch tensors
        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        done = torch.tensor(done, dtype=torch.int).to(self.device)

        return states, actions, rewards, next_states, done

    # Update a target network using a source network
    def update_target(self):
        for tp, p in zip(self.target.parameters(), self.qnet.parameters()):
            tp.data.copy_(p.data)
    

    def update(self, epi):
        if len(self.experiences) < self.batch_size:
            return # not enough experiences in replay buffer
        
        states, actions, rewards, next_states, done = self.get_batch()

        qvals = self.qnet(states)
        qvals = qvals.gather(1, actions.view(-1,1)).squeeze()

        q2vals = torch.max(self.target(next_states), dim=1).values

        td_targets = rewards + self.discount * q2vals * (1-done)

        loss = nn.MSELoss()(td_targets.detach(), qvals)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if epi != 0 and epi % self.target_update_period == 0:
            self.update_target()

    def get_action(self, state):
        # epsilon greedy search
        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, self.n_actions)
        else:
            qvals = self.qnet(state.unsqueeze(0))
            action = torch.argmax(qvals, dim=1).item()

        # update epsilon
        self.epsilon = max(self.epsilon - (1 / self.epsilon_steps), self.epsilon_min)
        return action

def train(n_episodes=10000, print_interval=20, save_interval=0, dir='./dqn_models', render=False):

    device = 'cuda' if torch.cuda.is_available else 'cpu'

    # create environment
    env = mario_env_wrapper.create_mario_env()
    state_shape = env.observation_space.shape
    n_channels = state_shape[0]
    n_actions = env.action_space.n

    model = DQN(n_channels, n_actions, device=device)
    
    episode_scores = np.zeros(n_episodes)
    last_x_pos = np.zeros(n_episodes)
    cumulative_score = 0
    flags_captured = 0

    for epi in tqdm(range(n_episodes)):
        state = env.reset()
        if render: env.render()
        done = False
        score = 0

        while not done:
            # select action 
            state_tensor = torch.from_numpy(state.__array__()).to(device)
            action = model.get_action(state_tensor)    

            # execute action
            next_state, reward, done, info = env.step(action)
            if render: env.render()

            # store in replay buffer
            model.store_transition(state.__array__(), action, reward, next_state.__array__(), done)
            
            # update model
            model.update(epi)
            
            score += reward
            state = next_state

        episode_scores[epi] = score
        cumulative_score += score
        last_x_pos[epi] = info['x_pos']

        if info['flag_get']:
            flags_captured += 1


        # print statistics
        if print_interval != 0 and epi != 0 and epi % print_interval == 0:
            print(f'Episode {epi} score: {score}, Average score over {print_interval} episodes: {cumulative_score/print_interval}')
            cumulative_score = 0

        # save checkpoint
        if save_interval != 0 and epi != 0 and epi % save_interval == 0:
            path = "%s/dqn_%d.pt" % (dir, epi)
            print(f"Saving model to {path}")
            torch.save(model.state_dict, path)  
        

    env.close()
    # save final model
    path = "%s/dqn_final.pt" % (dir)
    torch.save(model.state_dict(), path)

    # save episode scores to csv
    np.savetxt("dqn_episode_rewards.csv", episode_scores, delimiter=",")

    # save across positions
    np.savetxt("dqn_episode_pos.csv", last_x_pos, delimiter=",")
        
def test_model(modelpath="./dqn/dqn_final.pt", n_iterations=10, render=True):
    device = 'cuda' if torch.cuda.is_available else 'cpu'

    # create environment
    env = mario_env_wrapper.create_mario_env()
    state_shape = env.observation_space.shape
    n_channels = state_shape[0]
    n_actions = env.action_space.n

    model = DQN(n_channels, n_actions, device=device)
    model.load_state_dict(torch.load(modelpath))
    
    
    scores = []
    for iter in range(n_iterations):
        state = env.reset()
        if render: env.render()
        done = False
        score = 0

        while not done:
            with torch.no_grad:
                # select action 
                state_tensor = torch.from_numpy(state.__array__()).to(device)
                action = model.get_action(state_tensor)    

                # execute action
                next_state, reward, done, info = env.step(action)
                if render: env.render()

                score += reward
                state = next_state
        
        scores.append(score)

    assert(len(scores) == n_iterations)
    print(f"[TEST] Average score over 10 episodes: {np.mean(np.array(scores))}")


def main():
    train(n_episodes=10000, render=False)
    test_model()

if __name__ == '__main__':
    main()
        