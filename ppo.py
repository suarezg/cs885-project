import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
import torch.optim as optim
from torch.distributions import Categorical

from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT
import mario_env_wrapper
from tqdm import tqdm
import numpy as np


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done', 'prob_action')) # From CS885 A3Q1
# assumption: named tuples contain tensors

# PPO implementation heavily follows:
# https://github.com/seungeunrho/minimalRL and
# https://github.com/uvipen/Super-mario-bros-PPO-pytorch
# 

class PPO(nn.Module):

    def __init__(self, n_inputs, n_actions, lr=1e-3, gamma=0.95, lam=0.95, eps=0.2, c1=0.5, c2=0.01, device='cpu'):
        super(PPO, self).__init__()
        # Both Actor and Critic Network will share neural network parameters:
        # this requires that the loss function be written using both the
        # policy and value function

         # architecture of CNN from https://github.com/uvipen/Super-mario-bros-PPO-pytorch
        self.actor = nn.Sequential(
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
            nn.Linear(512, n_actions),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
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
            nn.Linear(512, 1)
        )
            
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # for storing experiences
        self.experiences = []

        # learning rate
        self.lr = lr

        # parameters for advantage estimator
        self.gamma = gamma
        self.lam = lam

        # parameter for clipped surrogate objective
        self.eps = eps

        # parameters for combined loss function
        self.c1 = c1
        self.c2 = c2

        self.device = device
        self.to(device)

    

    def actor_pass(self, x):
        return self.actor(x)

    def critic_pass(self, x):
        return self.critic(x)

    def forward(self,x):
        raise NotImplementedError
        

    def store_transition(self, state, action, reward, next_state, done, action_probs):
        trans = Transition(state, action, reward, next_state, done, action_probs)
        self.experiences.append(trans)

    def get_batch(self):
        
        states, actions, rewards, next_states, done, action_probs = [],[],[],[],[],[]
        for exp in self.experiences:
            states.append(torch.tensor(exp.state, dtype=torch.float))
            actions.append(exp.action)
            rewards.append(exp.reward)
            next_states.append(torch.tensor(exp.next_state, dtype=torch.float))
            done.append(1 if exp.done else 0)
            action_probs.append(exp.prob_action)

        # convert lists to pytorch tensors
        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        done = torch.tensor(done, dtype=torch.int).to(self.device)
        action_probs = torch.tensor(action_probs).to(self.device)

        # clear experience buffer
        self.experiences = []

        # TODO dimension check?

        return states, actions, rewards, next_states, done, action_probs

    def update(self, k_epochs=5):
        states, actions, rewards, next_states, done, old_probs = self.get_batch()

        for epoch in range(k_epochs):
            V = self.critic_pass(states).squeeze()
            V_next = self.critic_pass(next_states).squeeze()
            
            target = rewards + self.gamma * V_next * (1 - done)
            delta = target - V
            delta = delta.cpu().detach().numpy()

            # calculate advantage estimates, starting with A_T up to A_1 (reverse order)
            advantage_list = []
            advantage = 0
            for d in delta[::-1]:
                advantage = self.gamma * self.lam * advantage + d
                advantage_list.append(advantage)
            advantage_list.reverse()
            advantages = torch.tensor(advantage_list, dtype=torch.float).to(self.device)

            # create loss function
            probs = self.actor_pass(states)
            prob_action = probs.gather(1,actions.view(-1,1)).squeeze()
            ratio = torch.exp(torch.log(prob_action) - torch.log(old_probs)).squeeze()  # trick for division
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-self.eps, 1+self.eps)  * advantages
            
            policy_loss = torch.min(surr1, surr2).mean()
            mse_loss = nn.MSELoss(reduction='mean')(V, target.detach())
            entropy_loss = Categorical(probs).entropy().mean()

            loss = -policy_loss + self.c1 * mse_loss - self.c2 * entropy_loss

            
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

    def get_action(self, state):
        probs = self.actor_pass(state)
        dist = Categorical(probs)
        action = dist.sample().item()

        return action, probs

def train(n_episodes=2500, horizon=1024, k_epochs=30, print_interval=20, save_interval=0, dir='./ppo_models', render=False):
    
    device = 'cuda' if torch.cuda.is_available else 'cpu'

    # create environment
    env = mario_env_wrapper.create_mario_env()
    state_shape = env.observation_space.shape
    n_channels = state_shape[0]
    n_actions = env.action_space.n

    model = PPO(n_channels, n_actions, device=device)

    episode_scores = np.zeros(n_episodes)
    x_pos_in_episode = np.zeros(n_episodes)
    cumulative_score = 0


    #for epi in tqdm(range(n_episodes)):
    state = env.reset()
    if render: env.render()
    done = False
    score = 0
    num_flags = 0

    epi = 0
    pbar = tqdm(total=n_episodes)
    while epi < n_episodes:
        
        for t in range(horizon):
            state_tensor = torch.from_numpy(state.__array__()).to(device)
            action, probs = model.get_action(state_tensor.unsqueeze(0))

            next_state , reward, done, info = env.step(action)
            if render: env.render()
                    
            # store experience
            prob_a = probs[0][action]
            model.store_transition(state.__array__(), action, reward, next_state.__array__(), done, prob_a)

            # set state to next state
            state = next_state
            
            score += reward
            
            if done:
                # reached end of episode
                episode_scores[epi] = score
                cumulative_score += score
                
                x_pos_in_episode[epi] = info['x_pos']

                if info['flag_get']:
                    num_flags += 1

                # print statistics
                if print_interval != 0 and epi != 0 and epi % print_interval == 0:
                    print(f'Episode {epi} score: {score}, Average score over {print_interval} episodes: {cumulative_score/print_interval}')
                    cumulative_score = 0

                # save checkpoint
                if save_interval != 0 and epi != 0 and epi % save_interval == 0:
                    path = "%s/ppo_%d.pt" % (dir, epi)
                    print(f"Saving model to {path}")
                    torch.save(model.state_dict, path)

                epi += 1
                score = 0
                pbar.update(1)
                if epi == n_episodes:
                    break

                state = env.reset()
                if render: env.render()
        
        # update model after T (horizon) steps are taken
        model.update(k_epochs)
        
        
    env.close()

    print(f"Flags captured: {num_flags}")

    # save final model
    path = "%s/ppo_final.pt" % (dir)
    torch.save(model.state_dict(), path)

    # save episode scores to csv
    np.savetxt("ppo_episode_rewards.csv", episode_scores, delimiter=",")

    # save x positions
    np.savetxt("ppo_episode_pos.csv", x_pos_in_episode, delimiter=',')

def test(path="./ppo_models/ppo_final.pt", render=True):
    device = 'cuda' if torch.cuda.is_available else 'cpu'

    # create environment
    env = mario_env_wrapper.create_mario_env()
    state_shape = env.observation_space.shape
    n_channels = state_shape[0]
    n_actions = env.action_space.n

    #load model
    model = PPO(n_channels, n_actions, device=device)
    model.load_state_dict(torch.load(path))    

    state = env.reset()
    if render: env.render()
    state = torch.from_numpy(state.__array__()).unsqueeze(0).to(device)
    done = False

    score = 0
    while not done:
        probs = model.actor(state)
        a = torch.argmax(probs).item()

        s_prime , r, done, info = env.step(a)
        if render: env.render()

        state = torch.from_numpy(s_prime.__array__()).unsqueeze(0).to(device)

        score += r

    env.close()
    print(f"Evaluation score: {score}")


def main():
    train(render=True)
    test(render=True)    

if __name__ == '__main__':
    main()
        