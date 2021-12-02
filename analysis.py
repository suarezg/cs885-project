import numpy as np
import matplotlib.pyplot as plt

AVERAGING_WINDOW = 50
NUM_EPISODES = 2500

def parse_csv(model='ppo'):
    reward_path = "%s_episode_rewards.csv" % (model)
    position_path = "%s_episode_pos.csv" % (model)
    steps_path = "%s_episode_steps.csv" % (model)

    rewards = np.genfromtxt(reward_path, delimiter=',')
    positions = np.genfromtxt(position_path, delimiter=',')
    steps = np.genfromtxt(steps_path, delimiter=',')

    reward_avg = np.convolve(rewards, np.ones(AVERAGING_WINDOW)/ AVERAGING_WINDOW, mode='valid')
    pos_avg = np.convolve(positions, np.ones(AVERAGING_WINDOW) / AVERAGING_WINDOW, mode='valid')
    steps_avg = np.convolve(steps, np.ones(AVERAGING_WINDOW) / AVERAGING_WINDOW, mode='valid')

    return reward_avg, pos_avg, steps_avg

ppo_rewards, ppo_pos, ppo_steps = parse_csv('ppo')
dqn_rewards, dqn_pos, dqn_steps = parse_csv('dqn')
sql_rewards, sql_pos, sql_steps = parse_csv('sql')


plt.plot(ppo_rewards)
plt.plot(dqn_rewards)
plt.plot(sql_rewards)
plt.legend(['PPO', 'DQN', 'SQL'])
plt.xlabel('Episode')
plt.xlim(AVERAGING_WINDOW, NUM_EPISODES)
plt.ylabel('Total Rewards')
plt.title('Reward Comparison')
plt.show()
plt.savefig('reward-comparison.png')
plt.clf()


plt.plot(ppo_pos)
plt.plot(dqn_pos)
plt.plot(sql_pos)
plt.legend(['PPO', 'DQN', 'SQL'])
plt.xlabel('Episode')
plt.xlim(AVERAGING_WINDOW, NUM_EPISODES)
plt.ylabel('Farthest Position')
plt.title('Position Comparison')
plt.show()
plt.savefig('position-comparison.png')
plt.clf()


plt.plot(ppo_steps)
plt.plot(dqn_steps)
plt.plot(sql_steps)
plt.legend(['PPO', 'DQN', 'SQL'])
plt.xlabel('Episode')
plt.xlim(AVERAGING_WINDOW, NUM_EPISODES)
plt.ylabel('Steps Taken')
plt.title('Steps Comparison')
plt.show()
plt.savefig('steps-comparison.png')
plt.clf()


