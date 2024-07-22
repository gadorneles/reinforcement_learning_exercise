import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run(episodes=10000, use_model=False, render=False):
    env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True, render_mode="human" if render else None)

    if use_model:
        with open("frozen_lake_sarsa.pkl", "rb") as f:
            q = pickle.load(f)
    else:
        q = np.zeros([env.observation_space.n, env.action_space.n])  # Q-table 64x4

    learning_rate = 0.9  # Learning rate alpha
    discount_factor = 0.9  # Discount factor gamma

    # Epsilon greedy policy
    epsilon = 1 
    epsilon_decay = 0.0001  # Epsilon decay rate. 1/10000 = 0.0001
    random_number = np.random.default_rng() 

    # Store the rewards obtained in each episode
    episode_reward = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]  # Reset the environment and get the initial state
        terminated = False  # True when fail or success state
        truncation = False  # True when agent is stuck

        # Select the first action
        if (not use_model) and (random_number.random() < epsilon):
            action = env.action_space.sample()
        else:
            action = np.argmax(q[state, :])

        while(not terminated and not truncation):
            new_state, reward, terminated, truncation, _ = env.step(action)  # Take the action and observe the outcome state and reward

            # Select the next action
            if (not use_model) and (random_number.random() < epsilon):
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(q[new_state, :])

            # Update the Q-table
            if not use_model:
                q[state, action] = q[state, action] + learning_rate * (reward + discount_factor * q[new_state, next_action] - q[state, action])

            state = new_state  # Update the state
            action = next_action  # Update the action
        
        epsilon = max(epsilon - epsilon_decay, 0) 

        # Stabilize the learning rate after we're no longer exploring
        if epsilon == 0:
            learning_rate = 0.0001

        # Keep track of the rewards obtained in each episode
        if reward == 1:
            episode_reward[i] = 1

    env.close()
    print('Training is done.')

    if not use_model:
        sum_rewards = np.zeros(episodes)
        for t in range(episodes):
            sum_rewards[t] = np.sum(episode_reward[max(0, t-100):(t+1)])  # Sum of rewards in the last 100 episodes
        plt.plot(sum_rewards)
        plt.savefig("frozen_lake_sarsa.png")

        with open("frozen_lake_sarsa.pkl", "wb") as f:
            pickle.dump(q, f)
    else:
        # Create a pie chart to visualize the distribution of rewards
        reward_counts = np.bincount(episode_reward.astype(int))
        labels = ['0', '1']
        plt.pie(reward_counts, labels=labels, autopct='%1.1f%%')
        plt.title('Distribution of Rewards')
        plt.savefig("reward_distribution.png")
        plt.show()

if __name__ == "__main__":
    run(episodes=25000, use_model=False, render=False)