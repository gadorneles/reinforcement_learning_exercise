import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run(episodes=10000, use_model = False, render=False):

    env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=False, render_mode="human" if render else None)

    if use_model:
        f = open("frozen_lake_q_learning.pkl", "rb")
        q = pickle.load(f)
        f.close()
    else:
        q = np.zeros([env.observation_space.n, env.action_space.n]) # Q-table 64x4

    learning_rate = 0.9 # Learning rate alpha
    discount_factor = 0.9 # Discount factor gamma

    #Epsilon greedy policy
    epsilon = 1 
    epsilon_decay = 0.0001 # Epsilon decay rate. 1/10000 = 0.0001
    random_number = np.random.default_rng() 

    # Store the rewards obtained in each episode
    episode_reward = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0] # Reset the environment and get the initial state
        terminated = False #True when fail or success state
        truncation = False #True when agent is stuck

        while(not terminated and not truncation):

            #If the number generated is smaller than epsilon, the agent will explore. Otherwise, it will follow the Q-table.
            if (not use_model) and (random_number.random() < epsilon):
                action = env.action_space.sample() #possible actions: left (0), down (1), right (2), up (3)
            else:
                action = np.argmax(q[state,:])

            new_state, reward, terminated, truncation, _ = env.step(action) # Take the action and observe the outcome state and reward

            # Update the Q-table
            if not use_model:
                q[state,action] = q[state,action] + learning_rate * (reward + discount_factor * np.max(q[new_state,:]) - q[state,action])

            state = new_state # Update the state
        
        epsilon = max(epsilon - epsilon_decay, 0) 

        #Stabilize the learning rate after we're no longer exploring
        if epsilon == 0:
            learning_rate = 0.0001

        #Keep track of the rewards obtained in each episode
        if reward == 1:
            episode_reward[i] = 1

    
    env.close()
    print('Training is done.')

    if not use_model:
        sum_rewards = np.zeros(episodes)
        for t in range(episodes):
            sum_rewards[t] = np.sum(episode_reward[max(0, t-100):(t+1)]) # Sum of rewards in the last 100 episodes
        plt.plot(sum_rewards)
        plt.savefig("frozen_lake_q_learning.png")

        f = open("frozen_lake_q_learning.pkl", "wb")
        pickle.dump(q, f)
        f.close()
    else:
        # Create a pie chart to visualize the distribution of rewards
        reward_counts = np.bincount(episode_reward.astype(int))
        labels = ['0', '1']
        plt.pie(reward_counts, labels=labels, autopct='%1.1f%%')
        plt.title('Distribution of Rewards')
        plt.savefig("reward_distribution.png")
        plt.show()

if __name__ == "__main__":
    run(episodes=15000, use_model = True, render=True)