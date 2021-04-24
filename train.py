import gym
import time
import math
import random


def getState(obs):
    state = ''
    for o in obs:
        state += str(math.floor(o))
    
    return state


def policy(env, Q, state):
    #Find the maximum reward for a given state and see the action
    #Filter for current state and take the action with maximum reward
    current_state_knowledge = list(filter(lambda x: (x['state']==state), Q))

    if not current_state_knowledge:
        return env.action_space.sample()
    
    greedy_action = max(current_state_knowledge, key=lambda x:x['average'])['action']
    #Be greedy only epsilon of the time

    random_value = random.random() # random number in range [0.0,1.0)
    if random_value < 0.985:
        return greedy_action

    return env.action_space.sample()


def main():
    env = gym.make('CartPole-v0')

    total_rewards = 0
    Q_s_a = []  #{state:__,action:__,tot:__, count:__,average:___}
    Returns_s_a = []

    for i_episode in range(1000):
        observation = env.reset()

        episode = []
        total_rewards = 0
        print('Episode: ', i_episode)

        for t in range(200):
            env.render()
            state = getState(observation)  # Get the state
            action = policy(env, Q_s_a, state)  # Get the action

            #print(observation)

            observation, reward, done, info = env.step(action)

            total_rewards += reward # Update total reward for this episode

            episode.append({'state': state, 'action': action, 'reward': reward})

            if done:
                print("Episode finished after {} timesteps".format(t+1))

                episode = find_occurrences(episode)
                #print(episode)

                G = 0
                gamma = 0.95
                for cont, step in enumerate(reversed(episode)):

                    G = (gamma*G) + step['reward']
                    #First time visit!
                    if step['i'] == 1:
                        for q in Q_s_a:
                            if (q['state'] == step['state']) and (q['action'] == step['action']):
                                q['tot'] += G
                                q['count'] += 1
                                q['average'] = q['tot']/q['count'] 
                                continue
                        #New state action couple
                        Q_s_a.append({'state': step['state'], 'action': step['action'], 'tot': G, 'count': 1, 'average': G})
                #print(Q_s_a)

                print("Total rewards for this episode is: ", total_rewards)

                break

    env.close()

def find_occurrences(episode):
    for _, step in enumerate(episode):
        occurrences = 0

        if 'i' in step:
            continue

        for cont, item in enumerate(episode):
            if (step['state'] == item['state']) and (step['action'] == item['action']) :
                occurrences += 1
                episode[cont]['i'] = occurrences
    return episode


if __name__ == "__main__":
    main()