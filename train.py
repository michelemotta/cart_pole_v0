"""
On-policy first-visit MC control

Details about the problem:
https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
"""

import gym
import time
import math
import random
import pickle

#Number of episodes used for the statistics calculation
INFO_SCALER = 100
#Number of episodes used for the training
EPISODES = 200

def getState(obs):
    """
    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        3       Pole Angular Velocity     -Inf                    Inf
    """
    state = ''

    #TODO: experiment with various resolution for each state
    #Cart position (only integer)
    state += str(math.floor(obs[0]))
    #Cart velocity
    state += str(math.floor(obs[1]))
    #Pole angle
    state += str(math.floor(obs[2]*10.0))
    #Pole angular velocity
    state += str(math.floor(obs[3]))

    return state


def policy(env, Q, state, episode):
    """
    Find the maximum reward for a given state and find the greedy action.
    The greedy action is selected only 1-epsilon of the time.
    """
    current_state_knowledge = list(filter(lambda x: (x['state']==state), Q))

    #The state is not yet present in the policy
    if not current_state_knowledge:
        return env.action_space.sample()

    greedy_action = max(current_state_knowledge, key=lambda x:x['average'])['action']

    #Be greedy only epsilon of the time
    epsilon = 0.9 + episode/EPISODES

    #Random number in range [0.0,1.0)
    random_value = random.random()
    if random_value < epsilon:
        return greedy_action

    return env.action_space.sample()


def main():
    env = gym.make('CartPole-v0')

    Q_s_a = [] #{state:__,action:__,tot:__, count:__,average:___}

    list_rewards = []

    #Only for statistics: used to count the consecutive successfully episodes
    cont_consecutive_success = 0

    #Try to restore the older knowledge
    try:
        with open('./obj/previous_knowledge.pickle', 'rb') as p:
            Q_s_a = pickle.load(p)
    except:
        Q_s_a = []
        print("Old knowledge not found.")

    for i_episode in range(EPISODES):
        observation = env.reset()

        episode = []
        total_rewards = 0

        for t in range(200):
            env.render()

            # Get the state
            state = getState(observation)
            # Get the action
            action = policy(env, Q_s_a, state, i_episode)

            observation, reward, done, info = env.step(action)

            total_rewards += reward

            episode.append({'state': state, 'action': action, 'reward': reward})

            #Episode finished: update the state-action value function
            if done:

                #Enumerate all the times a state is visited
                episode = find_occurrences(episode)

                G = 0
                gamma = 0.95
                #Travel through steps in reverse order
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

                        #New state action: append it to the value function
                        Q_s_a.append({'state': step['state'], 'action': step['action'], 'tot': G, 'count': 1, 'average': G})

                print("Total rewards for episode {} is: ".format(i_episode), total_rewards)
                list_rewards.append(total_rewards)

                #Update the statistics
                if total_rewards >= 195.0:
                    cont_consecutive_success += 1
                    print("Consecutive successfull episodes: ", cont_consecutive_success)
                else:
                    cont_consecutive_success = 0

                break

        #After INFO_SCALER episodes calculate the averaged rewards
        if ((i_episode+1)%INFO_SCALER) == 0:
            print("Averaged rewards: ", sum(list_rewards)/INFO_SCALER)
            list_rewards.clear()

    env.close()

    try:
        with open('./obj/previous_knowledge.pickle', 'wb') as p:
            pickle.dump(Q_s_a, p)
    except:
        print("Error in saving the knowledge!")

#Find the number of times a state is visited within a single episode and report it
def find_occurrences(episode):
    for _, step in enumerate(episode):
        occurrences = 0

        #The state has been already counted in one or more previous iterations
        if 'i' in step:
            continue

        for cont, item in enumerate(episode):
            if (step['state'] == item['state']) and (step['action'] == item['action']) :
                occurrences += 1
                episode[cont]['i'] = occurrences
    return episode


if __name__ == "__main__":
    main()