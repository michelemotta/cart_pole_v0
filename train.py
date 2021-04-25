import gym
import time
import math
import random
import pickle

INFO_SCALER_STEP = 100
EPISODES = 200 #193.0 averaged on 50 samples after 2000 episodes


def getState(obs):
    #TODO: use only position and angle in order to reduce state
    #TODO: multiply the angle because is always -0.4<teta<0.4
    state = ''
    
    state += str(math.floor(obs[0]))
    state += str(math.floor(obs[1]))
    state += str(math.floor(obs[2]*10.0))
    state += str(math.floor(obs[3]))
    
    return state


def policy(env, Q, state, episode):
    #Find the maximum reward for a given state and see the action
    #Filter for current state and take the action with maximum reward
    current_state_knowledge = list(filter(lambda x: (x['state']==state), Q))

    if not current_state_knowledge:
        return env.action_space.sample()
    
    greedy_action = max(current_state_knowledge, key=lambda x:x['average'])['action']
    #Be greedy only epsilon of the time

    epsilon = 0.9 + episode/EPISODES

    random_value = random.random() # random number in range [0.0,1.0)
    if random_value < epsilon:
        return greedy_action

    return env.action_space.sample()


def main():
    env = gym.make('CartPole-v0')

    Q_s_a = []  #{state:__,action:__,tot:__, count:__,average:___}
    Returns_s_a = []

    list_rewards = []

    try:
        with open('./obj/previous_knowledge.pickle', 'rb') as p:
            Q_s_a = pickle.load(p)
    except:
        Q_s_a = []
        print("Older sessions not found.")

    cont_consecutive_success = 0

    for i_episode in range(EPISODES):
        observation = env.reset()

        episode = []
        total_rewards = 0

        print('Episode: ', i_episode)

        for t in range(200):
            env.render()
            state = getState(observation)  # Get the state
            action = policy(env, Q_s_a, state, i_episode)  # Get the action

            #print(observation)

            observation, reward, done, info = env.step(action)

            total_rewards += reward # Update total reward for this episode

            episode.append({'state': state, 'action': action, 'reward': reward})

            if done:
                #print("Episode finished after {} timesteps".format(t+1))

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
                list_rewards.append(total_rewards)

                if total_rewards >= 195.0:
                    cont_consecutive_success += 1
                    print("Consecutive successfull episodes: ", cont_consecutive_success)
                else:
                    cont_consecutive_success = 0

                break

        if ((i_episode+1)%INFO_SCALER_STEP) == 0:
            print("Averaged rewards: ", sum(list_rewards)/INFO_SCALER_STEP)
            list_rewards.clear()

    env.close()

    #try:
    with open('./obj/previous_knowledge.pickle', 'wb') as p:
        pickle.dump(Q_s_a, p)
    #except:
     #   print("Error in saving the knowledge!")

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