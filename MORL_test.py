import argparse
import pandas as pd
import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import collections
from ns3gym import ns3env
from dqw_agent import DWL

import torch

# Path to trained net
PATH = 'MORL'


def main():
    parser = argparse.ArgumentParser(description='Start simulation script on/off')
    parser.add_argument('--start',
                        type=int,
                        default=1,
                        help='Start ns-3 simulation script 0/1, Default: 1')
    # parser.add_argument('--iterations',
    #                     type=int,
    #                     default=5000,
    #                     help='Number of iterations, Default: 5000')
    args = parser.parse_args()
    startSim = bool(args.start)
    # iterationNum = int(args.iterations)
    iterationNum = 12


    port = 5555
    stepTime = 0.005 # seconds
    seed = 0
    debug = False

    env = ns3env.Ns3Env(port=port, stepTime=stepTime, startSim=startSim, simSeed=seed, debug=debug)
    env.reset();

    ob_space = env.observation_space
    state_shape = ob_space.shape
    ac_space = env.action_space
    action_num = 41
    num_objectives = 2
    print("Observation space: ", ob_space,  ob_space.dtype,state_shape)
    print("Action space: ", ac_space, ac_space.dtype)
    print("number of objective:", num_objectives)

    currIt = 0

    # The Q-learning agent parameters
    BATCH_SIZE = 32
    LR = .001                   # learning rate
    EPSILON = .95  # .05               # starting epsilon for greedy policy
    EPSILON_MIN = .1           # The minimal epsilon we want
    EPSILON_DECAY = 0.9  # .99995      # The minimal epsilon we want
    GAMMA = .9                # reward discount
    MEMORY_SIZE = 2000        # size of the replay buffer

    # The W-learning parameters
    WEPSILON = 0.99  #.01
    WEPSILON_DECAY = 0.9 # 0.9995
    WEPSILON_MIN = 0.01

    agent = DWL(state_shape, action_num, num_objectives, dnn_structure=True, epsilon=EPSILON, epsilon_min=EPSILON_MIN,
                    epsilon_decay=EPSILON_DECAY, wepsilon=WEPSILON, wepsilon_decay=WEPSILON_DECAY,
                    wepsilon_min=WEPSILON_MIN, memory_size=MEMORY_SIZE, learning_rate=LR, gamma=GAMMA)

    # Init list for information we need to collect during simulation
    num_of_steps = []
    # With 2 objectives
    coll_reward1 = []
    coll_reward2 = []
    loss_q1_episode = []
    loss_q2_episode = []
    loss_w1_episode = []
    loss_w2_episode = []
    pol1_sel_episode = []
    pol2_sel_episode = []
    num_of_invalid_actions = []

    try:
        while True:
            print("Start iteration: ", currIt)
            if currIt:
                print("load net******")
                agent.load(PATH)

            done = False
            rewardsSum1 = 0
            rewardsSum2 = 0
            qSum = 0
            qActions = 1
            lossSum = 0
            obs = env.reset()

            reward = 0
            info = None

            num_steps = 0
            selected_policies = []
            invalid_actions=0

            energy_saved_proportion = []

            energy_saved = 0

            rewards = []

            while True:
                print("obs:",obs)
                ndarray = np.fromiter(iter(obs), dtype=np.float64)
                nom_action, sel_policy = agent.get_action_nomination(ndarray)
                print("nom_action:", nom_action)

                num_steps += 1
                nextState, reward, done, info = env.step(nom_action)
                selected_policies.append(sel_policy)

                print("info:",info)
                ans = b.split(',')
                rewards.append(float(ans[0])) # energy
                energy_saved = float(ans[1])*1.0 / 670

                energy_saved_proportion.append(energy_saved)
                rewards.append(reward)


                if(float(info)==-32):
                    invalid_actions += 1
                print("---obs, reward, done, info: ", obs, rewards, done, info)

                agent.store_transition(ndarray, nom_action, rewards, nextState, done, sel_policy)
                agent.learn()
                rewardsSum1 = np.add(rewardsSum1, float(ans[0]))
                rewardsSum2 = np.add(rewardsSum2, reward)
                obs = nextState

                if done:
                    if currIt + 1 < iterationNum:
                        env.reset()
                    break
                else:
                    rewards.clear()

            mpl.rcdefaults()
            mpl.rcParams.update({'font.size': 16})
            fig, ax = plt.subplots(2, 2, figsize=(4, 2))
            plt.tight_layout(pad=0.3)

            ax[0, 0].plot(range(len(energy_saved_proportion)), coll_reward1, marker="", linestyle="-")
            ax[0, 0].set_title('saved power in each time step')
            ax[0, 0].set_xlabel('timestep')
            ax[0, 0].set_ylabel('saved power')


            q_loss, w_loss = agent.get_loss_values()
            print("Episode", currIt, "end_reward", rewards, "Sum of the reward:", qSum, "Num steps:", num_steps,
                  "Epsilon:", agent.epsilon, "Q loss:", q_loss, "W loss", w_loss)
            count_policies = collections.Counter(selected_policies)
            print("Policies selected in the episode:", count_policies, "Policy 1:", count_policies[0],
                  "Policy 2:", count_policies[1])
            count_policies = collections.Counter(selected_policies)
            # q_losses, w_losses = agent.get_loss_values()
            # Save the performance to lists
            num_of_steps.append(num_steps)
            coll_reward1.append(rewardsSum1)
            coll_reward2.append(rewardsSum2)
            pol1_sel_episode.append(count_policies[0])
            pol2_sel_episode.append(count_policies[1])
            loss_q1_episode.append(q_loss[0])
            loss_q2_episode.append(q_loss[1])
            loss_w1_episode.append(w_loss[0])
            loss_w2_episode.append(w_loss[1])
            num_of_invalid_actions.append(invalid_actions)

            agent.update_params()

            agent.save(PATH)

            currIt += 1
            if currIt == iterationNum:
                break

        # Save the results
        # df_results = pd.DataFrame()
        # df_results['episodes'] = range(1, iterationNum + 1)
        # df_results['num_steps'] = num_of_steps
        # df_results['col_reward1'] = coll_reward1
        # df_results['col_reward2'] = coll_reward2
        # df_results['policy1'] = pol1_sel_episode
        # df_results['policy2'] = pol2_sel_episode
        # df_results['loss_q1'] = loss_q1_episode
        # df_results['loss_q2'] = loss_q2_episode
        # df_results['loss_w1'] = loss_w1_episode
        # df_results['loss_w2'] = loss_w2_episode
        # df_results.to_csv('MORL.csv')

        mpl.rcdefaults()
        mpl.rcParams.update({'font.size': 16})
        fig, ax = plt.subplots(2, 2, figsize=(4, 2))
        plt.tight_layout(pad=0.3)

        ax[0, 0].plot(range(len(coll_reward1)), coll_reward1, marker="", linestyle="-")
        ax[0, 0].set_title('reward of energy saving')
        ax[0, 0].set_xlabel('episodes')
        ax[0, 0].set_ylabel('coll_reward1')

        ax[0, 1].plot(range(len(coll_reward2)), coll_reward2, marker="", linestyle="-")
        ax[0, 1].set_title('reward of Qos')
        ax[0, 1].set_xlabel('episodes')
        ax[0, 1].set_ylabel('coll_reward2')

        ax[1, 0].plot(range(len(loss_q1_episode)), loss_q1_episode, marker="", linestyle="-")
        ax[1, 0].set_title('loss of object 1')
        ax[1, 0].set_xlabel('episodes')
        ax[1, 0].set_ylabel('loss_q1_episode')

        ax[1, 1].plot(range(len(loss_q2_episode)), loss_q2_episode, marker="", linestyle="-")
        ax[1, 1].set_title('loss of object2')
        ax[1, 1].set_xlabel('episodes')
        ax[1, 1].set_ylabel('loss_q2_episode')

        plt.savefig('plots.png')
        plt.show()

    except KeyboardInterrupt:
        print("Ctrl-C -> Exit")
    finally:
        print("Done")


if __name__ == '__main__':
    main()






