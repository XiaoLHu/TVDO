import numpy as np
import torch
from torch.distributions import one_hot_categorical
import time
import pdb


class RolloutWorker:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args

        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon

    def generate_episode(self, episode_num=None, evaluate=False):
        if self.args.replay_dir != '' and evaluate and episode_num == 0:  # prepare for save replay of evaluation
            self.env.close()
        o, s, u, u_one_hot, avail_u, r, terminate, padded = [], [], [], [], [], [], [], []
        self.env.reset()

        terminated = False
        win_tag = False
        step = 0
        episode_reward = 0  # cumulative rewards
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        self.agents.policy.init_hidden(1)

        # epsilon
        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        while not terminated and step < self.episode_limit:
            # time.sleep(0.2)
            # pdb.set_trace()

            obs = self.env.get_obs()
            state = self.env.get_state()

            avail_actions, actions, actions_one_hot = [], [], []
            for agent_id in range(self.n_agents):
                avail_action = self.env.get_avail_agent_actions(agent_id)
                action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id,
                                                       avail_action, epsilon, evaluate)

                action_one_hot = np.zeros(self.args.n_actions)
                action_one_hot[action] = 1

                actions.append(np.int(action))
                actions_one_hot.append(action_one_hot)

                avail_actions.append(avail_action)
                last_action[agent_id] = action_one_hot

            reward, terminated, info = self.env.step(actions)
            win_tag = True if terminated and 'battle_won' in info and info['battle_won'] else False

            o.append(obs)
            s.append(state)
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_one_hot.append(actions_one_hot)
            avail_u.append(avail_actions)
            r.append([reward])
            terminate.append([terminated])
            padded.append([0.])

            episode_reward += reward
            step += 1
            if self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        obs = self.env.get_obs()
        state = self.env.get_state()
        o.append(obs)
        s.append(state)

        o_next, s_next = o[1:], s[1:]
        o, s = o[:-1], s[:-1]

        # get avail_action for last obs，because target_q needs avail_action in training
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_action = self.env.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_action)
        avail_u.append(avail_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]

        #    (o, o_next, s, s_next, u, u_one_hot, avail_u, avail_n_next, r, terminate, padded)
        for i in range(step, self.episode_limit):
            o.append(np.zeros((self.n_agents, self.obs_shape)))
            o_next.append(np.zeros((self.n_agents, self.obs_shape)))
            s.append(np.zeros(self.state_shape))
            s_next.append(np.zeros(self.state_shape))
            u.append(np.zeros([self.n_agents, 1]))
            u_one_hot.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
            r.append([0.])
            terminate.append([1.])
            padded.append([1.])

        episode = dict(o=o.copy(),
                       o_next=o_next.copy(),
                       s=s.copy(),
                       s_next=s_next.copy(),
                       u=u.copy(),
                       u_one_hot=u_one_hot.copy(),
                       avail_u=avail_u.copy(),
                       avail_u_next=avail_u_next.copy(),
                       r=r.copy(),
                       terminated=terminate.copy(),
                       padded=padded.copy()
                       )

        for key in episode.keys():
            episode[key] = np.array([episode[key]])

        if not evaluate:
            self.epsilon = epsilon

        if evaluate and episode_num == self.args.evaluate_epoch - 1 and self.args.replay_dir != '':
            self.env.save_replay()
            self.env.close()
        return episode, episode_reward, win_tag, step
