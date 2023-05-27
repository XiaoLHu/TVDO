import numpy as np
import threading


class ReplayBuffer:
    def __init__(self, args):
        self.args = args
        self.n_actions = self.args.n_actions
        self.n_agents = self.args.n_agents
        self.state_shape = self.args.state_shape
        self.obs_shape = self.args.obs_shape
        self.size = self.args.buffer_size
        self.episode_limit = self.args.episode_limit
        # memory management
        self.current_idx = 0
        self.current_size = 0
        # create the buffer to store info
        self.buffers = {'o': np.empty([self.size, self.episode_limit, self.n_agents, self.obs_shape]),
                        'o_next': np.empty([self.size, self.episode_limit, self.n_agents, self.obs_shape]),
                        's': np.empty([self.size, self.episode_limit, self.state_shape]),
                        's_next': np.empty([self.size, self.episode_limit, self.state_shape]),
                        'u': np.empty([self.size, self.episode_limit, self.n_agents, 1]),
                        'u_one_hot': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions]),
                        'avail_u': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions]),
                        'avail_u_next': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions]),
                        'r': np.empty([self.size, self.episode_limit, 1]),
                        'terminated': np.empty([self.size, self.episode_limit, 1]),
                        'padded': np.empty([self.size, self.episode_limit, 1])
                        }
        # thread lock
        self.lock = threading.Lock()

    def store_episode(self, episode_batch, episode_num):
        batch_size = episode_num  # episode_number
        with self.lock:
            idx = self._get_storage_idx(inc=batch_size)
            # store the information
            self.buffers['o'][idx] = episode_batch['o']
            self.buffers['o_next'][idx] = episode_batch['o_next']
            self.buffers['s'][idx] = episode_batch['s']
            self.buffers['s_next'][idx] = episode_batch['s_next']
            self.buffers['u'][idx] = episode_batch['u']
            self.buffers['u_one_hot'][idx] = episode_batch['u_one_hot']
            self.buffers['avail_u'][idx] = episode_batch['avail_u']
            self.buffers['avail_u_next'][idx] = episode_batch['avail_u_next']
            self.buffers['r'][idx] = episode_batch['r']
            self.buffers['terminated'][idx] = episode_batch['terminated']
            self.buffers['padded'][idx] = episode_batch['padded']
            if self.args.alg == 'maven':
                self.buffers['z'][idx] = episode_batch['z']

    def sample(self, batch_size):
        temp_buffer = {}
        idx = np.random.randint(0, self.current_size, batch_size)
        for key in self.buffers.keys():
            temp_buffer[key] = self.buffers[key][idx]
        return temp_buffer

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        # self.size = buffer_size
        if self.current_idx + inc <= self.size:
            idx = np.arange(self.current_idx, self.current_idx + inc)
            self.current_idx += inc
        elif self.current_idx < self.size:
            overflow = inc - (self.size - self.current_idx)
            idx_a = np.arange(self.current_idx, self.size)
            idx_b = np.arange(0, overflow)
            idx = np.concatenate([idx_a, idx_b])
            self.current_idx = overflow
        else:
            idx = np.arange(0, inc)
            self.current_idx = inc
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx
