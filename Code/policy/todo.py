import torch
import os
import pdb
from network.base_net import RNN
from network.tvdo_net import TVDONet


class TVDO:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        input_shape = self.obs_shape
        if args.last_action:
            input_shape += self.n_actions
        if args.reuse_network:
            input_shape += self.n_agents

        self.eval_rnn = RNN(input_shape, args)
        self.target_rnn = RNN(input_shape, args)
        self.eval_vdn_net = TVDONet()
        self.target_vdn_net = TVDONet()
        self.args = args
        if self.args.cuda:
            self.eval_rnn.cuda(self.args.gpu)
            self.target_rnn.cuda(self.args.gpu)
            self.eval_vdn_net.cuda(self.args.gpu)
            self.target_vdn_net.cuda(self.args.gpu)

        self.model_dir = args.model_dir + '/' + args.alg + '/' + args.map

        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_vdn_net.load_state_dict(self.eval_vdn_net.state_dict())

        self.eval_parameters = list(self.eval_vdn_net.parameters()) + list(self.eval_rnn.parameters())
        if args.optimizer == "RMS":
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=args.lr)

        self.eval_hidden = None
        self.target_hidden = None
        print('Init alg TVDO')


    def learn(self, episode_batch, max_episode_len, train_step, epsilon=None):
        episode_num = episode_batch['o'].shape[0]
        self.init_hidden(episode_num)
        for key in episode_batch.keys():
            if key == 'u':
                episode_batch[key] = torch.tensor(episode_batch[key], dtype=torch.long)
            else:
                episode_batch[key] = torch.tensor(episode_batch[key], dtype=torch.float32)

        u, avail_u, avail_u_next, reward, terminated = episode_batch['u'], episode_batch['avail_u'], \
                                                  episode_batch['avail_u_next'], episode_batch['r'],  \
                                                  episode_batch['terminated']
        mask = 1 - episode_batch["padded"].float()

        if self.args.cuda:
            u = u.cuda(self.args.gpu)
            reward = reward.cuda(self.args.gpu)
            terminated = terminated.cuda(self.args.gpu)
            mask = mask.cuda(self.args.gpu)

        # [episode_num, max_episode_len, n_agents, n_actions]
        q_evals, q_targets, q_evals_next = self.get_q_values(episode_batch, max_episode_len)

        q_evals_clone = q_evals.clone()
        q_evals = torch.gather(q_evals, dim=3, index=u).squeeze(3)
        q_total_eval = self.eval_vdn_net(q_evals)

        q_targets[avail_u_next == 0.0] = - 9999999
        q_targets = q_targets.max(dim=3)[0]
        q_total_target = self.target_vdn_net(q_targets)

        td_lambda_target = self.get_td_lambda_targets(reward, terminated, mask, q_total_target)
        td_lambda_error = q_total_eval - td_lambda_target.detach()

        masked_td_n_error = mask * td_lambda_error

        # E = rho * max{|Q_eval-Q_opt|}
        # [episode_num, max_episode_len, n_agents]
        # q_individual = torch.gather(q_evals, dim=-1, index=u).squeeze(-1)
        # pdb.set_trace()
        q_opt = q_evals_clone.max(dim=-1)[0]
        Q_Error = abs(q_evals - q_opt)
        Q_max_Error = self.args.rho * Q_Error.max(dim=2)[0]
        Q_max_Error = Q_max_Error.unsqueeze(-1)

        loss = ((masked_td_n_error + Q_max_Error) ** 2).sum() / mask.sum()
        # print('Loss is ', loss)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()

        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
            self.target_vdn_net.load_state_dict(self.eval_vdn_net.state_dict())

    # obtain the information of transition_idx-th in the set of episode_batch
    #    (inputs = [obs, u_one_hot[transition_idx-1], agent_id]
    #     inputs_next = [obs_next, u_one_hot[transition], agent_id])
    def _get_inputs(self, episode_batch, transition_idx):
        # obs: [batch_size, n_agents, obs_shape] map=3m obs:[32, 3, 30]
        # obs_next: [batch_size, n_agents, obs_shape] map=3m obs_next:[32, 3, 30]
        # u_one_hot: [batch_size, episode_limit, n_agents, n_actions] map=3m u_one_hot:[32, 59, 3, 9]
        obs, obs_next, u_one_hot = episode_batch['o'][:, transition_idx], \
                                  episode_batch['o_next'][:, transition_idx], episode_batch['u_one_hot'][:]
        episode_num = obs.shape[0]
        inputs, inputs_next = [], []
        # pdb.set_trace()
        inputs.append(obs)
        inputs_next.append(obs_next)

        if self.args.last_action:
            if transition_idx == 0:
                inputs.append(torch.zeros_like(u_one_hot[:, transition_idx]))
            else:
                inputs.append(u_one_hot[:, transition_idx - 1])
            inputs_next.append(u_one_hot[:, transition_idx])

        if self.args.reuse_network:
            inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
            inputs_next.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))

        # inputs[0]: obs[32, 3, 30]
        # inputs[1]: u_one_hot[32, 3, 9]
        # inputs[2]: agent_id[32, 3, 3]

        inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)
        # inputs_next: [episode_num * n_agents, obs_shape + n_actions + n_agents] map=3m, inputs_next:[96, 42]
        inputs_next = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs_next], dim=1)
        return inputs, inputs_next

    # [episode_num, max_episode_len, n_agents, n_actions]
    def get_q_values(self, episode_batch, max_episode_len):
        episode_num = episode_batch['o'].shape[0]

        q_evals, q_targets, q_evals_next = [], [], []
        for transition_idx in range(max_episode_len):
            inputs, inputs_next = self._get_inputs(episode_batch, transition_idx)
            if self.args.cuda:
                inputs = inputs.cuda(self.args.gpu)
                inputs_next = inputs_next.cuda(self.args.gpu)
                self.eval_hidden = self.eval_hidden.cuda(self.args.gpu)
                self.target_hidden = self.target_hidden.cuda(self.args.gpu)

            # pdb.set_trace()
            # q_eval: [episode_num * n_agents, n_actions] map=3m, q_eval[96, 9]
            q_eval, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden)
            # q_target: [episode_num * n_agents, n_actions] map=3m, q_target[96, 9]
            q_target, self.target_hidden = self.target_rnn(inputs_next, self.target_hidden)
            # q_eval_next: [episode_num * n_agents, n_actions] map=3m, q_eval[96, 9]
            q_eval_next, _ = self.eval_rnn(inputs_next, self.eval_hidden)

            q_eval = q_eval.view(episode_num, self.n_agents, -1)
            q_target = q_target.view(episode_num, self.n_agents, -1)
            q_eval_next = q_eval_next.view(episode_num, self.n_agents, -1)
            q_evals.append(q_eval)
            q_targets.append(q_target)
            q_evals_next.append(q_eval_next)

        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        q_evals_next = torch.stack(q_evals_next, dim=1)
        return q_evals, q_targets, q_evals_next

    def get_q_targets_by_td_lambda(self, reward, q_total_evals, q_total_targets, mask, max_episode_len):
        TD_lambda_errror = reward.new_zeros(*reward.shape)
        targets = reward + self.args.gamma * q_total_targets
        td_error = (targets.detach() - q_total_evals) * mask
        # q_error = (q_total_evals_next - q_total_targets) * (1 - terminated)
        transition_idx = max_episode_len - 1
        TD_lambda_errror[:, transition_idx] = td_error[:, transition_idx]
        for transition_idx in range(max_episode_len - 2, -1, -1):
            TD_lambda_errror[:, transition_idx] = td_error[:, transition_idx] + self.args.gamma * self.args.td_lambda \
                                        * (TD_lambda_errror[:, transition_idx + 1])
        # pdb.set_trace()
        return TD_lambda_errror

    def get_td_lambda_targets(self, reward, terminated, mask, target_qs):
        # Assumes  <target_qs > in B*T*A and <reward >, <terminated >, <mask > in (at least) B*T-1*1
        # Initialise  last  lambda -return  for  not  terminated  episodes
        '''
        ret = target_qs.new_zeros(*target_qs.shape)
        ret[:, -1] = target_qs[:, -1] * (1 - torch.sum(terminated, dim=1))+
        # Backwards  recursive  update  of the "forward  view"
        for t in range(ret.shape[1] - 2, -1, -1):
            ret[:, t] = td_lambda * gamma * ret[:, t + 1] + mask[:, t] * (
                        reward[:, t] + (1 - td_lambda) * gamma * target_qs[:, t + 1] * (1 - terminated[:, t]))
        # Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
        return ret[:, 0:-1]
        '''

        # reward, terminated: [ns, bs]; target_qs: [ns-1, bs]; mask:[ns-1,bs]
        # >> ret: [ns-1, bs]
        # pdb.set_trace()
        td_lambda = self.args.td_lambda
        gamma = self.args.gamma
        q_tgt = reward + (1 - td_lambda) * gamma * (1 - terminated) * target_qs
        ret = torch.zeros_like(q_tgt)
        ret[:, -1] = q_tgt[:, -1]
        for t in range(q_tgt.shape[1] - 2, -1, -1):
            ret[:, t] = td_lambda * gamma * ret[:, t+1] + mask[:, t] * q_tgt[:, t]
        return ret


    def init_hidden(self, episode_num):
        self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.target_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))

    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.eval_vdn_net.state_dict(), self.model_dir + '/' + num + '_tvdo_net_params.pkl')
        torch.save(self.eval_rnn.state_dict(),  self.model_dir + '/' + num + '_rnn_net_params.pkl')
