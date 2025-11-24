import torch.nn as nn
import torch.nn.functional as F

import torch as th
from torch.distributions import Categorical


class DotSelector(nn.Module):
    def __init__(self, input_shape, args):
        super(DotSelector, self).__init__()
        self.args = args
        self.epsilon_start = self.args.epsilon_start
        self.epsilon_finish = self.args.task_epsilon_finish
        self.epsilon_anneal_time = self.args.epsilon_anneal_time
        self.epsilon_anneal_time_exp = self.args.epsilon_anneal_time_exp
        self.delta = (self.epsilon_start - self.epsilon_finish) / self.epsilon_anneal_time
        self.task_action_spaces_update_start = self.args.task_action_spaces_update_start
        self.epsilon_start_t = 0
        self.epsilon_reset = True

        self.fc1 = nn.Linear(args.rnn_hidden_dim, 2 * args.rnn_hidden_dim)
        self.fc2 = nn.Linear(2 * args.rnn_hidden_dim, args.rnn_hidden_dim)

        self.epsilon = 0.05

    def forward(self, inputs, task_latent):
        x = self.fc2(F.relu(self.fc1(inputs.view(-1, self.args.rnn_hidden_dim))))
        x = x.unsqueeze(-1)
        task_latent_reshaped = task_latent.unsqueeze(0).repeat(x.shape[0], 1, 1)

        task_q = th.bmm(task_latent_reshaped, x).squeeze()
        return task_q

    def select_task(self, task_qs, test_mode=False, t_env=None):
        self.epsilon = self.epsilon_schedule(t_env)
        n_task = task_qs.shape[-1]
        if test_mode:
            self.epsilon = 0.0

        masked_q_values = task_qs.detach().clone()
        random_numbers = th.rand_like(task_qs[:, 0])
        pick_random = (random_numbers < self.epsilon).long()
        random_tasks = Categorical(th.ones(task_qs.shape).float().to(self.args.device)).sample().long()

        picked_tasks = pick_random * random_tasks + (1 - pick_random) * masked_q_values.max(dim=-1)[1]
        return picked_tasks

    def epsilon_schedule(self, t_env):
        if t_env is None:
            return 0.05

        if t_env > self.task_action_spaces_update_start and self.epsilon_reset:
            self.epsilon_reset = False
            self.epsilon_start_t = t_env
            self.epsilon_anneal_time = self.epsilon_anneal_time_exp
            self.delta = (self.epsilon_start - self.epsilon_finish) / self.epsilon_anneal_time

        if t_env - self.epsilon_start_t > self.epsilon_anneal_time:
            epsilon = self.epsilon_finish
        else:
            epsilon = self.epsilon_start - (t_env - self.epsilon_start_t) * self.delta

        return epsilon
