from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
from sklearn.cluster import AgglomerativeClustering
from modules.task_selectors import REGISTRY as task_selector_REGISTRY
import copy
from modules.masknetwork import MaskNetwork
import numpy as np
import os
from components.clustering import REGISTRY as cluster_REGISTRY
import re


class TAPPSMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.hidden_states = None

        self.n_tasks = args.n_task_clusters
        self.task_hidden_states = None
        self.task_interval = args.task_interval
        self.task_selector = task_selector_REGISTRY[args.task_selector](input_shape, args)
        self.task_latent = th.ones(self.n_tasks, self.args.rnn_hidden_dim).to(args.device)
        self.task_mask = th.ones(self.n_tasks, self.args.rnn_hidden_dim).to(args.device)
        self.state_dim = args.state_shape
        self.selected_tasks = th.randint(0, self.n_tasks, (self.n_agents, 1))
        self.state = th.ones(self.state_dim).to(args.device)
        self.mask_net = MaskNetwork(self.args, self.state_dim, self.task_latent, self.n_tasks)

        from collections import deque
        self.hidden_state_buffer = deque(maxlen=args.update_cluster_start)

        self.clusterer = cluster_REGISTRY[self.args.cluster_method](self.args)
        self.cluster_update_counter = 0
        self.map_name = args.env_args['map_name']

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions, self.selected_tasks

    def forward(self, ep_batch, t, test_mode=False, t_env=None):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]

        task_outputs = None
        if t % self.task_interval == 0:
            task_outputs = self.task_selector(self.hidden_states, self.task_latent)
            self.selected_tasks = self.task_selector.select_task(task_outputs, test_mode=test_mode, t_env=t_env).squeeze()

        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states, self.selected_tasks, self.task_mask)

        if self.hidden_states is not None:
            self.hidden_state_buffer.append(self.hidden_states.clone().detach())

        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)

    def parameters(self):
        params = list(self.agent.parameters())
        params += list(self.task_selector.parameters())
        return params

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())
        self.task_selector.load_state_dict(other_mac.task_selector.state_dict())
        self.task_latent = copy.deepcopy(other_mac.task_latent)
        self.task_mask = copy.deepcopy(other_mac.task_mask)

    def cuda(self):
        self.agent.cuda()
        self.task_selector.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))
        th.save(self.task_selector.state_dict(), "{}/task_selector.th".format(path))
        th.save(self.task_latent, "{}/task_latent.pt".format(path))
        th.save(self.task_mask, "{}/task_mask.pt".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        self.task_selector.load_state_dict(th.load("{}/task_selector.th".format(path),
                                           map_location=lambda storage, loc: storage))
        self.task_latent = th.load("{}/task_latent.pt".format(path),
                                   map_location=lambda storage, loc: storage).to(self.args.device)
        self.task_mask = th.load("{}/task_mask.pt".format(path),
                                   map_location=lambda storage, loc: storage).to(self.args.device)

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape

    def cluster_hidden_states(self):
        if not hasattr(self, 'cluster_update_counter'):
            self.cluster_update_counter = 0
        self.cluster_update_counter += 1

        if self.cluster_update_counter % self.args.update_cluster_start != 0:
            return

        if len(self.hidden_state_buffer) >= self.args.update_cluster_start:
            self.hidden_state_buffer = list(self.hidden_state_buffer)[-self.args.update_cluster_start:]

            all_hidden_states = th.cat(self.hidden_state_buffer, dim=0)
            all_hidden_states_np = all_hidden_states.cpu().numpy()

            cluster_centers_np, filtered_labels, filtered_indices, filtered_hidden_states = self.clusterer.fit(all_hidden_states_np)
            self.n_tasks = len(cluster_centers_np)
            self.args.n_task_clusters = self.n_tasks
            self.task_latent = th.tensor(cluster_centers_np, device=self.args.device, dtype=th.float32)

            task_latent = self.task_latent.detach()
            self.mask_net.to_cuda()
            self.task_mask = self.mask_net(self.state, task_latent, self.n_tasks).detach()


    def task_mask_params(self):
        return list(self.mask_net.parameters())

    def optimize_masknet(self):
        hidden_states = self.hidden_states.detach().clone()
        selected_roles = self.selected_tasks.detach().clone()
        role_mask = self.task_mask.detach().clone()
        info_nce_loss = self.mask_net.optimize_masknet(role_mask, hidden_states, selected_roles)
        return info_nce_loss
