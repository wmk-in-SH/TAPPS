import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch as th
from torch.optim import Adam
from components.standarize_stream import RunningMeanStd
from modules.prioritized_memory import PER_Memory
from torch.optim import RMSprop


class TAPPSLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.n_agents = args.n_agents
        self.params = list(mac.parameters())
        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = Adam(params=self.params, lr=args.lr)

        self.target_mac = copy.deepcopy(mac)

        self.training_steps = 0
        self.last_target_update_step = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

        device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=device)
        if self.args.standardise_rewards:
            self.rew_ms = RunningMeanStd(shape=(1,), device=device)

        self.task_interval = args.task_interval
        self.device = self.args.device

        self.mask_net_params = list(self.mac.task_mask_params())
        self.mask_net_optimiser = RMSprop(params=self.mask_net_params, lr=args.lr, alpha=args.optim_alpha,
                                          eps=args.optim_eps)
        self.task_cluster_updated = True
        self.time_step_durations = []
        self.masknet_update_interval = 5000
        self.last_masknet_update_t = 0

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        target_mac_out = th.stack(target_mac_out[1:], dim=1)

        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        if self.args.double_q:
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        if self.mixer is not None:
            chosen_action_qvals_clone = chosen_action_qvals.clone().detach()
            chosen_action_qvals_clone.requires_grad = True
            target_max_qvals_clone = target_max_qvals.clone().detach()
            chosen_action_q_tot_vals = self.mixer(chosen_action_qvals_clone, batch["state"][:, :-1])
            target_max_q_tot_vals = self.target_mixer(target_max_qvals_clone, batch["state"][:, 1:])

        if self.args.standardise_returns:
            target_max_q_tot_vals = target_max_qvals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean

        if self.args.differ_reward and self.args.env == "sc2":
            indi_terminated = batch["indi_terminated"][:, :-1].float()
            targets = rewards + self.args.gamma * (1 - terminated) * target_max_q_tot_vals
            td_error = (chosen_action_q_tot_vals - targets.detach())
            masked_td_error = td_error * mask
            mixer_loss = (masked_td_error ** 2).sum() / mask.sum()
            self.optimiser.zero_grad()
            chosen_action_qvals_clone.retain_grad()
            chosen_action_q_tot_vals.retain_grad()
            mixer_loss.backward()

            grad_l_qtot = chosen_action_q_tot_vals.grad.repeat(1, 1, self.args.n_agents) + 1e-8
            grad_l_qi = chosen_action_qvals_clone.grad
            grad_qtot_qi = th.clamp(grad_l_qi / grad_l_qtot, min=-10, max=10)
            mixer_grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
            self.optimiser.step()
            q_rewards = self.cal_indi_reward(grad_qtot_qi, td_error, chosen_action_qvals, target_max_qvals,
                                             indi_terminated)
            q_rewards_clone = q_rewards.clone().detach()
            q_targets = q_rewards_clone + self.args.gamma * (1 - indi_terminated) * target_max_qvals
            q_td_error = (chosen_action_qvals - q_targets.detach())
            q_mask = batch["filled"][:, :-1].float().repeat(1, 1, self.args.n_agents)
            q_mask[:, 1:] = q_mask[:, 1:] * (1 - indi_terminated[:, :-1]) * (1 - terminated[:, :-1]).repeat(1, 1,
                                                                                                            self.args.n_agents)
            q_mask = q_mask.expand_as(q_td_error)

            masked_q_td_error = q_td_error * q_mask
            q_selected_weight, selected_ratio = self.select_trajectory(masked_q_td_error.abs(), q_mask, t_env)
            q_selected_weight = q_selected_weight.clone().detach()
            loss = (masked_q_td_error ** 2 * q_selected_weight).sum() / q_mask.sum()
        else:
            targets = rewards + self.args.gamma * (1 - terminated) * target_max_q_tot_vals.detach()

            if self.args.standardise_returns:
                self.ret_ms.update(targets)
                targets = (targets - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

            td_error = (chosen_action_q_tot_vals - targets.detach())

            mask = mask.expand_as(td_error)

            masked_td_error = td_error * mask

            loss = (masked_td_error ** 2).sum() / mask.sum()

        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (t_env - self.last_masknet_update_t) >= self.masknet_update_interval:
            info_nce_loss = self.mac.optimize_masknet()
            self.mask_net_optimiser.zero_grad()
            info_nce_loss.backward()
            self.mask_net_optimiser.step()
            self.last_masknet_update_t = t_env

        self.mac.cluster_hidden_states()
        if 'noar' in self.args.mac:
            self.target_mac.task_selector.update_tasks(self.mac.n_tasks)

        self.training_steps += 1
        if self.args.target_update_interval_or_tau > 1 and (self.training_steps - self.last_target_update_step) / self.args.target_update_interval_or_tau >= 1.0:
            self._update_targets_hard()
            self.last_target_update_step = self.training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        if t_env > self.args.task_cluster_update_start:
            self.mac.cluster_hidden_states()
            if 'noar' in self.args.mac:
                self.target_mac.task_selector.update_tasks(self.mac.n_tasks)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

    def _update_targets_hard(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def _update_targets_soft(self, tau):
        for target_param, param in zip(self.target_mac.parameters(), self.mac.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        if self.mixer is not None:
            for target_param, param in zip(self.target_mixer.parameters(), self.mixer.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))

    def cal_indi_reward(self, grad_qtot_qi, mixer_td_error, qi, target_qi, indi_terminated):
        grad_td = th.mul(grad_qtot_qi, mixer_td_error.repeat(1, 1, self.args.n_agents))
        reward_i = - grad_td + qi - self.args.gamma * (1 - indi_terminated) * target_qi
        return reward_i

    def select_trajectory(self, td_error, mask, t_env):
        if self.args.warm_up:
            if t_env / self.args.t_max <= self.args.warm_up_ratio:
                selected_ratio = t_env * (self.args.selected_ratio_end - self.args.selected_ratio_start) / (
                            self.args.t_max * self.args.warm_up_ratio) + self.args.selected_ratio_start
            else:
                selected_ratio = self.args.selected_ratio_end
        else:
            selected_ratio = self.args.selected_ratio

        if self.args.selected == 'all':
            return th.ones_like(td_error).cuda(), selected_ratio
        elif self.args.selected == 'greedy':
            valid_num = mask.sum().item()
            selected_num = int(valid_num * selected_ratio)
            td_reshape = td_error.reshape(-1)
            sorted_td, _ = th.topk(td_reshape, selected_num)
            pivot = sorted_td[-1]
            weight = th.where(td_error >= pivot, th.ones_like(td_error), th.zeros_like(td_error))
            return weight, selected_ratio
        elif self.args.selected == 'greedy_weight':
            valid_num = mask.sum().item()
            selected_num = int(valid_num * selected_ratio)
            td_reshape = td_error.reshape(-1)
            sorted_td, _ = th.topk(td_reshape, selected_num)
            pivot = sorted_td[-1]
            weight = th.where(td_error >= pivot, td_error - pivot, th.zeros_like(td_error))
            norm_weight = weight / weight.max()
            return norm_weight, selected_ratio
        elif self.args.selected == 'PER':
            memory_size = int(mask.sum().item())
            memory = PER_Memory(memory_size)
            for b in range(mask.shape[0]):
                for t in range(mask.shape[1]):
                    for na in range(mask.shape[2]):
                        pos = (b, t, na)
                        if mask[pos] == 1:
                            memory.store(td_error[pos].cpu().detach(), pos)
            selected_num = int(memory_size * selected_ratio)
            mini_batch, selected_pos, is_weight = memory.sample(selected_num)
            weight = th.zeros_like(td_error)
            for idxs, pos in enumerate(selected_pos):
                weight[pos] += is_weight[idxs]
            return weight, selected_ratio
        elif self.args.selected == 'PER_hard':
            memory_size = int(mask.sum().item())
            selected_num = int(memory_size * selected_ratio)
            return PER_Memory(self.args, td_error, mask).sample(selected_num), selected_ratio
        elif self.args.selected == 'PER_weight':
            memory_size = int(mask.sum().item())
            selected_num = int(memory_size * selected_ratio)
            return PER_Memory(self.args, td_error, mask).sample_weight(selected_num, t_env), selected_ratio