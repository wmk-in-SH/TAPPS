import torch as th
import torch.nn as nn
import torch.nn.functional as F


class AttentionNetwork(nn.Module):
    def __init__(self, task_dim):
        super(AttentionNetwork, self).__init__()
        self.Q = nn.Linear(task_dim, task_dim)
        self.K = nn.Linear(task_dim, task_dim)
        self.V = nn.Linear(task_dim, task_dim)

    def forward(self, task_embeddings):
        if task_embeddings.dim() == 2:
            task_embeddings = task_embeddings.unsqueeze(0)

        Q = self.Q(task_embeddings)
        K = self.K(task_embeddings)
        V = self.V(task_embeddings)

        attention_scores = th.matmul(Q, K.transpose(1, 2)) / th.sqrt(th.tensor(K.size(-1), dtype=th.float32))
        attention_weights = F.softmax(attention_scores, dim=-1)

        attention_output = th.matmul(attention_weights, V)
        return attention_weights, attention_output


class MaskNetwork(nn.Module):
    def __init__(self, args, state_dim, task_latent, n_tasks):
        super(MaskNetwork, self).__init__()
        self.args = args
        self.temperature = args.temperature
        self.task_latent_dim = args.task_latent_dim
        self.mask_dim = args.rnn_hidden_dim
        self.n_tasks = n_tasks
        self.n_agents = args.n_agents
        self.batch_size = args.batch_size
        self.task_latent = task_latent
        self.device = args.device
        self.rnn_hidden_dim = args.rnn_hidden_dim

        self.fc1 = nn.Linear(state_dim, self.mask_dim)
        self.fc2 = nn.Linear(self.mask_dim, self.rnn_hidden_dim * self.rnn_hidden_dim)
        self.fc3 = nn.Linear(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.fc4 = nn.Linear(self.rnn_hidden_dim, self.mask_dim)

        self.atten_model = AttentionNetwork(self.rnn_hidden_dim).to(self.device)

    def to_cuda(self):
        self.fc1.cuda()
        self.fc2.cuda()
        self.fc3.cuda()
        self.fc4.cuda()

    def forward(self, state, task_embedding, n_tasks):
        state_pre = F.relu(self.fc1(state))
        weight = self.fc2(state_pre)
        weight = weight.view(-1, self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.n_tasks = n_tasks

        task_embedding = task_embedding.view(1, self.n_tasks, self.rnn_hidden_dim)
        x = th.bmm(task_embedding, weight)
        x = F.relu(x)

        if self.args.use_tasks_attention:
            attention_matrix, attention_outputs = self.atten_model(self.task_latent)
            attention_matrix_expanded = attention_matrix.unsqueeze(0)
            attention_matrix_expanded = attention_matrix_expanded.transpose(1, 2)
            x = th.matmul(attention_matrix_expanded, x).squeeze(0)

        x = F.relu(self.fc3(x))
        mask_logits = self.fc4(x)
        mask = th.sigmoid(mask_logits)
        return mask.view(self.n_tasks, self.mask_dim)

    def info_nce_loss(self, anchor, positive, negatives):
        if positive.size(0) == 0 or negatives.size(0) == 0:
            return th.tensor(0.0, requires_grad=True).to(anchor.device)

        anchor_dot_positive = th.sum(anchor * positive, dim=-1)
        anchor_dot_positives = anchor_dot_positive / self.temperature

        negative_dot_anchor = th.matmul(negatives, anchor.unsqueeze(-1)).squeeze(-1)
        negative_dot_anchors = negative_dot_anchor / self.temperature

        logits = th.cat([anchor_dot_positives, negative_dot_anchors], dim=-1)
        logits = logits.view(-1, logits.size(-1))

        labels = th.zeros(logits.size(0), dtype=th.long).to(anchor.device)

        loss = F.cross_entropy(logits, labels)
        return loss

    def get_batch_positive_negative_samples(self, hidden_states, selected_tasks, task_id):
        batch_size, n_agents, embed_dim = hidden_states.size()

        positive_samples = []
        negative_samples = []

        for b in range(batch_size):
            pos_samples = []
            neg_samples = []
            for i in range(n_agents):
                if selected_tasks[b, i] == task_id:
                    pos_samples.append(hidden_states[b, i])
                else:
                    neg_samples.append(hidden_states[b, i])

            if pos_samples:
                positive_samples.append(th.stack(pos_samples))
            else:
                positive_samples.append(th.empty(0, embed_dim))

            if neg_samples:
                negative_samples.append(th.stack(neg_samples))
            else:
                negative_samples.append(th.empty(0, embed_dim))

        return positive_samples, negative_samples

    def optimize_masknet(self, task_embeddings, hidden_states, selected_tasks):
        hidden_states = hidden_states.reshape(self.batch_size, self.n_agents, -1)
        selected_tasks = selected_tasks.reshape(self.batch_size, self.n_agents, 1)
        task_embeddings = task_embeddings.unsqueeze(0).repeat(self.batch_size, 1, 1)

        info_nce_loss = 0
        for t in range(self.n_tasks):
            task_embedding = task_embeddings[:, t, :]
            positive_samples, negative_samples = self.get_batch_positive_negative_samples(hidden_states, selected_tasks,
                                                                                          t)
            for b in range(self.batch_size):
                info_nce_loss += self.info_nce_loss(task_embedding[b], positive_samples[b], negative_samples[b])

        info_nce_loss /= self.n_tasks
        return info_nce_loss
