# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .base_learner import BaseLearner
import model
import torch
import torch.nn as nn
import torch.optim as optim

class TransformerLearner(BaseLearner):
    def __init__(self, optimizer, lr, model_type, vocsize, emsize, buffer_len, nhead, nhid, nlayers, dropout, learn_iterations, warmup, after_warmup):
        criterion = nn.CrossEntropyLoss()
        super(TransformerLearner, self).__init__(
            criterion, vocsize, learn_iterations)
        self.model = model.TransformerModel(
            vocsize, emsize, nhead, nhid, nlayers, dropout)
        self.dmodel = emsize
        if lr == 42:
            self.lr = self.dmodel**-0.5
        else:
            self.lr = lr
        self.step = 1
        self.warmup = warmup
        self.after_warmup = after_warmup
        self.buffer_len = buffer_len
        self.buffer = None
        kwargs = {}
        if optimizer == 'Adam':
            kwargs['betas'] = (0.9, 0.98)
            kwargs['eps'] = 1e-9
        lr = self.compute_lr()
        self.optimizer = getattr(optim, optimizer)(self.model.parameters(), lr=lr)

    def compute_lr(self):
        return self.lr * min(self.step**-0.5 if self.after_warmup == 'decrease' else self.warmup**-0.5,
                self.step*self.warmup**-1.5)
    
    def learn(self, *args):
        self.optimizer.param_groups[0]['lr'] = self.compute_lr()
        self.step += 1
        ret = super(TransformerLearner, self).learn(*args)
        return ret

    def predict(self, data, hidden):
        self.append_to_buffer(data)
        output = self.model(self.get_buffered_data())
        output = output[-data.size(0):,:]
        return output, hidden

    def append_to_buffer(self, data):
        if self.buffer is None:
            self.buffer = data.detach().clone()
        else:
            self.buffer = torch.cat([self.buffer, data], dim=0)
            self.buffer = self.buffer[-self.buffer_len:,:]

    def get_buffered_data(self):
        return self.buffer

    def generate(self, data, hidden):
        raise RuntimeError("Not implemented (because of missing buffering)")
        output = self.model(data)
        return output.view(-1, self.vocsize), None

    def train_model(self, loss, prediction, data, targets):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def get_num_parameters(self):
        return sum(p.view(-1).size(0) for p in self.model.parameters())

    def create_hidden_states(self, bsz):
        return None

    def train_mode(self):
        self.model.train()

    def evaluate_mode(self):
        self.model.eval()

