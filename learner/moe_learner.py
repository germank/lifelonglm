# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.optim as optim
import torch.nn as nn
import model
from collections import deque
import numpy as np
from model import repackage_hidden
import train_weights
from .base_mixture_learner import BaseMixtureOfRNNsLearner

class MoELearner(BaseMixtureOfRNNsLearner):
    def __init__(self, rnn_type, nin, nout, ninp, nhid, nlayers,
            max_memory_size, lr, batch_size, clip, 
            optimizer, module_normalization,
            train_weights_before_predict, weights_trainer, 
            learn_iterations, diverse_ensembling,
            tie_weights=False,
            w_window=20, dropout=0.2, is_cuda = True):
        super(MoELearner, self).__init__(rnn_type, nin, nout, ninp, nhid, nlayers,
            max_memory_size, lr, batch_size, clip, optimizer, 
            module_normalization, train_weights_before_predict, weights_trainer,
            learn_iterations,
            tie_weights, w_window, dropout, is_cuda=is_cuda)
        self.diverse_ensembling = diverse_ensembling
        self.warming = False

    def _initialize_model(self):
        for i in range(self.max_memory_size):
            self._create_new_rnn()
        self._create_optimizer()

    def train_model(self, loss, prediction, data, targets):
        outputs = self.last_outputs
        self.train_modules(data, outputs, targets)
        self.train_weights(data, outputs, targets,
                until_convergence=False)

    def train_modules(self, data, outputs, targets):
        weights = self.last_weights
        if self.diverse_ensembling and not self.warming:
            predictions = self.get_prediction_unreduced(weights.detach(), outputs)
            losses = self.get_loss_unreduced(predictions, targets.repeat(len(outputs)))
            losses = losses.reshape(len(outputs), -1)
            loss = losses.min(dim=0).values.mean()
        else:
            prediction = self.get_prediction(weights.detach(), outputs)
            loss = self.get_loss(prediction, targets)
        self.backpropagate_and_train_modules(loss)

    def backpropagate_and_train_modules(self, loss):
        n_modules = self.get_n_modules()
        self.optimizer.zero_grad()
        loss.backward()
        for i in range(n_modules):
            torch.nn.utils.clip_grad_norm_(self.rnns[i].parameters(), self.clip)
        self.optimizer.step()

    def _create_optimizer(self):
        params = []
        for i in range(self.max_memory_size):
            for j, param in enumerate(self.rnns[i].parameters()):
                params.append(param)
        opt_cls = getattr(torch.optim, self.optimizer_algorithm)
        self.optimizer = opt_cls(params, lr=self.lr)

    def reset_lr(self):
        self.optimizer.param_groups[0]['lr'] = self.lr

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def warmup_start(self):
        self.after_warmup_weights_trainer = self.weights_trainer
        self.weights_trainer = train_weights.FixedWeights(self.get_n_modules(),normalize=True)
        self.weights_trainer.n = self.get_n_modules()
        if next(self.after_warmup_weights_trainer.parameters()).is_cuda:
            self.weights_trainer.cuda()
        self.warming = True

    def warmup_end(self):
        if self.warming:
            self.weights_trainer = self.after_warmup_weights_trainer
            del self.after_warmup_weights_trainer
            self._create_optimizer()
            self.warming = False

