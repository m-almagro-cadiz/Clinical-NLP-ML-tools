#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

# Loss for reducing the relevance of already learnt instances
class FocalLoss(nn.Module):
   # Initialization
   def __init__(self, alpha=1, gamma=2, logits=False, reduce=True, pos_weight=(torch.Tensor([1]))):
      super(FocalLoss, self).__init__()
      self.alpha = alpha
      self.gamma = gamma
      self.logits = logits
      self.reduce = reduce
      self.pos_weight = pos_weight
   
   # Compute Cross Entropy with modifications
   def forward(self, inputs, targets):
      if self.logits:
         BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False, pos_weight=self.pos_weight)
      else:
         BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False, weight=self.pos_weight)
      pt = torch.exp(-BCE_loss)
      F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
      if self.reduce:
         return torch.mean(F_loss)
      else:
         return F_loss