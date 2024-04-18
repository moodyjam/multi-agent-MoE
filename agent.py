import torch.nn as nn
import torch
from torch import optim
from resnet import ResNet, BasicBlock
from models import MLP
import numpy as np

class Agent(nn.Module):
   def __init__(self, config, model, idx, num_labels):
      super(Agent, self).__init__()
      self.config = config
      self.model = ResNet(BasicBlock, [3, 3, 3], num_classes=num_labels)
      self.idx = idx
      self.set_flattened_params()
      self.register_buffer("dual", torch.zeros_like(self.flattened_params))
      self.neighbor_params = []
      
   def set_flattened_params(self):
      self.flattened_params = torch.nn.utils.parameters_to_vector(self.parameters()).clone().detach()
   
   def get_flattened_params(self):
      return self.flattened_params
   
class MixtureOfExpertsAgent(nn.Module):
   def __init__(self, num_agents, encoder, idx, id, num_labels, prototypes):
      super(MixtureOfExpertsAgent, self).__init__()
      self.model = MLP(num_labels, prototype_dim = prototypes.shape[-1]) # ResNet(BasicBlock, [3, 3, 3], num_labels)
      self.encoder = encoder
      self.idx = idx
      self.prototypes = nn.Parameter(prototypes)
      self.encoder_flattened = torch.nn.utils.parameters_to_vector(self.encoder.parameters()).clone().detach()
      self.prototypes_flattened = torch.nn.utils.parameters_to_vector(self.prototypes).clone().detach()
      self.register_buffer("dual", torch.zeros_like(torch.cat([self.encoder_flattened, self.prototypes_flattened])))
      self.id = id
      
   def set_flattened_params(self):
      self.encoder_flattened = torch.nn.utils.parameters_to_vector(self.encoder.parameters()).clone().detach()
      self.prototypes_flattened = torch.nn.utils.parameters_to_vector(self.prototypes).clone().detach()
   
   def get_flattened_params(self):
      return torch.cat([self.encoder_flattened, self.prototypes_flattened])

   
        