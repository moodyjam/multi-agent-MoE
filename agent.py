import torch.nn as nn
import torch
from torch import optim
from resnet import ResNet, BasicBlock
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
      self.model = ResNet(BasicBlock, [3, 3, 3], num_labels)
      self.encoder = encoder
      self.idx = idx
      self.prototype = nn.Parameter(prototypes[idx])
      self.encoder_flattened = torch.nn.utils.parameters_to_vector(self.encoder.parameters()).clone().detach()
      self.register_buffer("dual", torch.zeros_like(self.encoder_flattened))
      self.register_buffer("all_prototypes", prototypes.clone().detach())
      self.register_buffer("excluded_indices", torch.tensor([i for i in range(num_agents) if i != self.idx]))
      self.update_timestamps = {agent_idx: 0 for agent_idx in range(num_agents)}
      self.id = id
      
   def set_flattened_params(self):
      self.encoder_flattened = torch.nn.utils.parameters_to_vector(self.encoder.parameters()).clone().detach()
   
   def get_flattened_params(self):
      return self.encoder_flattened
   
   def update_other_prototypes(self, other_prototypes, other_update_timestamps):
      for agent_idx, prototype in enumerate(other_prototypes):
         if other_update_timestamps[agent_idx] > self.update_timestamps[agent_idx]:
            # Update the prototype with the 
            self.all_prototypes[agent_idx] = prototype
            # Update the timestamp with the later timestamp
            self.update_timestamps[agent_idx] = other_update_timestamps[agent_idx]

   def update_own_prototype(self, step):
      self.all_prototypes[self.idx] = self.prototype.clone().detach()
      self.update_timestamps[self.idx] = step
   
   def get_all_prototypes(self):
      return self.all_prototypes, self.update_timestamps
   
   def get_all_other_prototypes(self):
      return self.all_prototypes[self.excluded_indices]
   
   def get_prototype(self):
      own_prototype = self.prototype.clone().detach()
      return own_prototype

   
        