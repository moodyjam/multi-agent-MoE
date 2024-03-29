import torch.nn as nn
import torch
from torch import optim
from resnet import ResNet, BasicBlock

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
   def __init__(self, config, encoder, idx, id, encoder_out_dim, num_labels):
      super(MixtureOfExpertsAgent, self).__init__()
      self.config = config
      self.model = ResNet(BasicBlock, [3, 3, 3], num_classes=num_labels) # Initialize a brand new resnet20 per agent
      self.encoder = encoder
      self.prototype = nn.Parameter(torch.randn(encoder_out_dim))
      self.idx = idx
      self.encoder_flattened = torch.nn.utils.parameters_to_vector(self.encoder.parameters()).clone().detach()
      self.register_buffer("dual", torch.zeros_like(self.encoder_flattened))
      self.neighbor_params = []
      self.id = id
      
   def set_flattened_params(self):
      self.encoder_flattened = torch.nn.utils.parameters_to_vector(self.encoder.parameters()).clone().detach()
   
   def get_flattened_params(self):
      return self.encoder_flattened
   
   def get_prototype(self):
      return self.prototype.clone().detach()

   
        