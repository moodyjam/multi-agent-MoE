import torch.nn as nn
import torch
from torch import optim

class Agent(nn.Module):
   def __init__(self, config, model, idx):
      super(Agent, self).__init__()
      self.config = config
      self.model = model
      self.idx = idx
      self.set_flattened_params()
      self.register_buffer("dual", torch.zeros_like(self.flattened_params))
      self.neighbor_params = []
      
   def set_flattened_params(self):
      self.flattened_params = torch.nn.utils.parameters_to_vector(self.parameters()).clone().detach()
   
   def get_flattened_params(self):
      return self.flattened_params
   
class MixtureOfExpertsAgent(nn.Module):
   def __init__(self, config, model, idx, id):
      super(MixtureOfExpertsAgent, self).__init__()
      self.config = config
      self.model = model
      self.idx = idx
      self.encoder_flattened = torch.nn.utils.parameters_to_vector(self.model.encoder.parameters()).clone().detach()
      self.register_buffer("dual", torch.zeros_like(self.encoder_flattened))
      self.neighbor_params = []
      self.id = id
      
   def set_flattened_params(self):
      self.encoder_flattened = torch.nn.utils.parameters_to_vector(self.model.encoder.parameters()).clone().detach()
   
   def get_flattened_params(self):
      return self.encoder_flattened
   
   def get_prototype(self):
      return self.model.prototype.clone().detach()

   
        