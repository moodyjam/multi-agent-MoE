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
      # self.model = ResNet(BasicBlock, [3, 3, 3], num_classes=num_labels)
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
   def __init__(self, num_agents, encoder, idx, id, num_labels, prototypes):
      super(MixtureOfExpertsAgent, self).__init__()
      self.model = MLP(num_labels, prototype_dim = prototypes.shape[-1]) # ResNet(BasicBlock, [3, 3, 3], num_labels)
      self.encoder = encoder
      self.idx = idx
      self.prototypes = nn.Parameter(prototypes)
      self.set_flattened_params()
      self.register_buffer("dual", torch.zeros_like(torch.cat([self.encoder_flattened, self.prototypes_flattened])))
      self.id = id
      self.data_routing_map = []
      self.encoded_data = []
      self.encoded_data_labels = []
      self.data_routing_dict = dict()
      self.fc = nn.Linear(prototypes.shape[-1], num_labels)
      
   def set_flattened_params(self):
      self.encoder_flattened = torch.nn.utils.parameters_to_vector(self.encoder.parameters()).clone().detach()
      self.prototypes_flattened = torch.nn.utils.parameters_to_vector(self.prototypes).clone().detach()
   
   def get_flattened_params(self):
      return torch.cat([self.encoder_flattened, self.prototypes_flattened])
   
   def update_data_routing(self, data_routing_map, encoded_data, encoded_data_labels):
      self.data_routing_map.append(data_routing_map)
      self.encoded_data.append(encoded_data)
      self.encoded_data_labels.append(encoded_data_labels)

   def store_data_routing(self, timestep):
      data_routing_map = torch.cat(self.data_routing_map, dim=0)
      encoded_data = torch.cat(self.encoded_data, dim=0)
      encoded_data_labels = torch.cat(self.encoded_data_labels, dim=0)

      self.data_routing_dict[self.idx] = {"data_routing_map": data_routing_map,
                                          "encoded_data": encoded_data,
                                          "encoded_data_labels": encoded_data_labels,
                                          "timestep": timestep}
      
      self.encoded_data = []
      self.encoded_data_labels = []
      self.data_routing_map = []
      
   def get_data_routing(self):
      return self.data_routing_dict
   
   def update_data_routing_dict(self, other_data_routing_dict):
      # Update the current data to have the most relevant information
      for idx in other_data_routing_dict:
         try:
            if self.data_routing_dict[idx]["timestep"] < other_data_routing_dict[idx]["timestep"]:
               self.data_routing_dict[idx] = other_data_routing_dict[idx]
         except KeyError:
            self.data_routing_dict[idx] = other_data_routing_dict[idx]
      
   
      


   
        