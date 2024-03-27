import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L
from util import create_graph
from copy import deepcopy
import torch
from agent import MixtureOfExpertsAgent
import torch.nn.functional as F
import numpy as np
from datamodule import DATASET_IDX_MAP

# From the code
class MNISTConvNet(nn.Module):
    """Implements a basic convolutional neural network with one
    convolutional layer and two subsequent linear layers for the MNIST
    classification problem.
    """

    def __init__(self, num_filters=3, kernel_size=5, linear_width=64, num_nodes=10, num_labels=10):
        super().__init__()
        conv_out_width = 28 - (kernel_size - 1)
        pool_out_width = int(conv_out_width / 2)
        fc1_indim = num_filters * (pool_out_width ** 2)

        # Our f network
        self.encoder = nn.Sequential(
            nn.Conv2d(1, num_filters, kernel_size, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten())
        
        # Our g network
        # self.gating = nn.Embedding(num_nodes, fc1_indim)
        self.gating = nn.Parameter(torch.randn(num_nodes, fc1_indim))
            
        # Our h network
        self.specialist = nn.Sequential(
            nn.Linear(fc1_indim, linear_width),
            nn.ReLU(inplace=True),
            nn.Linear(linear_width, num_labels),
            nn.LogSoftmax(dim=1),
        )
        
    def encode(self, x):
        return self.encoder(x)
    
    def get_scores(self, x):
        print()

    def forward(self, x):
        return self.seq(x)

# define the LightningModule
class MultiAgentMoE(L.LightningModule):
    def __init__(self,
                 agent_config,
                 graph_type="complete",
                 fiedler_value=None,
                 num_classes=10,
                 B=2,
                 rho=0.5,
                 lr_start=0.005,
                 lr_finish=0.0005,
                 oits=2000,
                 rho_update=0.0003,
                 topk = 2,
                 num_labels = 10):
        super().__init__()
        self.num_nodes = len(agent_config)
        self.G, self.G_connectivity = create_graph(num_nodes = self.num_nodes,
                                                   graph_type = graph_type,
                                                   target_connectivity = fiedler_value)
        
        base_model = MNISTConvNet(num_labels=num_labels)
        self.agent_id_to_idx = {agent["id"]: i for i, agent in enumerate(agent_config)}

        # Initialize the networks for each agent
        self.agent_config = agent_config
        self.agents = nn.ModuleDict({agent["id"]: MixtureOfExpertsAgent(config=agent_config[i],
                                          model=deepcopy(base_model),
                                          idx=i)
                                          for i, agent in enumerate(self.agent_config)})
        
        self.dataset_names = list(DATASET_IDX_MAP.keys())
        
        self.automatic_optimization = False
        self.criterion = torch.nn.NLLLoss()
        self.rho = rho
        
        self.lr_schedule = torch.linspace(
                lr_start,
                lr_finish,
                oits,
            )
        
        self.val_accs = {dataset_name: 0.0 for dataset_name in self.dataset_names}
        self.val_acc_counts = {dataset_name: 0 for dataset_name in self.dataset_names}
        
        self.manual_global_step=0
        
        self.save_hyperparameters()

    def calculate_loss(self, own_loss, theta_reg, curr_agent, scores, neighbor_losses):
        gating_loss = (scores[:, curr_agent.idx] * own_loss.item()).mean()
        for neighbor_idx in neighbor_losses:
            gating_loss += (scores[:, neighbor_idx] * neighbor_losses[neighbor_idx]).mean()
            
        theta = torch.cat([torch.nn.utils.parameters_to_vector(curr_agent.model.encoder.parameters()),
                           torch.nn.utils.parameters_to_vector(curr_agent.model.gating)], dim=-1)
        reg = torch.sum(torch.square(torch.cdist(theta.reshape(1, -1), theta_reg)))
        return own_loss + torch.dot(curr_agent.dual, theta) + self.rho * reg + gating_loss
        
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward

        for agent_id in self.agents:
            curr_agent = self.agents[agent_id]
            curr_agent.set_flattened_params()
        self.rho *= (1 + self.hparams.rho_update) 
        
        for agent_idx, agent_id in enumerate(self.agents):
            curr_agent = self.agents[agent_id]
            curr_agent.opt = optim.Adam(curr_agent.parameters(), lr=self.lr_schedule[self.manual_global_step])

        # Technically this can be done in parallel
        all_losses = []
        for agent_idx, agent_id in enumerate(self.agents):
            curr_agent = self.agents[agent_id]     
            neighbor_indices = list(self.G.neighbors(curr_agent.idx))
            neighbor_params = torch.stack([self.agents[self.agent_config[idx]['id']].get_flattened_params() for idx in neighbor_indices])
            theta = curr_agent.get_flattened_params()
            curr_agent.dual += self.rho * (theta - neighbor_params).sum(0)
            theta_reg = (theta + neighbor_params) / 2
            curr_batch = batch[curr_agent.idx]
            x, y, names = curr_batch
            split_size = x.shape[0] // self.hparams.B
            
            if agent_idx == 0:
                self.log("learning_rate", self.lr_schedule[self.manual_global_step], logger=True)
            
            # Loop through B times for the current agent
            for tau in range(self.hparams.B):
                x_split = x[tau*split_size:(tau+1)*split_size]
                y_split = y[tau*split_size:(tau+1)*split_size]
                
                curr_agent.opt.zero_grad()
                x_encoded = curr_agent.model.encoder(x_split)
                scores = F.softmax(x_encoded @ curr_agent.model.gating.T, dim=-1)
                x_out = curr_agent.model.specialist(x_encoded)
                own_loss = self.criterion(x_out, y_split)
                             
                # Added optimization loop for neighbor losses  
                # Here more communication would occur 
                neighbor_losses = dict()
                for neighbor_idx in neighbor_indices:
                    neighbor_agent = self.agents[self.agent_config[neighbor_idx]['id']]
                    neighbor_agent.opt.zero_grad()
                    neighbor_pred = neighbor_agent.model.specialist(x_encoded.clone().detach())
                    neighbor_loss = self.criterion(neighbor_pred, y_split)
                    neighbor_losses[neighbor_idx] = neighbor_loss.item()
                    self.manual_backward(neighbor_loss)
                    neighbor_agent.opt.step()
                
                loss = self.calculate_loss(own_loss, theta_reg, curr_agent=curr_agent, scores=scores, neighbor_losses=neighbor_losses)
                self.manual_backward(loss)
                curr_agent.opt.step()

            self.log(f"{agent_id}_train_loss", loss, logger=True)
            all_losses.append(loss.item())
        self.log(f"train_loss", np.mean(all_losses), logger=True, prog_bar=True)
        
        self.manual_global_step += 1

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, datasets = batch
        all_encoded = []
        all_scores = []
        for agent_id in self.agents:
            curr_agent = self.agents[agent_id]
            x_encoded = curr_agent.model.encoder(x)
            all_encoded.append(x_encoded)
            scores = x_encoded @ curr_agent.model.gating.T
            all_scores.append(scores)
            
        mean_scores = torch.stack(all_scores).mean(0)
        mean_encoded = torch.stack(all_encoded).mean(0)

        topk = torch.max(mean_scores, dim=-1) # Just output of the best model for now
        preds = torch.ones_like(topk.indices) * -1

        # For each datapoint, use each of the specialists
        for agent_id in self.agents:
            curr_agent = self.agents[agent_id]
            curr_mask = curr_agent.idx == topk.indices
            logits = curr_agent.model.specialist(mean_encoded[curr_mask])
            curr_preds = torch.argmax(logits, dim=-1)
            preds[curr_mask] = curr_preds
            self.log(f'{agent_id}_specialist_use', curr_mask.sum().item(), on_epoch=True, logger=True)

        for dataset_idx, dataset in enumerate(self.dataset_names):
            mask = (datasets == dataset_idx)
            if mask.sum() > 0:
                self.val_accs[self.dataset_names[dataset_idx]] += torch.sum(y[mask] == preds[mask]).float()
                self.val_acc_counts[self.dataset_names[dataset_idx]] += len(y[mask])

    def on_validation_epoch_end(self):
        # Compute average loss over the epoch

        for dataset_idx, dataset_name in enumerate(self.dataset_names):
            if self.val_acc_counts[dataset_name] > 0:
                dataset_acc = self.val_accs[dataset_name] / self.val_acc_counts[dataset_name]

                # Log the average loss
                self.log(f'{dataset_name}_val_acc', dataset_acc, logger=True, prog_bar=True)

        # Reset for the next epoch
        self.val_accs = {dataset_name: 0.0 for dataset_name in self.dataset_names}
        self.val_acc_counts = {dataset_name: 0 for dataset_name in self.dataset_names}


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

        
        