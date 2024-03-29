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
from models import SimpleEncoder
from datamodule import DATASET_IDX_MAP

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
                 num_labels = 10,
                 tau = 10):
        
        super().__init__()
        self.num_nodes = len(agent_config)
        self.G, self.G_connectivity = create_graph(num_nodes = self.num_nodes,
                                                   graph_type = graph_type,
                                                   target_connectivity = fiedler_value)
        
        base_encoder = SimpleEncoder()
        self.agent_id_to_idx = {agent["id"]: i for i, agent in enumerate(agent_config)}

        encoder_out_dim = 16*5*5
        base_prototypes = torch.rand(self.num_nodes, encoder_out_dim) * 2 - 1

        # Initialize the networks for each agent
        self.agent_config = agent_config
        self.agents = nn.ModuleDict({agent["id"]: MixtureOfExpertsAgent(config=agent_config[i],
                                          encoder=deepcopy(base_encoder),
                                          idx=i,
                                          id=agent["id"],
                                          prototypes=base_prototypes,
                                          num_labels=num_labels) # Change this if needed to adjust for a different encoder
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

    def calculate_loss(self, x, y, theta_reg, curr_agent, neighbor_prototypes, log=False):
        x_encoded = curr_agent.encoder(x)
        x_out = curr_agent.model(x)
        aux_loss = self.criterion(x_out, y)
        
        # Here learn prototype vectors
        num_neighbors = neighbor_prototypes.shape[0]
        curr_prototype_normed = F.normalize(curr_agent.prototype, dim=-1).unsqueeze(0)
        x_encoded_normed = F.normalize(x_encoded, dim=-1)
        neighbor_prototypes_normed = F.normalize(neighbor_prototypes, dim=-1)
        sim_loss = -(curr_prototype_normed * x_encoded_normed).sum(1).mean()
        diff_loss = (neighbor_prototypes_normed.unsqueeze(1) * x_encoded_normed.unsqueeze(0)).sum(-1).mean(-1).max()
        
        theta = torch.nn.utils.parameters_to_vector(curr_agent.encoder.parameters())
        reg = torch.sum((theta.reshape(1, -1) - theta_reg)**2)
        dual_loss = torch.dot(curr_agent.dual, theta)
        reg_loss = self.rho * reg
        
        if log:
            self.log(f"{curr_agent.id}_train_sim_loss", sim_loss, logger=True)
            self.log(f"{curr_agent.id}_train_diff_loss", diff_loss, logger=True)
            self.log(f"{curr_agent.id}_train_aux_loss", aux_loss, logger=True)
            self.log(f"{curr_agent.id}_train_dual_loss", dual_loss, logger=True)
            self.log(f"{curr_agent.id}_train_reg_loss", reg_loss, logger=True)
        
        # Should that dot product be negative?
        return aux_loss + sim_loss + diff_loss + dual_loss + reg_loss
        
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward

        for agent_id in self.agents:
            curr_agent = self.agents[agent_id]
            curr_agent.set_flattened_params()
            curr_agent.update_own_prototype(self.manual_global_step)
        self.rho *= (1 + self.hparams.rho_update) 
        
        # Set the optimizers for each agent
        for agent_idx, agent_id in enumerate(self.agents):
            curr_agent = self.agents[agent_id]
            curr_agent.opt = optim.Adam(curr_agent.parameters(), lr=self.lr_schedule[self.manual_global_step])

        # Technically this can be done in parallel
        all_losses = []
        all_indices = np.arange(len(self.agents))
        for agent_idx, agent_id in enumerate(self.agents):
            curr_agent = self.agents[agent_id]     
            neighbor_indices = list(self.G.neighbors(curr_agent.idx))
            neighbor_params = torch.stack([self.agents[self.agent_config[idx]['id']].get_flattened_params() for idx in neighbor_indices])

            # Now align the prototypes from the neighbors
            for neighbor_idx in neighbor_indices:
                neighbor_agent = self.agents[self.agent_config[neighbor_idx]['id']]
                neighbor_prototypes, neighbor_timestamps = neighbor_agent.get_all_prototypes()
                curr_agent.update_other_prototypes(neighbor_prototypes, neighbor_timestamps)

            neighbor_prototypes = torch.stack([self.agents[self.agent_config[idx]['id']].get_prototype() for idx in all_indices if idx != curr_agent.idx])
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
                
                if tau == 0:
                    log = True
                else:
                    log = False
                
                curr_agent.opt.zero_grad()
                loss = self.calculate_loss(x_split,
                                           y_split,
                                           theta_reg,
                                           curr_agent=curr_agent,
                                           neighbor_prototypes=neighbor_prototypes,
                                           log = log)
                self.manual_backward(loss)
                curr_agent.opt.step()

            all_losses.append(loss.item())
        self.log(f"train_loss", np.mean(all_losses), logger=True, prog_bar=True)
        
        self.manual_global_step += 1
        if self.manual_global_step >= self.trainer.max_steps:
            self.trainer.should_stop = True  # This will gracefully end the training

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, datasets = batch
        
        encode_agent = self.agents[self.agent_config[0]["id"]]
        x_encoded = encode_agent.encoder(x)
        
        # Routing mechanism
        all_prototypes = torch.stack([self.agents[agent_id].get_prototype() for agent_id in self.agents])
        sims = F.normalize(all_prototypes, dim=-1) @ F.normalize(x_encoded, dim=-1).T
        routing = sims.argmax(0)
        preds = torch.ones_like(y) * -1

        # For each datapoint, use each of the specialists
        for agent_id in self.agents:
            curr_agent = self.agents[agent_id]
            curr_mask = curr_agent.idx == routing
            if curr_mask.sum() > 0:
                logits = curr_agent.model(x[curr_mask])
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

        
        