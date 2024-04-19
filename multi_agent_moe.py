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
from models import SimpleImageEncoder
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
                 prototype_dim = 128,
                 routing_weight = 1.0,
                 dinno_weight = 1.0,
                 data_routing_thresh = 0.5,
                 K=2,
                 use_max_diff = True):
        
        super().__init__()
        self.num_nodes = len(agent_config)
        self.G, self.G_connectivity = create_graph(num_nodes = self.num_nodes,
                                                   graph_type = graph_type,
                                                   target_connectivity = fiedler_value)
        
        base_encoder = SimpleImageEncoder(prototype_dim) # ResNetEncoder(BasicBlock, [3, 3, 3], prototype_dim)
        self.agent_id_to_idx = {agent["id"]: i for i, agent in enumerate(agent_config)}

        base_prototypes = torch.rand(self.num_nodes, prototype_dim) * 2 - 1

        # Initialize the networks for each agent
        self.agent_config = agent_config
        self.agents = nn.ModuleDict({agent["id"]: MixtureOfExpertsAgent(num_agents=self.num_nodes,
                                          encoder=deepcopy(base_encoder),
                                          idx=i,
                                          id=agent["id"],
                                          prototypes=base_prototypes,
                                          num_labels=num_labels) # Change this if needed to adjust for a different encoder
                                          for i, agent in enumerate(self.agent_config)})
        
        self.dataset_names = list(DATASET_IDX_MAP.keys())
        
        self.automatic_optimization = False
        self.criterion = torch.nn.NLLLoss()
        self.routing_criterion = torch.nn.MSELoss()
        self.rho = rho
        
        if oits > 10000:
            self.lr_schedule = torch.cat([torch.linspace(lr_start, lr_finish, 10000), torch.ones(size=(oits-10000,))*lr_finish])
        else:
            self.lr_schedule = torch.linspace(lr_start, lr_finish, oits)
        
        self.val_accs = {dataset_name: 0.0 for dataset_name in self.dataset_names}
        self.val_acc_counts = {dataset_name: 0 for dataset_name in self.dataset_names}

        self.val_accs_oracle = {dataset_name: 0.0 for dataset_name in self.dataset_names}
        self.val_acc_counts_oracle = {dataset_name: 0 for dataset_name in self.dataset_names}

        self.agent_routing_accs = {agent["id"]: 0.0 for agent in self.agent_config}
        self.agent_routing_counts = {agent["id"]: 0 for agent in self.agent_config}
        
        self.manual_global_step=0
        
        self.save_hyperparameters()

    def calculate_loss(self, x, y, theta_reg, curr_agent, log=False):
        x_encoded = curr_agent.encoder(x)
        x_out = curr_agent.model(x_encoded)
        aux_loss = self.criterion(x_out, y)
        
        # Here learn prototype vectors
        routing_logits = (F.normalize(curr_agent.prototypes, dim=-1) @ F.normalize(x_encoded.clone().detach(),dim=-1).T).T
        routing_loss = self.routing_criterion(routing_logits[:, curr_agent.idx], torch.ones_like(y).float())

        # FIXME Change routing back to the way it previously was
        data_routing_map = routing_logits > self.hparams.data_routing_thresh
        data_routing_map[:, curr_agent.idx] = False

        # Store the data that we are going to route to the other agents
        curr_agent.update_data_routing(data_routing_map, x_encoded.clone().detach(), y)

        encoder_flattened = torch.nn.utils.parameters_to_vector(curr_agent.encoder.parameters())
        prototypes_flattened = torch.nn.utils.parameters_to_vector(curr_agent.prototypes)
        theta = torch.cat([encoder_flattened, prototypes_flattened])
        reg = torch.sum((theta.reshape(1, -1) - theta_reg)**2)
        dual_loss = torch.dot(curr_agent.dual, theta)
        reg_loss = self.rho * reg
        
        if log:
            self.log(f"{curr_agent.id}_train_routing_loss", routing_loss, logger=True)
            # self.log(f"{curr_agent.id}_train_diff_loss", diff_loss, logger=True)
            self.log(f"{curr_agent.id}_train_aux_loss", aux_loss, logger=True)
            self.log(f"{curr_agent.id}_train_dual_loss", dual_loss, logger=True)
            self.log(f"{curr_agent.id}_train_reg_loss", reg_loss, logger=True)
        
        # Should that dot product be negative?
        return aux_loss + self.hparams.routing_weight * routing_loss + self.hparams.dinno_weight * (dual_loss + reg_loss)
        
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward

        for agent_id in self.agents:
            curr_agent = self.agents[agent_id]
            curr_agent.set_flattened_params()
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

            for neighbor_idx in neighbor_indices:
                neighbor_agent = self.agents[self.agent_config[neighbor_idx]['id']]
                curr_agent.update_data_routing_dict(neighbor_agent.get_data_routing())

            theta = curr_agent.get_flattened_params()
            curr_agent.dual += self.rho * (theta - neighbor_params).sum(0)
            theta_reg = (theta + neighbor_params) / 2
            curr_batch = batch[curr_agent.idx]
            x, y, names, agent_indices = curr_batch
            split_size = x.shape[0] // self.hparams.B
            
            if agent_idx == 0:
                self.log("learning_rate", self.lr_schedule[self.manual_global_step], logger=True)
            
            # Loop through B times for the current agent
            for tau in range(self.hparams.B):
                x_split = x[tau*split_size:(tau+1)*split_size]
                y_split = y[tau*split_size:(tau+1)*split_size]
                
                curr_agent.opt.zero_grad()
                loss = self.calculate_loss(x_split,
                                           y_split,
                                           theta_reg,
                                           curr_agent = curr_agent,
                                           log = False)
                self.manual_backward(loss)
                curr_agent.opt.step()

            # Now train on the data routed to us
            curr_agent.store_data_routing(self.manual_global_step)
            routed_data = []
            routed_labels = []
            for other_idx in curr_agent.data_routing_dict:
                if other_idx == curr_agent.idx:
                    continue # Don't retrain on our own data
                mask = curr_agent.data_routing_dict[other_idx]["data_routing_map"][:,curr_agent.idx]
                masked_data = curr_agent.data_routing_dict[other_idx]["encoded_data"][mask]
                masked_labels = curr_agent.data_routing_dict[other_idx]["encoded_data_labels"][mask]
                if masked_data.shape[0] > 0:
                    routed_data.append(masked_data)
                    routed_labels.append(masked_labels)
            
            if len(routed_data) > 0:
                print()
                

            # Data routing.

            all_losses.append(loss.item())
        self.log(f"train_loss", np.mean(all_losses), logger=True, prog_bar=True)
        
        self.manual_global_step += 1
        if self.manual_global_step >= self.trainer.max_steps:
            self.trainer.should_stop = True  # This will gracefully end the training

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, datasets, agent_indices = batch

        # FIXME Change this back to the way it previously was

        b = x.shape[0] 
        
        encode_agent = self.agents[self.agent_config[0]["id"]]
        x_encoded = encode_agent.encoder(x)
        
        # Routing mechanism
        sims = F.normalize(encode_agent.prototypes,dim=-1) @ F.normalize(x_encoded,dim=-1).T
        routing_topk = torch.topk(sims, k=self.hparams.K, dim=0)

        # This gets the maximum accuracy we could achieve by perfect routing
        oracle_preds = torch.ones_like(y) * -1

        for agent_id in self.agents:
            curr_agent = self.agents[agent_id]
            oracle_mask = curr_agent.idx == agent_indices

            if oracle_mask.sum() > 0:
                for k_index in range(self.hparams.K):
                    if k_index > 0:
                        curr_mask = curr_mask | (curr_agent.idx == routing_topk.indices[k_index])
                    else:
                        curr_mask = curr_agent.idx == routing_topk.indices[k_index]

                self.agent_routing_accs[agent_id] += (curr_mask & oracle_mask).sum()
                self.agent_routing_counts[agent_id] += oracle_mask.sum()

                # Evaluate on the oracle mask
                logits_oracle = curr_agent.model(x_encoded[oracle_mask])
                curr_oracle_preds = torch.argmax(logits_oracle, dim=-1)
                oracle_preds[oracle_mask] = curr_oracle_preds

        # For each datapoint, use each of the specialists
        pred_logits = torch.zeros((b, self.hparams.num_labels)).type_as(logits_oracle)
        agent_use = {agent_id: 0 for agent_id in self.agents}
        for k_index in range(self.hparams.K):
            for agent_id in self.agents:
                curr_agent = self.agents[agent_id]
                curr_mask = curr_agent.idx == routing_topk.indices[k_index]

                if curr_mask.sum() > 0:
                    logits = curr_agent.model(x_encoded[curr_mask])
                    pred_logits[curr_mask] += routing_topk.values[k_index][curr_mask].unsqueeze(1) * logits

                agent_use[agent_id] += curr_mask.sum().item()

        preds = torch.argmax(pred_logits, dim=-1)

        assert pred_logits.sum()
        assert not torch.any(pred_logits.sum(-1) == 0), "Some prediction logits not assigned. This is unexpected."

        for agent_id in self.agents:
            self.log(f'{agent_id}_use', agent_use[agent_id], on_epoch=True, logger=True)

        for dataset_idx, dataset in enumerate(self.dataset_names):
            mask = (datasets == dataset_idx)
            if mask.sum() > 0:
                self.val_accs[self.dataset_names[dataset_idx]] += torch.sum(y[mask] == preds[mask]).float()
                self.val_acc_counts[self.dataset_names[dataset_idx]] += len(y[mask])

                self.val_accs_oracle[self.dataset_names[dataset_idx]] += torch.sum(y[mask] == oracle_preds[mask]).float()
                self.val_acc_counts_oracle[self.dataset_names[dataset_idx]] += len(y[mask])

    def on_validation_epoch_end(self):
        # Compute average loss over the epoch

        for dataset_idx, dataset_name in enumerate(self.dataset_names):
            if self.val_acc_counts[dataset_name] > 0:
                dataset_acc = self.val_accs[dataset_name] / self.val_acc_counts[dataset_name]

                # Log the accuracy
                self.log(f'{dataset_name}_val_acc', dataset_acc, logger=True, prog_bar=True)

            if self.val_acc_counts_oracle[dataset_name] > 0:
                dataset_acc = self.val_accs_oracle[dataset_name] / self.val_acc_counts_oracle[dataset_name]
                self.log(f'{dataset_name}_val_acc_oracle', dataset_acc, logger=True)

        total_routing_accs = []
        for agent in self.agent_config:
            if self.agent_routing_counts[agent["id"]] > 0:
                agent_routing_acc = self.agent_routing_accs[agent["id"]] / self.agent_routing_counts[agent["id"]]
                self.log(f'{agent["id"]}_routing_acc', agent_routing_acc, logger=True)
                total_routing_accs.append(agent_routing_acc.item())

        # Calculate the total routing accuracy
        self.log(f'total_routing_acc', np.mean(total_routing_accs), logger=True)

        # Reset for the next epoch
        self.val_accs = {dataset_name: 0.0 for dataset_name in self.dataset_names}
        self.val_acc_counts = {dataset_name: 0 for dataset_name in self.dataset_names}

        self.val_accs_oracle = {dataset_name: 0.0 for dataset_name in self.dataset_names}
        self.val_acc_counts_oracle = {dataset_name: 0 for dataset_name in self.dataset_names}

        self.agent_routing_accs = {agent["id"]: 0.0 for agent in self.agent_config}
        self.agent_routing_counts = {agent["id"]: 0 for agent in self.agent_config}


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

        
        