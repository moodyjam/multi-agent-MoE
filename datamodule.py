import os
import pickle
from typing import List
from lightning.pytorch import LightningDataModule
from torchvision import transforms, datasets
from torch.utils.data import Subset, DataLoader, ConcatDataset, default_collate
import numpy as np
from torch.utils.data import Dataset

DATASET_IDX_MAP = {
    "MNIST": 0,
    "FashionMNIST": 1,
    "CIFAR-10": 2,
    "SVHN": 3,
    "KMNIST": 4
}

DATASET_AGENT_MAP = {
    ("MNIST", True) : 0,
    ("MNIST", False) : 1,
    ("FashionMNIST", True) : 2,
    ("FashionMNIST", False) : 3,
    ("CIFAR-10", True) : 4,
    ("CIFAR-10", False) : 5,
    ("SVHN", True) : 6,
    ("SVHN", False) : 7,
    ("KMNIST", True) : 8,
    ("KMNIST", False) : 9,
}

def get_custom_collate_fn(dataset_name):
    def custom_collate_fn(batch):
        # Now, dataset_name is accessible here as a closure variable
        batch = default_collate(batch)  # This handles the basic collation
        
        # Assuming your batch is a list of tuples (data, label)
        batch_dict = {
            'data': batch[0],
            'labels': batch[1],
            'dataset_name': dataset_name
        }
        
        return batch_dict

    return custom_collate_fn

class DatasetWrapper(Dataset):
    def __init__(self, dataset, dataset_name, dataset_idx_map=None, dataset_agent_map=None):
        self.dataset = dataset
        self.dataset_name = dataset_name

        if dataset_idx_map is None:
            dataset_idx_map = DATASET_IDX_MAP
        if dataset_agent_map is None:
            dataset_agent_map = DATASET_AGENT_MAP
            
        self.dataset_idx_map = dataset_idx_map
        self.dataset_agent_map = dataset_agent_map
        self.labels_map = {'MNIST': 0, 'FashionMNIST': 10, 'CIFAR-10': 20, 'SVHN': 30, 'KMNIST': 40}
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, label = self.dataset[idx]
        agent_idx = self.dataset_agent_map[(self.dataset_name, label < 5)]
        label = label + self.labels_map[self.dataset_name]
        
        return data, label, self.dataset_idx_map[self.dataset_name], agent_idx


class ModularDataModule(LightningDataModule):
    """
    A Custom DataModule that can load in multiple datasets at once split into subsets by labels
    """
    def __init__(self, data_dir: str = "./datasets",
                 batch_size: int = 32,
                 agent_config: List = None,
                 cache_dir: str = "./cache",
                 validation_split: float = 0.1,
                 custom_transforms: dict = None,
                 num_workers: int = 10,
                 dataset_agent_map: dict = None,
                 dataset_idx_map: dict = None):
        
        super().__init__()
        self.data_dir = data_dir
        self.agent_config = agent_config
        self.dataset_agent_map = dataset_agent_map
        self.dataset_idx_map = dataset_idx_map

        self.dataset_names = []
        for i, agent in enumerate(agent_config):
            num_agent_datasets = len(agent['data'])
            for i in range(num_agent_datasets):
                if agent['data'][i]['dataset'] not in self.dataset_names:
                    self.dataset_names.append(agent['data'][i]['dataset'])

        self.cache_dir = cache_dir

        self.transforms = custom_transforms or {
            'MNIST': transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),
            'FashionMNIST': transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),
            'CIFAR-10': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),
            'CIFAR-100': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),
            'SVHN': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),
            'KMNIST': transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),
        }

        self.cache = self.load_cache()
        self.validation_split = validation_split
        self.save_hyperparameters()

    def load_cache(self):
        cache_path = os.path.join(self.cache_dir, 'data_module_cache.pkl')
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        else:
            return {}
        
    def load_dataset(self, dataset_name, train, transform):
        if dataset_name == 'MNIST':
            return datasets.MNIST(self.data_dir, train=train, download=True, transform=transform)
        elif dataset_name == 'FashionMNIST':
            return datasets.FashionMNIST(self.data_dir, train=train, download=True, transform=transform)
        elif dataset_name == 'CIFAR-10':
            return datasets.CIFAR10(self.data_dir, train=train, download=True, transform=transform)
        elif dataset_name == 'SVHN':
            return datasets.SVHN(self.data_dir, split= 'train' if train else 'test', download=True, transform=transform)
        elif dataset_name == 'KMNIST':
            return datasets.KMNIST(self.data_dir, train=train, download=True, transform=transform)
        elif dataset_name == 'CIFAR-100':
            return datasets.CIFAR100(self.data_dir, train=train, download=True, transform=transform)
        # Add Datasets Here
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

    def save_cache(self):
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        cache_path = os.path.join(self.cache_dir, 'data_module_cache.pkl')
        with open(cache_path, 'wb') as f:
            pickle.dump(self.cache, f)

    def prepare_data(self):
        # Download datasets if not already present
        for dataset_name in self.dataset_names:
            self.load_dataset(dataset_name, train=True, transform=self.transforms[dataset_name])

    def setup(self, stage=None):
        self.agent_datasets = {agent["id"]: dict() for agent in self.agent_config}
        train_datasets = {dataset_name: self.load_dataset(dataset_name,
                                                         train=True,
                                                         transform=self.transforms[dataset_name]) for dataset_name in self.dataset_names}
        test_datasets = {dataset_name: self.load_dataset(dataset_name,
                                                         train=False,
                                                         transform=self.transforms[dataset_name]) for dataset_name in self.dataset_names}

        for dataset_name in self.dataset_names:
            config_key = f"{dataset_name}"
            if config_key not in self.cache:
                print(f"Creating labels cache for {dataset_name}...")
                train_dataset = train_datasets[dataset_name]
                test_dataset = test_datasets[dataset_name]
                train_indices_dict, val_indices_dict, test_indices_dict = dict(), dict(), dict()

                # Each dataset must have a targets property that allows you to access all the targets in the dataset
                try:
                    targets = np.array(train_dataset.targets)
                except AttributeError:
                    targets = np.array(train_dataset.labels)
                
                unique_labels = np.unique(targets)
                    
                for label in unique_labels:
                    label = label.item()
                    train_indices_dict[label] = list(np.where(targets == label)[0])
                    test_indices_dict[label] = list(np.where(targets == label)[0])

                    # Shuffle indices
                    np.random.shuffle(train_indices_dict[label])
                    split = int(np.floor(self.validation_split * len(train_indices_dict[label])))
                    train_indices_dict[label] = train_indices_dict[label][split:]
                    val_indices_dict[label] = train_indices_dict[label][:split]

                self.cache[config_key] = {"train": train_indices_dict, "val": val_indices_dict, "test": test_indices_dict}

                # Update the cache file with new indices
                self.save_cache()

        # Loop through the different agents
        for agent in self.agent_config:
            num_agent_datasets = len(agent['data'])
            agent_id = agent['id']
            agent_train_dataset, agent_val_dataset, agent_test_dataset = [], [], []

            # Each agent can mix and match datasets
            for i in range(num_agent_datasets):
                dataset_name = agent['data'][i]['dataset']
                train_dataset = train_datasets[dataset_name]
                test_dataset = train_datasets[dataset_name]
                for label in agent['data'][i]['labels']:
                    agent_train_dataset.append(DatasetWrapper(Subset(train_dataset,
                                                                     self.cache[dataset_name]["train"][label]),
                                                                     dataset_name,
                                                                     dataset_idx_map=self.dataset_idx_map,
                                                                     dataset_agent_map=self.dataset_agent_map))
                    agent_val_dataset.append(DatasetWrapper(Subset(train_dataset,
                                                                   self.cache[dataset_name]["val"][label]),
                                                                   dataset_name,
                                                                   dataset_idx_map=self.dataset_idx_map,
                                                                   dataset_agent_map=self.dataset_agent_map))
                    agent_test_dataset.append(DatasetWrapper(Subset(test_dataset,
                                                                    self.cache[dataset_name]["test"][label]),
                                                                    dataset_name,
                                                                    dataset_idx_map=self.dataset_idx_map,
                                                                    dataset_agent_map=self.dataset_agent_map))

            self.agent_datasets[agent_id]["train"] = ConcatDataset(agent_train_dataset)
            self.agent_datasets[agent_id]["val"] = ConcatDataset(agent_val_dataset)
            self.agent_datasets[agent_id]["test"] = ConcatDataset(agent_test_dataset)

            if len(self.agent_datasets[agent_id]["train"]) == 0 or len(self.agent_datasets[agent_id]["val"]) == 0 or len(self.agent_datasets[agent_id]["test"]) == 0:
                raise ValueError

    def train_dataloader(self):
        # Combine MNIST and Fashion MNIST training subsets
        dataloaders = [DataLoader(self.agent_datasets[agent["id"]]["train"],
                                  batch_size=self.hparams.batch_size,
                                  shuffle=True,
                                  pin_memory=True,
                                  num_workers=self.hparams.num_workers,
                                  persistent_workers=True) for agent in self.agent_config]
        return dataloaders

    def val_dataloader(self):
        # Return a list of validation DataLoaders for MNIST and Fashion MNIST
        full_dataset = ConcatDataset([self.agent_datasets[agent["id"]]["val"] for agent in self.agent_config])
        dataloader = DataLoader(full_dataset,
                                batch_size=self.hparams.batch_size,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=self.hparams.num_workers,
                                persistent_workers=True)
        return dataloader
    
    def test_dataloader(self):
        # Return a list of validation DataLoaders for MNIST and Fashion MNIST
        full_dataset = ConcatDataset([self.agent_datasets[agent["id"]]["test"] for agent in self.agent_config])
        dataloader = DataLoader(full_dataset,
                                batch_size=self.hparams.batch_size,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=self.hparams.num_workers,
                                persistent_workers=True)
        return dataloader



