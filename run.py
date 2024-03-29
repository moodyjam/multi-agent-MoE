from datetime import datetime
import yaml
import argparse
from datamodule import ModularDataModule
from dinno import DiNNO
from multi_agent_moe import MultiAgentMoE
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    
def get_model(config):
    # Define your model here
    if config["algorithm"] == "dinno":
        model = DiNNO(agent_config=config["agents"],
                    graph_type=config["graph_type"],
                    fiedler_value=config["fiedler_value"],
                    oits=config["max_steps"])
    else:
        model = MultiAgentMoE(agent_config=config["agents"],
                    graph_type=config["graph_type"],
                    fiedler_value=config["fiedler_value"],
                    oits=config["max_steps"],
                    num_labels=config["num_labels"],
                    lr_start=config["lr_start"],
                    lr_finish=config["lr_finish"],
                    B=config["inner_steps"],
                    rho=config["rho"],
                    rho_update=config["rho_update"])
        
    return model

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Train a Model with PyTorch Lightning')
    parser.add_argument('--config', type=str, default='configs/mnist_config.yaml', help='Path to config file')

    args = parser.parse_args()

    config = load_config(args.config)

    # Define a run name
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    wandb_logger = WandbLogger(project=config["project"], id=f"{config['run_name']}_{formatted_time}", log_model='all')
    checkpoint_callback = ModelCheckpoint(monitor="CIFAR-10_val_acc", mode="max")

    model = get_model(config)

    # Define the datamodule here
    datamodule = ModularDataModule(
        batch_size=config['batch_size'],
        agent_config=config['agents'],
        num_workers=config['num_workers'],
    )

    # Define the trainer here
    trainer = Trainer(
        max_steps=config['max_steps'],
        devices=config['gpus'] if config['gpus'] > 0 else None,
        accelerator=config['accelerator'],
        callbacks=[checkpoint_callback],
        logger=wandb_logger,
        # limit_train_batches=.1,
    )
    
    trainer.fit(model, datamodule=datamodule)


