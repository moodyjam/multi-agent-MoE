import wandb

from datamodule import ModularDataModule
from lightning.pytorch.loggers import WandbLogger
from run import load_config, get_model

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

def main():
     # Initialize a wandb run
    fixed_config = load_config('configs/dinno_4_datasets.yaml')

    with wandb.init() as run:
        dynamic_config = run.config

        # Merge the two configurations, giving preference to wandb parameters
        # This allows overriding fixed parameters with wandb parameters if necessary
        config = {**fixed_config, **dict(dynamic_config)}

        model = get_model(config)
        
        wandb_logger = WandbLogger(project=config["project"], log_model='all')

        # Define the datamodule here
        datamodule = ModularDataModule(
            batch_size=config['batch_size'],
            agent_config=config['agents'],
            num_workers=config['num_workers'],
        )

        checkpoint_callback = ModelCheckpoint(monitor="CIFAR-10_val_acc", mode="max")

        # Trainer setup
        trainer = Trainer(
            max_steps=config['max_steps'],
            devices=config['gpus'] if config['gpus'] > 0 else None,
            accelerator=config['accelerator'],
            callbacks=[checkpoint_callback],
            logger=wandb_logger,
        )
        
        # Training and testing
        trainer.fit(model, datamodule=datamodule)

        wandb.finish()  # Optional, to explicitly end the wandb run

if __name__=='__main__':

    main()