import wandb


def setup_wandb(entity, model_name, run_name):
    wandb.login()
    wandb.init(project="seizure-classification",
               entity=entity,
               config={
                   "architecture": model_name,
                   "dataset": "Temple University Hospital EEG Seizure Dataset'@'v1.5.2 ",
               })
    wandb.run.name = run_name
