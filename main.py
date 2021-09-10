import wandb
import numpy as np
import warnings
from tqdm import tqdm

from src.data_preparation.data_handler import DataHandler
from src.models import ModelMapping
from src.utils.utils import default_argument_parser, set_seed

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torchmetrics import Accuracy

warnings.simplefilter("ignore", category=FutureWarning)


def setup_wandb(entity, model, run_name):
    wandb.login()
    wandb.init(project="seizure-detection",
               entity=entity,
               config={
                   "architecture": model.get_name(),
                   "dataset": "Temple University Hospital EEG Seizure Dataset'@'v1.5.2 ",
               })

    wandb.watch(model, log='all', log_freq=100)
    wandb.run.name = run_name


def train(model, loader, optimizer, criterion, metric):
    t = tqdm(loader)
    loss_train = []
    acc_train = []
    for batch_index, (inputs, labels) in enumerate(t):
        optimizer.zero_grad()

        pred_cls, pred_y = model.predict(inputs)
        loss = criterion(pred_y, labels)
        acc = metric(pred_y, labels)

        loss_train.append(loss.item())
        acc_train.append(acc.item())

        wandb.log({"pred": wandb.Histogram(pred_cls.detach().numpy())}, commit=False)
        wandb.log({"true": wandb.Histogram(labels.detach().numpy())}, commit=False)

        loss.backward()
        optimizer.step()

        t.set_description('(=> Train) Accuracy: {:.4f}, Loss: {:.4f}'.format(acc, loss))

    return {"loss": np.mean(loss_train), "accuracy": np.mean(acc_train)}


def validate(model, loader, criterion, metric):
    t = tqdm(loader)
    loss_val = []
    acc_val = []
    with torch.no_grad():
        for batch_index, (inputs, labels) in enumerate(t):
            _, val_pred_y = model.predict(inputs)
            loss = criterion(val_pred_y, labels)
            acc = metric(val_pred_y, labels)

            loss_val.append(loss.item())
            acc_val.append(acc.item())

            t.set_description('(=> Test) Accuracy: {:.4f}, Loss: {:.4f}'.format(acc, loss))

    return {"val_loss": np.mean(loss_val), "val_accuracy": np.mean(acc_val)}


def run_experiment(args):
    set_seed(args.random_seed)
    data_handler = DataHandler(data_dir=args.data_dir,
                               batch_size=args.batch_size,
                               data_split=args.data_split,
                               shuffle_dataset=args.shuffle_dataset,
                               weighted_sampler=args.weighted_sampler,
                               model_name=args.model_name,
                               dynamic_window=args.num_models,
                               bandpass_freqs=[args.bandpass_min, args.bandpass_max]
                               )
    train_loader, valid_loader = data_handler.get_data_loaders()

    model = ModelMapping[args.model_name](input_shape=data_handler.get_input_shape(),
                                          num_classes=data_handler.get_num_classes())

    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)
    metric = Accuracy()

    setup_wandb(entity=args.wandb_entity,
                model=model,
                run_name=args.run_name)

    t = tqdm(range(args.n_epochs))
    for epoch in t:
        train_dict = train(model, train_loader, optimizer, criterion, metric)
        valid_dict = validate(model, valid_loader, criterion, metric)

        wandb.log(train_dict, step=epoch, commit=False)
        wandb.log(valid_dict, step=epoch, commit=False)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    options = vars(args)
    print(options)

    run_experiment(args)
