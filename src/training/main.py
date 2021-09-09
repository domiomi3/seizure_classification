import torch
import wandb
import torchmetrics

import numpy as np

from src.data_preparation.preprocess import DataPreparator
from models.cnn_dense import CNNDenseModel
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from tqdm import tqdm

data_path = "/home/dominika/ML Projects/MSc/seizure_detection/data/raw"

preparator = DataPreparator(data_dir=data_path)
data, labels = preparator.get_data(), preparator.get_labels()
adjs = preparator.adjs
window_input_shape = preparator.input_shape
num_classes = preparator.classes

batch_size = 32
input_shape = (batch_size, adjs) + window_input_shape
cnn_dense = CNNDenseModel(input_shape, num_classes)

criterion = CrossEntropyLoss()
optimizer = Adam(cnn_dense.parameters(), lr=1e-5)
metric = torchmetrics.Accuracy()

n_epochs = 50
wandb.login()
wandb.init(project="seizure-detection",
           entity="domiomi3",
           config={
               "architecture": "CNN-Dense",
               "dataset": "Temple University Hospital EEG Seizure Dataset'@'v1.5.2 ",
           })

wandb.watch(cnn_dense, log='all', log_freq=100)
wandb.run.name = "baseline_combined_lr=1e-51"

dataset_size = len(labels)
train_idx, val_idx = train_test_split(list(range(dataset_size)), test_size=0.3)

p_bar = tqdm(range(n_epochs))

for epoch in p_bar:
    p_bar.set_description("Epoch: " + str(epoch))

    loss_train = []
    acc_train = []
    loss_val = []
    acc_val = []
    dist = []

    for i in tqdm(range(int(len(train_idx)/batch_size)), desc="Training loop"):
        optimizer.zero_grad()

        indices = train_idx[i:i + batch_size]
        batch_y = labels[indices]
        batch_x = data[indices, :, :, :]

        # forward + backward + optimize
        pred_cls, pred_y = cnn_dense.predict(batch_x)
        loss = criterion(pred_y, batch_y)
        acc = metric(pred_y, batch_y)

        loss_train.append(loss.item())
        acc_train.append(acc.item())
        # dist.append([p, y] for p, y in zip(pred_y, batch_y))
        wandb.log({"pred": wandb.Histogram(pred_cls.detach().numpy())})
        wandb.log({"true": wandb.Histogram(batch_y.detach().numpy())})

        loss.backward()
        optimizer.step()

    with torch.no_grad():  # no gradient needed
        for j in tqdm(range(int(len(val_idx)/batch_size)), desc="Validation loop"):
            indices = val_idx[j:j + batch_size]
            val_batch_y = labels[indices]
            val_batch_x = data[indices, :, :, :]
            val_pred_y = cnn_dense.predict(val_batch_x)[1]
            val_loss = criterion(val_pred_y, val_batch_y)
            val_acc = metric(val_pred_y, val_batch_y)

            loss_val.append(val_loss.item())
            acc_val.append(val_acc.item())

    # dist_table = wandb.Table(data=dist, columns=["prediction", "truth"])
    wandb.log({"loss": np.mean(loss_train), "accuracy": np.mean(acc_train),
               "val_loss": np.mean(loss_val), "val_accuracy": np.mean(acc_val),
               "epoch": epoch})
    # wandb.log({'class distribution': wandb.plot.histogram(dist_table,"prediction", title="Seizure Type Distribution")})
