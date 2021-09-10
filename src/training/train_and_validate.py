import wandb
import torch
import numpy as np

from tqdm import tqdm


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
