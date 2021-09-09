import torch
import wandb

import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix

from wandb.keras import WandbCallback


from src.data_preparation.preprocess import DataPreparator


class ModelTrainer:
    """Creates a model based on constructor parameters. Enables to start training using train_model()."""

    @staticmethod
    def setup_device():
        """Return torch cuda device if available, else fallback to CPU. TPU untested."""
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def to_categorical(output, num_classes):
        """ 1-hot encodes a tensor """
        return np.eye(num_classes, dtype='uint8')[output]

    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def set_dataset(self):
        preparator = DataPreparator(data_dir=self.data_dir)
        data, labels = preparator.prepare_dataset()
        self.data = data
        self.labels = labels
        self.dataset_params = [preparator.adjs, preparator.input_shape, preparator.classes]

    def train_model(self, model, is_wandb=True):
        wandb.login()
        wandb.init(project="seizure-detection",
                   entity="domiomi3",
                   config={
            "architecture": "CNN-LSTM",
            "dataset": "Temple University Hospital EEG Seizure Dataset'@'v1.5.2 ",
        })

        total_acc = 0
        skf = StratifiedKFold(n_splits=self.k_fold_splits, shuffle=True)
        for train_idx, test_idx in skf.split(self.data[0], self.labels):
            x_train, y_train = [d[train_idx] for d in self.data], self.labels[train_idx]
            x_test, y_test = [d[test_idx] for d in self.data], self.labels[test_idx]

            y_train_cat = self.to_categorical(y_train, len(np.unique(y_train)))
            y_test_cat = self.to_categorical(y_test, len(np.unique(y_train)))

            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            model.fit(x_train, y_train_cat, batch_size=32, epochs=50, shuffle=True, validation_split=0.1, callbacks=[WandbCallback()])
            y_pred = model.predict(x_test).argmax(axis=-1)
            self.evaluate(y_pred, y_test)
        wandb.finish()

    @staticmethod
    def evaluate(y_pred, y_test):
        acc = accuracy_score(y_test, y_pred)
        print("Fold accuracy: {:2.2f}".format(acc))
        matrix = confusion_matrix(y_test, y_pred)
        print("Fold confusion matrix: \n {}".format(matrix))