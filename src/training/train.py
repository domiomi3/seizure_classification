from abc import ABC, abstractmethod

import torch


class ModelTrainer(ABC):

    @staticmethod
    def setup_device():
        """Return torch cuda device if available, else fallback to CPU. TPU untested."""
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")\


    @abstractmethod
    def __init__(self,
                 task: str,
                 batch_size=1000,
                 learning_rate=0.001,
                 num_epochs=20,
                 num_workers=2,
                 optimizer="Adam",
                 ):

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.num_workers = num_workers
        self.optimizer = optimizer

        if task in ["seizure", "patient"]:
            self.task = task
        else:
            raise NameError("Invalid task")

    def train(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError
