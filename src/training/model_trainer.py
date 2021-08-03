import torch

import logger

from abc import ABC, abstractmethod
from torch.utils.data import DataLoader

from torchsampler import ImbalancedDatasetSampler

from dataset import EEGDataset


class ModelTrainer(ABC):

    @staticmethod
    def setup_device():
        """Return torch cuda device if available, else fallback to CPU. TPU untested."""
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def __init__(self,
                 data_dir: str,
                 cv_file: str,
                 task: str,
                 augmentation="original",
                 batch_size=128,
                 learning_rate=0.001,
                 log_weights=False,
                 num_epochs=20,
                 num_workers=2,
                 optimizer="Adam",
                 is_knn=False,
                 sampling: str = None,
                 ):

        self.augmentation = augmentation
        self.cv_file = cv_file
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.learning_rate = learning_rate
        self.log_weights = log_weights
        self.num_epochs = num_epochs
        self.num_workers = num_workers
        self.sampling = sampling,
        self.optimizer = optimizer
        self.is_knn = is_knn

        if task in ["seizure", "patient"]:
            self.task = task
        else:
            raise NameError("Invalid task")

        # setup logger
        config = locals().copy()
        del config['data_dir']
        del config['log_weights']
        self.logger = logger.TrainingLogger(self, log_weights, config)

        # setup datasets
        datasets, dataloaders = self.setup_dataloaders()
        self.datasets = datasets
        self.dataloaders = dataloaders
        self.device = self.setup_device()

    def collate_fn_padd(self, batch):
        labels = [t[1] for t in batch]
        lengths = torch.tensor([t[0].shape[0] for t in batch]).to(self.device)
        max_len = 3141120
        a = [torch.Tensor(t[0].float()).to(self.device) for t in batch]
        new_batch = [torch.nn.ConstantPad1d((0, max_len-lengths[i]), 0)(d).numpy() for i,d in enumerate(a)]
        return new_batch, labels

    def setup_sampler(self, data):
        """Returns sample for the chosen sampling operation, or None if no sampling is desired"""
        class_distribution = data.class_distribution()
        num_classes = data.num_classes()

        if self.sampling == "undersampling":
            minority_class_key = min(class_distribution, key=class_distribution.get)
            minority_class_samples = class_distribution()[minority_class_key]

            return ImbalancedDatasetSampler(data, num_samples=minority_class_samples * num_classes)

        elif self.sampling == "oversampling":
            majority_class_key = max(class_distribution, key=class_distribution.get)
            majority_class_samples = class_distribution()[majority_class_key]

            return ImbalancedDatasetSampler(data, num_samples=majority_class_samples * num_classes)

        elif self.sampling == "resampling":
            return ImbalancedDatasetSampler(data)

        else:
            return None

    def setup_dataloaders(self):
        """Returns atuple of (datasets, dataloaders) for the chosen split and task"""
        datasets = {}
        dataloaders = {}

        for split in ["train", "val"]:
            if split == "train":
                augmentation = self.augmentation
            else:
                augmentation = "original"
            datasets[split] = EEGDataset(
                self.data_dir, task=self.task, cv_file=self.cv_file, split=split, augmentation=augmentation,
                logger=self.logger.get())
            if split == "train":
                sampler = self.setup_sampler(datasets[split])
                shuffle = True
            else:
                sampler = None
                shuffle = False
            if self.is_knn:
                dataloaders[split] = DataLoader(
                    datasets[split], batch_size=datasets[split].__len__(), sampler=sampler,
                    shuffle=shuffle, num_workers=self.num_workers)
            else:
                dataloaders[split] = DataLoader(
                    datasets[split], batch_size=self.batch_size, collate_fn=self.collate_fn_padd, sampler=sampler,
                    shuffle=shuffle, num_workers=self.num_workers)

        return datasets, dataloaders

    def setup_model(self, name):
        self.logger.info("Setup %s model: ", name)

    def train_model(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError
