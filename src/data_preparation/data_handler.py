import torch
import numpy as np

from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from torch.utils.data import DataLoader

from src.data_preparation.dataset import TUHDataset


class DataHandler:
    def __init__(self, data_dir: str,
                 batch_size=32,
                 data_split=0.3,
                 shuffle_dataset=True,
                 weighted_sampler=True,
                 model_name="cnn_dense",
                 dynamic_window=8,
                 bandpass_freqs=[0.1, 47]):
        self.preprocessed_dataset = TUHDataset(data_dir=data_dir,
                                               model_name=model_name,
                                               dynamic_window=dynamic_window,
                                               bandpass_freqs=bandpass_freqs)
        self.dataset_size = len(self.preprocessed_dataset)
        self.batch_size = batch_size
        self.data_split = data_split
        self.shuffle_dataset = shuffle_dataset
        self.weighted_sampler = weighted_sampler

    def get_data_loaders(self):
        indices = list(range(self.dataset_size))
        split = int(np.floor(self.data_split * self.dataset_size))
        if self.shuffle_dataset:
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        # Creating PT data samplers and loaders:
        if self.weighted_sampler:
            train_class_sample_count = torch.tensor([(self.preprocessed_dataset.labels[train_indices] == t).sum()
                                                     for t in torch.unique(self.preprocessed_dataset.labels, sorted=True)])
            weight = 1. / train_class_sample_count.float()
            train_samples_weights = torch.tensor([weight[t] for t in self.preprocessed_dataset.labels[train_indices]])
            train_sampler = WeightedRandomSampler(train_samples_weights, len(train_samples_weights))
            train_dataset = torch.utils.data.TensorDataset(self.preprocessed_dataset.data[train_indices],
                                                           self.preprocessed_dataset.labels[train_indices])
            train_loader = DataLoader(train_dataset,
                                      batch_size=self.batch_size,
                                      sampler=train_sampler,
                                      drop_last=True)

        else:
            train_sampler = SubsetRandomSampler(train_indices)
            train_loader = DataLoader(self.preprocessed_dataset,
                                      batch_size=self.batch_size,
                                      sampler=train_sampler,
                                      drop_last=True)

        val_sampler = SubsetRandomSampler(val_indices)
        val_loader = DataLoader(self.preprocessed_dataset,
                                  batch_size=self.batch_size,
                                  sampler=val_sampler,
                                  drop_last=True)

        return train_loader, val_loader

    def get_input_shape(self):
        return (self.batch_size, self.preprocessed_dataset.dynamic_window) + self.preprocessed_dataset.matrix_shape

    def get_num_classes(self):
        return self.preprocessed_dataset.classes

