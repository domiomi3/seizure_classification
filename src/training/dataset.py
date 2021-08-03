import re
import os

import numpy as np
import torch

import dill as pickle

from collections import Counter
from torchvision import transforms
from torch.utils.data import Dataset


class EEGDataset(Dataset):
    def __init__(self, data_dir: str, cv_file: str, split="train", task="seizure", augmentation="original", logger=None):
        self.splits = ["train", "val"]
        self.tasks = ["seizure", "patient"]
        self.cv_file = cv_file
        self.data_dir = data_dir

        if task in self.tasks:
            self.task = task
        else:
            raise NameError("Invalid split name")
        if split in self.splits:
            self.split = split
        else:
            raise NameError("Invalid split name")
        self.labels = self._create_labels()

        data_transformations = {
            "original": [],
            "flatten": [transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))]
        }

        if type(augmentation) is list:
            applied_augmentations = []
            for augmentation_method in augmentation:
                if logger is not None:
                    logger.info("Activated augmentation '%s' for %s set.", augmentation_method, split)
                applied_augmentations.extend(data_transformations[augmentation_method])
        else:
            if logger is not None:
                logger.info("Activated augmentation '%s' for %s set.", augmentation, split)
            applied_augmentations = data_transformations[augmentation]

        self.transformations = transforms.Compose([*applied_augmentations])

    def class_distribution(self):
        """Returns dict of (label: number_of_samples)"""
        return dict(Counter(self.labels))

    def class_labels(self):
        return np.unique(self.labels)

    def _create_labels(self):
        split_files = pickle.load(open(os.path.join(self.data_dir, self.cv_file), 'rb'))["1"][self.split]
        labels = []
        for pkl_file in split_files:
            labels.append(pkl_file.split("_")[5].split(".")[0])
        return labels

    def load_pickle(self, file: str):
        for pkl_file in os.listdir(self.data_dir):
            if re.search(file, pkl_file):
                return pickle.load(open(os.path.join(self.data_dir, file), 'rb')).data
        raise NameError("Invalid file name")

    def num_classes(self):
        return len(np.unique(self.labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        # if idx not in range(1, self.__len__()):
        #     raise NameError("Invalid sample index")
        split_files = pickle.load(open(os.path.join(self.data_dir, self.cv_file), 'rb'))["1"][self.split]
        file = split_files[idx-1]
        data = np.asarray(self.load_pickle(file).data)
        label = file.split("_")[5].split(".")[0]
        if self.transformations:
            data = self.transformations(data)
        return data, label
