import collections
import re
import os

import dill as pickle
from torch.utils.data import Dataset


class EEGDataset(Dataset):
    def __init__(self, data_dir: str, split="train"):
        self.splits = ["train", "dev"]
        self.data_dir = data_dir
        self.seizure_type_data = collections.namedtuple('seizure_type_data', ['patient_id', 'seizure_type', 'data'])

        if split in self.splits:
            self.split = split
        else:
            raise NameError("Invalid split name")

    def __len__(self):
        return len(os.listdir(self.data_dir))

    def __getitem__(self, file: str):
        for pkl_file in os.listdir(self.data_dir):
            if re.search(file, pkl_file):
                type_data = pickle.load(open(os.path.join(self.data_dir, file), 'rb'))
                time_series_data = type_data.data
                label = file.split("_")[5].split(".")[0]
                return time_series_data, label

        raise NameError("Invalid sample index")
