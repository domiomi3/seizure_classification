import collections
import random
import re
import os

import dill as pickle
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class EEGDataset(Dataset):
    def __init__(self, data_dir: str,  cv_files: dict, shuffle=True, split="train", task="seizure"):
        self.splits = ["train", "val"]
        self.cv_files = {"patient": cv_files["patient"],
                         "seizure": cv_files["seizure"]}
        self.data_dir = data_dir
        self.seizure_type_data = collections.namedtuple('seizure_type_data', ['patient_id', 'seizure_type', 'data'])
        if task in self.cv_files.keys():
            self.cv_file = self.cv_files[task]
        else:
            raise NameError("Invalid task name")
        if split in self.splits:
            self.split = split
        else:
            raise NameError("Invalid split name")
        self.indices = self._create_sampling_windows_indices(shuffle=shuffle)

    def _create_sampling_windows_indices(self, shuffle=True):
        split_files = pickle.load(open(self.cv_file, 'rb'))["1"][self.split]
        sw_array = []
        for pkl_file in split_files:
            sw_no = self.get_sampling_windows_number(pkl_file)
            for sw in range(0, sw_no-1):
                sw_array.append(str(pkl_file + f"_index_{sw}"))
        if shuffle:
            random.shuffle(sw_array)
        return sw_array

    def create_split_dataset(self, batch_size=None, idx=None, transformation=None):
        if batch_size:
            batch = self.get_indices(self.generate_random_indices(size=batch_size, idx=idx))
        else:
            batch = self.get_indices(self.indices)
        data = []
        labels = []
        for index in batch:
            if transformation == "KNN" or transformation == "SGD":
                data.append((self.__getitem__(index)[0]).flatten())
            labels.append(self.__getitem__(index)[1])
        if transformation == "SGD":
            StandardScaler().fit(data)
        return data, labels

    def generate_random_indices(self, size: int, idx: int):
        return self.indices[(idx*size):(idx*size+size)]

    @staticmethod
    def get_index(file: str):
        return int(file.split("_index_")[1])

    def get_indices(self, file_list: list):
        return [self.get_index(file) for file in file_list]

    def get_sampling_windows_number(self, file: str):
        data = self.load_pickle(file)
        return data.shape[0]

    def get_max_sampling_windows_number(self):
        max_sw_no = 0
        for pkl_file in os.listdir(self.data_dir):
            sw_no = self.get_sampling_windows_number(pkl_file)
            max_sw_no = sw_no if sw_no > max_sw_no else max_sw_no
        return max_sw_no

    def load_pickle(self, file: str):
        for pkl_file in os.listdir(self.data_dir):
            if re.search(file, pkl_file):
                return pickle.load(open(os.path.join(self.data_dir, file), 'rb')).data
        raise NameError("Invalid file name")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        if idx in range(0, self.__len__()):
            file, index = self.indices[idx].split("_index_")
            data = self.load_pickle(file)
            sampling_window = data[int(index)]
            label = file.split("_")[5].split(".")[0]

            return sampling_window, label
        else:
            raise NameError("Invalid sample index")
