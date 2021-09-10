import numpy as np

from torch import Tensor, LongTensor
from torch.utils.data import Dataset

from pyeeglab import TUHEEGSeizureDataset, Pipeline, CommonChannelSet, \
    LowestFrequency, ToDataframe, DynamicWindow, BinarizedSpearmanCorrelation, \
    ToNumpy, BandPassFilter, CorrelationToAdjacency, BandPower, GraphWithFeatures, ForkedPreprocessor


class TUHDataset(Dataset):
    def __init__(self,
                 data_dir=None,
                 model_name=None,
                 dynamic_window=None,
                 bandpass_freqs=[None, None]):
        super().__init__()
        self.data_dir = data_dir
        self.dataset = TUHEEGSeizureDataset(self.data_dir)
        self.classes = None

        self.matrix_shape = None
        self.dynamic_window = dynamic_window
        self.model_name = model_name
        if self.model_name in ["cnn_dense", "cnn_lstm"]:
            preprocessing_methods = [CommonChannelSet(),
                                     LowestFrequency(),
                                     ToDataframe(),
                                     DynamicWindow(self.dynamic_window),
                                     BinarizedSpearmanCorrelation(),
                                     ToNumpy()]
        elif self.model_name in ["cnn_dense_filter", "cnn_lstm_filter"]:
            preprocessing_methods = [CommonChannelSet(),
                                     LowestFrequency(),
                                     BandPassFilter(bandpass_freqs[0], bandpass_freqs[1]),
                                     ToDataframe(),
                                     DynamicWindow(self.dynamic_window),
                                     BinarizedSpearmanCorrelation(),
                                     ToNumpy()]
        elif self.model_name == "gat_lstm":
            preprocessing_methods = [CommonChannelSet(),
                                     LowestFrequency(),
                                     BandPassFilter(bandpass_freqs[0], bandpass_freqs[1]),
                                     ToDataframe(),
                                     DynamicWindow(self.dynamic_window),
                                     ForkedPreprocessor(
                                         inputs=[
                                             [BinarizedSpearmanCorrelation(), CorrelationToAdjacency()],
                                             BandPower()
                                         ],
                                         output=GraphWithFeatures()
                                     )]
        else:
            raise NameError("Incorrect model name - can't preprocess data!")

        to_numpy = False if "gat" in self.model_name else True
        self.data, self.labels = self.preprocess_dataset(preprocessing_methods, to_numpy)

    def preprocess_dataset(self, preprocessing_methods, to_numpy=True, is_bckg=False, is_combined=True):
        preprocessing = Pipeline(
            preprocessing_methods  # , to_numpy=to_numpy
        )
        preprocessed_dataset = self.dataset.set_pipeline(preprocessing).load()

        data, labels, encoder = preprocessed_dataset["data"], \
                                LongTensor(preprocessed_dataset["labels"]), \
                                preprocessed_dataset["labels_encoder"]

        bckg_idx = encoder.index("bckg")
        cpsz_idx = encoder.index("cpsz")
        fnsz_idx = encoder.index("fnsz")
        spsz_idx = encoder.index("spsz")
        tnsz_idx = encoder.index("tnsz")
        tcsz_idx = encoder.index("tcsz")
        absz_idx = encoder.index("absz")
        mysz_idx = encoder.index("mysz")
        gnsz_idx = encoder.index("gnsz")

        if is_combined and not is_bckg:
            encoder = ["cf", "gn", "ct"]
            new_labels = [0] * len(labels)
            no_del_indices = []
            for i, j in enumerate(labels):
                if j == bckg_idx or j == absz_idx or j == mysz_idx:
                    new_labels[i] = 3
                else:
                    no_del_indices.append(i)  # delete bckg and minority classes
                    if j == fnsz_idx or j == cpsz_idx or j == spsz_idx:
                        new_labels[i] = 0  # CF combined focal seizures
                    elif j == gnsz_idx:
                        new_labels[i] = 1  # GN generalized seizures
                    elif j == tnsz_idx or j == tcsz_idx:
                        new_labels[i] = 2  # CT combined tonic seizures
            data = data[no_del_indices, :, :, :]
            labels = Tensor(new_labels).long()[no_del_indices]

        else:
            raise NameError("Incorrect flags!")

        self.classes = len(np.unique(labels))
        self.matrix_shape = data[0].shape[1:]
        inputs = [[] for _ in range(self.dynamic_window)]
        for d in data:
            for i in range(self.dynamic_window):
                inputs[i].append(d[i].reshape(*self.matrix_shape))
        data = Tensor([np.array(i) for i in inputs]).permute(1, 0, 2, 3)  # reshape to (batch_size,channels,w_length,
        # w_height) format

        return data, labels

    def __getitem__(self, index):
        return self.data[index, :, :, :], self.labels[index]

    def __len__(self):
        return len(self.labels)
