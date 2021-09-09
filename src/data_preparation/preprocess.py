#!/usr/bin/env python
import warnings
import os
import sys

import numpy as np

from torch import Tensor, LongTensor

from pyeeglab import TUHEEGSeizureDataset, Pipeline, CommonChannelSet, \
    LowestFrequency, ToDataframe, DynamicWindow, BinarizedSpearmanCorrelation, \
    ToNumpy, BandPassFilter, CorrelationToAdjacency, BandPower, GraphWithFeatures, ForkedPreprocessor

warnings.simplefilter(action='ignore')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


class DataPreparator:
    def __init__(self, data_dir: str, model_name="cnn_dense", dynamic_window=8, bandpass_freqs=[0.1, 47]):
        self.model_name = model_name
        self.dynamic_window = dynamic_window
        self.min_bandpass_freq, self.max_bandpass_freq = bandpass_freqs[0], bandpass_freqs[1]
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
                                     BandPassFilter(self.min_bandpass_freq, self.max_bandpass_freq),
                                     ToDataframe(),
                                     DynamicWindow(self.dynamic_window),
                                     BinarizedSpearmanCorrelation(),
                                     ToNumpy()]
        elif self.model_name == "gat_lstm":
            preprocessing_methods = [CommonChannelSet(),
                                     LowestFrequency(),
                                     BandPassFilter(self.min_bandpass_freq, self.max_bandpass_freq),
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
        self.data_dir = data_dir
        self.preprocessing_methods = preprocessing_methods
        self.dataset = TUHEEGSeizureDataset(self.data_dir)

    def preprocess_dataset(self):
        if "gat" in self.model_name:
            preprocessing = Pipeline(
                self.preprocessing_methods, to_numpy=False
            )
        else:
            preprocessing = Pipeline(
                self.preprocessing_methods
            )
        return self.dataset.set_pipeline(preprocessing).load()

    def clean_dataset(self, is_bckg=False, is_combined=True):
        preprocessed_dataset = self.preprocess_dataset()
        data, labels, encoder = preprocessed_dataset["data"], LongTensor(preprocessed_dataset["labels"]), \
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
                    no_del_indices.append(i) # delete bckg and minority classes
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
        self.adjs = data[0].shape[0]
        self.input_shape = data[0].shape[1:]
        self.encoder = encoder
        return data, labels

    def get_data(self):
        data = self.clean_dataset()[0]
        inputs = [[] for _ in range(self.adjs)]
        for d in data:
            for i in range(self.adjs):
                inputs[i].append(d[i].reshape(*self.input_shape))
        data = Tensor([np.array(i) for i in inputs]).permute(1, 0, 2, 3)  # reshape to (batch_size,channels,w_length,
        # w_height) format
        return data

    def get_labels(self):
        return self.clean_dataset()[1]
