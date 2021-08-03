from abc import ABC

import numpy as np

from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_class_weight

from model_trainer import ModelTrainer


class SGDTrainer(ModelTrainer, ABC):
    def __init__(self, data_dir: str, cv_file: str, task: str, augmentation="original"):
        super().__init__(data_dir, cv_file, task, augmentation)

    def adjust_weights(self, weights):
        new_weights = {}
        for label in self.datasets["train"].class_labels():
            new_weights[label] = weights[label]
        return new_weights

    def compute_weights(self):
        class_labels = self.datasets["train"].class_labels()
        cw_array = compute_class_weight('balanced', classes=class_labels, y=self.datasets["train"].labels)
        class_weights = {}
        for idx, label in enumerate(class_labels):
            class_weights[label] = cw_array[idx]
        return class_weights

    def train_model(self):
        self.setup_model("Stochastic Gradient Descent")
        class_weights = self.compute_weights()
        # new_class_weights = self.adjust_weights(class_weights)
        model = SGDClassifier(shuffle=True, loss='log', class_weight=class_weights)
        self.logger.setup_model(model)
        # model.to(self.setup_device())
        for epoch in range(1, self.num_epochs + 1):
            self.logger.info("Epoch %i/%i", epoch, self.num_epochs)
            self.logger.info('-' * 10)
            pred = []
            targets = []
            for i, (data, labels) in enumerate(self.dataloaders["train"]):
                # StandardScaler().fit(data)
                model.partial_fit(data, labels, classes=self.datasets["train"].class_labels())
                if i % 5:
                    for d_, l_ in self.dataloaders['val']:
                        # StandardScaler().fit(d_)
                        pred.append(model.predict(d_))
                        targets.append(l_)
                    result1 = classification_report(targets, pred)
                    print("Classification report: ", result1)
                    result2 = accuracy_score(targets, pred)
                    print("Accuracy: ", result2)

    # def evaluate(self):
    #     pred_labels = model.predict(dev_data)
    #     # result = confusion_matrix(dev_labels, pred_labels)
    #     # print("Confusion matrix: ", result)
    #     result1 = classification_report(dev_labels, pred_labels)
    #     print("Classification report: ", result1)
    #     result2 = accuracy_score(dev_labels, pred_labels)
    #     print("Accuracy: ", result2)
    #     # print(len(dev_labels), pred_labels.shape())
    #     # print(type(mean_squared_error(dev_labels, pred_labels), ((dev_labels - pred_labels)**2).mean(ax=))
    #     # improvements.append(mean_squared_error(dev_labels, pred_labels))
