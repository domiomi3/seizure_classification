import numpy as np
from sklearn.metrics import classification_report, mean_squared_error, accuracy_score

from sklearn.linear_model import SGDClassifier
from sklearn.utils import compute_class_weight

from train import ModelTrainer


class SGDTrainer(ModelTrainer):
    def __init__(self, task: str):
        super().__init__(task)

    def compute_weights(self, train_labels):
        cw_array = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
        class_weights = {}
        for idx, label in enumerate(np.unique(train_labels)):
            class_weights[label] = cw_array[idx]
        return class_weights

    def train(self, train_data, train_labels, weights):
        sgd_model = SGDClassifier(shuffle=True, loss='log', class_weight=weights)
        sgd_model.partial_fit(train_data, train_labels, classes=np.unique(train_labels))
        return sgd_model

    def evaluate(self, model, dev_data, dev_labels):
        pred_labels = model.predict(dev_data)
        # result = confusion_matrix(dev_labels, pred_labels)
        # print("Confusion matrix: ", result)
        result1 = classification_report(dev_labels, pred_labels)
        print("Classification report: ", result1)
        result2 = accuracy_score(dev_labels, pred_labels)
        print("Accuracy: ", result2)
        # print(len(dev_labels), pred_labels.shape())
        # print(type(mean_squared_error(dev_labels, pred_labels), ((dev_labels - pred_labels)**2).mean(ax=))
        # improvements.append(mean_squared_error(dev_labels, pred_labels))

