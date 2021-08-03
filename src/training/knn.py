from abc import ABC

from model_trainer import ModelTrainer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


class KNNTrainer(ModelTrainer, ABC):
    def __init__(self, data_dir: str, cv_file: str, task: str,augmentation="original"):
        super().__init__(data_dir, cv_file, task,augmentation)

    def train_model(self):
        self.setup_model("K-Nearest Neighbours")
        model = KNeighborsClassifier(algorithm="kd_tree")
        train_data = [self.datasets["train"].__getitem__(i)[0] for i in range(1, self.datasets["train"].__len__()+1)]
        train_labels = self.datasets["train"].labels

        self.logger.setup_model(model)
        model.fit(train_data, train_labels)
        test_data = [self.datasets["test"].__getitem__(i) for i in range(1, self.datasets["test"].__len__())]
        test_labels = self.datasets["test"].labels
        pred_labels = model.predict(test_data)
        result = confusion_matrix(test_labels, pred_labels)
        print("Confusion matrix: ", result)
        result1 = classification_report(test_labels, pred_labels)
        print("Classification report: ", result1)
        result2 = accuracy_score(test_labels, pred_labels)
        print("Accuracy: ", result2)
