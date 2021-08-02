from abc import ABC

from model_trainer import ModelTrainer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


class KNNTrainer(ModelTrainer, ABC):
    def __init__(self, data_dir: str, cv_file: str, task: str):
        super().__init__(data_dir, cv_file, task)

    def train_model(self):
        self.setup_model("K-Nearest Neighbours")
        model = KNeighborsClassifier(algorithm="kd_tree")
        train_data, train_labels = self.dataloaders["train"]
        self.logger.setup_model(model)
        model.fit(train_data, train_labels)
        test_data, test_labels = self.dataloaders["test"]
        pred_labels = model.predict(test_data)
        result = confusion_matrix(test_labels, pred_labels)
        print("Confusion matrix: ", result)
        result1 = classification_report(test_labels, pred_labels)
        print("Classification report: ", result1)
        result2 = accuracy_score(test_labels, pred_labels)
        print("Accuracy: ", result2)
