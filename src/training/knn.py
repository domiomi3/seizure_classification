from train import ModelTrainer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


class KNNTrainer(ModelTrainer):
    def __init__(self, task: str):
        super().__init__(task)

    def train(self, train_data, train_labels):
        model = KNeighborsClassifier(algorithm="kd_tree")
        model.fit(train_data, train_labels)
        return model

    def evaluate(self, model, dev_data, dev_labels):
        pred_labels = model.predict(dev_data)
        result = confusion_matrix(dev_labels, pred_labels)
        print("Confusion matrix: ", result)
        result1 = classification_report(dev_labels, pred_labels)
        print("Classification report: ", result1)
        result2 = accuracy_score(dev_labels, pred_labels)
        print("Accuracy: ", result2)
