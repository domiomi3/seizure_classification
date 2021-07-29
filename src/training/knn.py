from train import ModelTrainer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


class KNNTrainer(ModelTrainer):
    def __init__(self, data_dir: str, task: str, cv_files: dict):
        super().__init__(data_dir, task, cv_files)

    @staticmethod
    def image_to_feature_vector(image):
        return image.flatten()

    def train(self):
        train_data, train_labels = self.datasets["train"].create_split_dataset(transformation="KNN")
        model = KNeighborsClassifier(algorithm="kd_tree")
        model.fit(train_data, train_labels)
        return model

    def evaluate(self, model):
        dev_data, dev_labels = self.datasets["val"].create_split_dataset(transformation=None)
        pred_labels = model.predict(dev_data)
        result = confusion_matrix(dev_labels, pred_labels)
        print("Confusion matrix: ", result)
        result1 = classification_report(dev_labels, pred_labels)
        print("Classification report: ", result1)
        result2 = accuracy_score(dev_labels, pred_labels)
        print("Accuracy: ", result2)
