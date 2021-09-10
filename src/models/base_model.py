from torch.nn import Module


class BaseModel(Module):
    def __init__(self, batch_size, num_models, num_classes):
        super().__init__()
        self.batch_size = batch_size
        self.num_models = num_models
        self.num_classes = num_classes

    def predict(self, xs):
        pass

    def get_name(self):
        return self.__class__.__name__
