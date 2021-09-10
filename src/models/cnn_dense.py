from torch import stack, max
from torch.nn import Linear, Softmax

from .base_model import BaseModel
from .cnn import CNNModel


class CNNDenseModel(BaseModel):
    def __init__(self, input_shape, num_classes):
        super().__init__(batch_size=input_shape[0],
                         num_models=input_shape[1],
                         num_classes=num_classes)
        self.cnns = [CNNModel() for _ in range(self.num_models)]
        self.fcl = Linear(in_features=648 * self.num_models, out_features=self.num_classes)
        self.softmax = Softmax()

    def predict(self, xs):
        cnn_outputs = []
        for i in range(self.num_models):
            cnn_out = self.cnns[i].forward(xs[:, i, :, :].unsqueeze(1))
            cnn_outputs.append(cnn_out.squeeze(0))
        fcl_input = stack(cnn_outputs, dim=1).reshape(self.batch_size, 648 * self.num_models)
        fcl_output = self.fcl(fcl_input)
        soft_pred = self.softmax(fcl_output)
        return max(soft_pred, dim=1)[-1], soft_pred  # predicted class labels, batch_size len


