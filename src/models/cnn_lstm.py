from torch import stack, max
from torch.nn import Linear, Softmax, LSTM

from .base_model import BaseModel
from .cnn import CNNModel


class CNNLSTMModel(BaseModel):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        self.batch_size = input_shape[0]
        self.num_models = input_shape[1]
        self.cnns = [CNNModel() for _ in range(self.num_models)]
        self.lstm = LSTM(648,32)
        self.fcl = Linear(in_features=256, out_features=num_classes)
        self.softmax = Softmax()

    def predict(self, xs):
        cnn_outputs = []
        for i in range(self.num_models):
            cnn_out = self.cnns[i].forward(xs[:, i, :, :].unsqueeze(1))
            cnn_outputs.append(cnn_out.squeeze(0))
        cnn_output = stack(cnn_outputs, dim=1).reshape(self.batch_size, 648 * self.num_models)
        lstm_input = cnn_output.resize(self.batch_size, self.num_models, 648)
        fcl_input = self.lstm(lstm_input)[0].reshape(32, 32*8)
        fcl_output = self.fcl(fcl_input)
        soft_pred = self.softmax(fcl_output)
        return max(soft_pred, dim=1)[-1], soft_pred  # predicted class labels, batch_size len
