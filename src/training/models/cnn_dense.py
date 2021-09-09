from torch import stack, max
from torch.nn import Conv2d, MaxPool2d, Module, Flatten, Linear, Softmax, Sequential


class CNNDenseModel(Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        self.batch_size = input_shape[0]
        self.num_models = input_shape[1]
        self.cnns = [CNNModel() for _ in range(self.num_models)]
        self.fcl = Linear(in_features=648 * self.num_models, out_features=num_classes) #todo
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


class CNNModel(Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        conv_layers = [Conv2d(in_channels=1,
                              out_channels=8,
                              kernel_size=3),
                       MaxPool2d(kernel_size=2),
                       Flatten()]

        self.conv_layers = Sequential(*conv_layers)

    def forward(self, x):
        return self.conv_layers(x)
