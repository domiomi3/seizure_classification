from torch.nn import Conv2d, MaxPool2d, Module, Flatten, Sequential


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
