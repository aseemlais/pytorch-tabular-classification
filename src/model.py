import torch.nn as nn


HIDDEN_NEURONS = 10


class RiceClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.input_layer = nn.Linear(10, HIDDEN_NEURONS)
        self.output_layer = nn.Linear(HIDDEN_NEURONS, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.output_layer(x)
        x = self.sigmoid(x)

        return x

    