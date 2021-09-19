from torch import nn
from torch.nn import functional as F

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super(Model, self).__init__()

        self.dense1 = nn.Linear(input_size, hidden_size, bias=True)
        self.dropout1 = nn.Dropout(0.3)
        self.dense2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.dropout2 = nn.Dropout(0.3)
        self.dense3 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.dropout3 = nn.Dropout(0.3)
        self.out = nn.Linear(hidden_size, output_size, bias=True)

    def forward(self, x):
        x = self.dropout1(F.relu(self.dense1(x)))
        x = self.dropout2(F.relu(self.dense2(x)))
        x = self.dropout3(F.relu(self.dense3(x)))
        x = self.out(x)
        return x