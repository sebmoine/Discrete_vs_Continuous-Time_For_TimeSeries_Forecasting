import torch 
import torch.nn as nn

# For Multivariate :
#       - Multiple Input Series     : [[X_1, X_2]k, [X_1, X_2]k-1, ..., [X_1, X_2]1] [y_1]0  
#                                   x_input = array([[80, 85], [90, 95], [100, 105]])
#                                   x_input = x_input.reshape((1, n_steps, n_features))
#       - Multiple Parallel Series  : [[X_1, X_2]k, [X_1, X_2]k-1, ..., [X_1, X_2]1] [y_1, y_2]0
#                                   x_input = array([[70,75,145], [80,85,165], [90,95,185]])
#                                   x_input = x_input.reshape((1, n_steps, n_features))

from src.utils.log_checkpoint import *

class VanillaLSTM(nn.Module):
    """
        VanillaLSTM : LSTM layer + 1 Dense Layer
    """
    def __init__(self, input_size, hidden_size, num_stacked_layer, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layer = num_stacked_layer

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layer, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layer, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_stacked_layer, batch_size, self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0,c0))
        out = self.fc(out[:, -1, :])
        return out


class StackedLSTM(nn.Module):
    """
        StackedLSTM : Several LSTM Layers one on top of another
        LSTM Layer take a 3D input and give back a 2D output, so need an adjustement for k previous LSTM Layers to produce a 3D output
    """
    def __init__(self):
        super(StackedLSTM).__init__()


class BidirectionalLSTM(nn.Module):
    """
        BidirectionalLSTM : Predict forward and backward, concatenate both prédictions at time t
    """
    def __init__(self):
        super(BidirectionalLSTM).__init__()

