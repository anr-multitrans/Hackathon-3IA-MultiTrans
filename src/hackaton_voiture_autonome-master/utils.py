import torch

class MSEcustom(torch.nn.Module):
    def __init__(self):
        super(MSEcustom, self).__init__()

    def forward(self, y_true, y_pred):
        # Flatten predictions and ground truths
        result = torch.sum((y_true - y_pred)**2)/y_true.shape[0]
        return result