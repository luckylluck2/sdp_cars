import torch.nn as nn

class DNN(nn.Module):

    def __init__(self, hidden_layers=3, hidden_dim=32, input_dim=3, output_dim=2):
        super(DNN, self).__init__()
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.input_layer = nn.Linear(self.input_dim, self.hidden_dim)
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)
        
        model_layers = []
        model_layers.append(self.input_layer)
        model_layers.append(nn.ReLU())
        for i in range(self.hidden_layers):
            model_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            model_layers.append(nn.ReLU())
        model_layers.append(nn.Dropout(p=0.5))
        model_layers.append(self.output_layer)
        
        self.dnn = nn.Sequential(*model_layers)

    def forward(self, x):
        output = self.dnn(x)
        return output
