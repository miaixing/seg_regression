import torch
from torch import nn
from torch.nn import Linear, ModuleList, Dropout, LeakyReLU
from torch import Tensor

class CubicClassifier(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dims: list = (64, 128, 64),
            dropout: float = 0.1
    ):
        super(CubicClassifier, self).__init__()
        self.bridge_layer = Linear(
            in_features=input_dim,
            out_features=hidden_dims[0],
            bias=True
        )
        self.linear_layers = ModuleList(
            Linear(in_features=hidden_dims[i], out_features=hidden_dims[i+1], bias=True)
            for i in range(len(hidden_dims)-1)
        )
        self.output_layer = Linear(hidden_dims[-1], output_dim, bias=True)
        self.dropout = Dropout(p=dropout)
        self.ELU = LeakyReLU()

    def forward(self, x):
        x = self.ELU(self.bridge_layer(x))
        for layer in self.linear_layers:
            x = self.ELU(layer(x))
            x = self.dropout(x)

        return self.output_layer(x)

class Regressor(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dims: list = (64, 128, 64),
            dropout: float = 0.1
    ):
        super(Regressor, self).__init__()
        self.bridge_layer = Linear(input_dim, hidden_dims[0], bias=True)
        self.linear_layers = ModuleList(
            Linear(hidden_dims[i], hidden_dims[i+1], bias=True)
            for i in range(len(hidden_dims)-1)
        )
        self.dropout = Dropout(p=dropout)
        self.output_layer = Linear(hidden_dims[-1], 1, bias=True)
        self.ELU = LeakyReLU()

    def forward(self, x):
        x = self.ELU(self.bridge_layer(x))
        for layer in self.linear_layers:
            x = self.ELU(layer(x))
            x = self.dropout(x)

        return self.output_layer(x)

class SegmentedRegression(nn.Module):
    def __init__(
            self,
            input_dim: int,
            classifier_hidden_dims: list = (64, 128, 64),
            regressor_hidden_dims: list = (64, 128, 64),
            dropout: float = 0.1
    ):
        super(SegmentedRegression, self).__init__()
        self.classifier = CubicClassifier(
            input_dim=input_dim,
            hidden_dims=classifier_hidden_dims,
            output_dim=3,
            dropout=dropout
        )
        self.neg_seq_regressor = Regressor(
            input_dim=input_dim,
            hidden_dims=regressor_hidden_dims,
            dropout=dropout
        )
        self.mid_seq_regressor = Regressor(
            input_dim=input_dim,
            hidden_dims=regressor_hidden_dims,
            dropout=dropout
        )
        self.pos_seq_regressor = Regressor(
            input_dim=input_dim,
            hidden_dims=regressor_hidden_dims,
            dropout=dropout
        )

    def forward(self, x: Tensor, x_category: Tensor):
        # x_category must be one-hot encoded Tensor, not a integer!
        predict_category = self.classifier(x)

        enrichment = torch.cat(
            [self.neg_seq_regressor(x), self.mid_seq_regressor(x), self.pos_seq_regressor(x)],
            dim=1
        )
        print(enrichment.shape)
        score = enrichment*x_category
        print(score.shape)

        return predict_category, score


if __name__ == "__main__":
    model = SegmentedRegression(input_dim=10*20)
    print(model)
    data = torch.randn(size=(32, 200))
    classification = torch.randn(size=(32, 3))
    out = model(data, classification)[1]
    print(out.shape)
