import torch
import torch.nn as nn


class PredictionHead(nn.Module):
    """
    Universal head for CNN / GNN / LSTM encoders.
    Always uses: projection -> ReLU -> dropout -> output
    """

    def __init__(self,
                 d_in: int,
                 hidden: int = 256,
                 n_out: int = 1,
                 dropout: float = 0.2):
        super().__init__()
        self.d_in = d_in
        self.hidden = hidden
        self.n_out = n_out
        self.dropout = dropout

        self._validate_parameters()

        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_out)
        )

    def _validate_parameters(self) -> None:
        if self.d_in <= 0:
            raise ValueError(f'input dimension must be positive, got {self.d_in}')
        if self.dropout < 0.0 or self.dropout > 1.0:
            raise ValueError(f'dropout must be between 0.0 and 1.0, got {self.dropout}')
        if self.hidden <= 0:
            raise ValueError(f'hidden dimension must be positive, got {self.hidden}')
        if self.n_out <= 0:
            raise ValueError(f'output dimension must be positive, got {self.n_out}')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f'Expected 2D input (batch_size, features), got {x.dim()}D with shape {x.shape}')

        if x.size(1) != self.d_in:
            raise ValueError(f'Input feature dimension {x.size(1)} does not match expected {self.d_in}')

        return self.net(x)
