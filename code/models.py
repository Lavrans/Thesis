from datetime import date

from pandas import DataFrame
from scipy.stats import norm
from sub_modules import ContinuesEmbedding
from sub_modules import GateAndNorm
from sub_modules import GRNNoContext
from sub_modules import InterpretableMultiHeadAttention
from sub_modules import LSTMEncoder
from sub_modules import StaticEnrichment
from sub_modules import StaticVarEncoder
from sub_modules import StaticVarSelection
from torch import Tensor
from torch import cat
from torch import exp
from torch import log
from torch import nn
from torch import sqrt
from torch import stack
from torch import tensor
from torch.jit import ScriptModule
from torch.jit import script_method
from utils import get_mlp_data


class TFT(nn.Module):
    def __init__(
        self,
        dimensions: int,
        no_static_features: int,
        dropout: float,
        num_heads: int = 1,
        lstm_layers: int = 1,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.static_embedding = ContinuesEmbedding(dimensions)
        self.temporal_embedding = ContinuesEmbedding(dimensions)
        self.static_var_selection = StaticVarSelection(no_static_features, dimensions, dropout)
        self.static_enc = StaticVarEncoder(dimensions, dropout)
        self.temporal_encoder = LSTMEncoder(dimensions, dropout, lstm_layers)
        self.static_enrichment = StaticEnrichment(dimensions, dropout)
        self.attention = InterpretableMultiHeadAttention(num_heads, dimensions, dropout)
        self.grn = GRNNoContext(dimensions, dropout)
        self.add_norm_after_attention = GateAndNorm(dimensions)
        self.add_norm_final = GateAndNorm(dimensions)
        self.w_o = nn.Linear(dimensions, 1)
        self.relu = nn.ReLU()

    def forward(self, statics, temporal):
        statics = self.static_embedding(statics)
        temporal = self.temporal_embedding(temporal)

        statics, mask = self.static_var_selection(statics)

        c_s, c_c, c_h, c_e = self.static_enc(statics)
        encoded = self.temporal_encoder(temporal, c_s, c_c, c_h)
        x = self.static_enrichment(encoded, c_e)
        head, attn = self.attention(x, x, x)
        x = self.add_norm_after_attention(head, x)
        x = self.grn(x)
        x = self.add_norm_final(x, encoded)
        x = x[:, -1, :].squeeze()
        x = self.w_o(x)
        if not self.training:
            x = self.relu(x)
        return x, attn, mask


class LstmMLP(ScriptModule):
    def __init__(self, *args, **kwargs) -> None:  # type: ignore
        super().__init__(*args, **kwargs)
        self.lstm = nn.LSTM(input_size=1, hidden_size=8, num_layers=3, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(12, 200),
            nn.BatchNorm1d(200),
            nn.LeakyReLU(0.05),
            nn.Linear(200, 200),
            nn.BatchNorm1d(200),
            nn.LeakyReLU(0.05),
            nn.Linear(200, 200),
            nn.BatchNorm1d(200),
            nn.LeakyReLU(0.05),
            nn.Linear(200, 200),
            nn.BatchNorm1d(200),
            nn.LeakyReLU(0.05),
            nn.Linear(200, 1),
        )
        self.relu = nn.ReLU()

    @script_method
    def forward(self, x: tuple[Tensor, Tensor]) -> Tensor:
        r, s = x
        r = r.unsqueeze(dim=2)
        # r = self.lstm_bn(r.permute(0, 2, 1)).permute(0, 2, 1)
        lstm_out, _ = self.lstm(r)
        mlp_in = cat([lstm_out[:, -1, :], s], dim=1)
        out = self.mlp(mlp_in)
        if not self.training:
            out = self.relu(out)
        return out


class MLP(ScriptModule):
    def __init__(self, num_layers: int, hidden_size: int, leaky_alpha: float, *args, **kwargs) -> None:  # type: ignore
        super().__init__(*args, **kwargs)
        layers = [nn.Linear(5, hidden_size), nn.BatchNorm1d(hidden_size), nn.LeakyReLU(leaky_alpha)]
        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.LeakyReLU(leaky_alpha))
        layers.append(nn.Linear(hidden_size, 1))
        self.mlp = nn.Sequential(*layers)
        self.relu = nn.ReLU()

    @script_method
    def forward(self, x: Tensor) -> Tensor:
        x = self.mlp(x)
        if not self.training:
            x = self.relu(x)
        return x


class BS(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.N = norm.cdf

    def forward(self, x):
        S, K, T, r, sigma = x.T
        r = r / 100
        d1 = (log(S / K) + (r + sigma**2 / 2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        return S * self.N(d1) - K * exp(-r * T) * self.N(d2)


if __name__ == '__main__':
    model = BS()
    model.to('mps')
    s, t = get_mlp_data(date(2020, 1, 1), date(2020, 6, 1))
    s = tensor(s)
    t = tensor(t)
    out = model(s)
    loss = sqrt((t - out) ** 2)
    print(*s.T, out, t, loss)
    df = DataFrame(stack([*s.T, out, t, loss]).T.numpy(), columns=['S', 'K', 'T', 'r', 'IV', 'pred', 'target', 'loss'])
    print(df.describe())
