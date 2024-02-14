import math

import torch
from torch import Tensor
from torch import nn


class ContinuesEmbedding(nn.Module):
    def __init__(self, d: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear = nn.Linear(1, d)

    def forward(self, x):
        return self.linear(x.unsqueeze(2))


class GateAndNorm(nn.Module):
    def __init__(self, d: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gate = nn.Sequential(nn.Linear(d, d), nn.Sigmoid())
        self.linear = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x, residual):
        return self.norm(self.gate(x) * self.linear(x) + residual)


class GRN(nn.Module):
    def __init__(self, d: int, dropout: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context = nn.Linear(d, d, bias=False)
        self.input = nn.Linear(d, d)
        self.elu = nn.ELU()
        self.linear = nn.Sequential(nn.Linear(d, d), nn.Dropout(dropout))
        self.gnorm = GateAndNorm(d)

    def forward(self, x, c):
        return self.gnorm(self.linear(self.elu(self.input(x) + self.context(c))), x)


class GRNNoContext(nn.Module):
    def __init__(self, d: int, dropout: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = nn.Sequential(nn.Linear(d, d), nn.ELU(), nn.Linear(d, d), nn.Dropout(dropout))
        self.gnorm = GateAndNorm(d)

    def forward(self, x):
        return self.gnorm(self.net(x), x)


class StaticGRN(nn.Module):
    def __init__(self, d: int, dropout: float, in_features: int, out_features: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = nn.Sequential(nn.Linear(in_features, d), nn.ELU(), nn.Linear(d, out_features), nn.Dropout(dropout))
        self.linear = nn.Linear(in_features, out_features)
        self.gnorm = GateAndNorm(out_features)

    def forward(self, x):
        return self.gnorm(self.net(x), self.linear(x))


class StaticVarSelection(nn.Module):
    def __init__(self, no_features: int, d: int, dropout: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        in_features = no_features * d
        self.no_features = no_features
        self.grn_mask = StaticGRN(d, dropout, in_features, no_features)
        self.grns = nn.ModuleList([GRNNoContext(d, dropout) for _ in range(no_features)])
        self.softmax = nn.Softmax(-1)

    def forward(self, x: Tensor):
        flatten = x.flatten(1)
        mask = self.softmax(self.grn_mask(flatten).unsqueeze(1))
        splits = x.split(1, 1)
        vecs = [self.grns[i](splits[i]) for i in range(self.no_features)]
        vecs = torch.concat(vecs, 1)
        return torch.matmul(mask, vecs).squeeze(1), mask


class StaticVarEncoder(nn.Module):
    def __init__(self, d: int, dropout: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grn_c = GRNNoContext(d, dropout)
        self.grn_h = GRNNoContext(d, dropout)
        self.grn_e = GRNNoContext(d, dropout)
        self.grn_s = GRNNoContext(d, dropout)

    def forward(self, x: Tensor):
        return self.grn_s(x), self.grn_c(x), self.grn_h(x), self.grn_e(x)


class LSTMEncoder(nn.Module):
    def __init__(self, d: int, dropout: float, num_layers: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_layers = num_layers
        self.grn = GRN(d, dropout)
        self.lstm = nn.LSTM(d, d, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.gnorm = GateAndNorm(d)

    def forward(self, x, c_s, c_c, c_h):
        c_h = c_h.repeat((self.num_layers, 1, 1))
        c_c = c_c.repeat((self.num_layers, 1, 1))
        context = c_s.unsqueeze(1).expand(x.shape)
        residual = self.grn(x, context)
        encoded_vecs, _ = self.lstm(residual, (c_h, c_c))
        encoded_vecs = self.dropout(encoded_vecs)
        return self.gnorm(encoded_vecs, residual)


class StaticEnrichment(nn.Module):
    def __init__(self, d: int, dropout: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grn = GRN(d, dropout)

    def forward(self, x, c_e):
        context = c_e.unsqueeze(1).expand(x.shape)
        return self.grn(x, context)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, q, k, v):
        _, _, d = q.shape
        q_scaled = q / math.sqrt(d)
        attn = torch.bmm(q_scaled, k.transpose(-2, -1))
        attn = nn.functional.softmax(attn, dim=-1)
        output = torch.bmm(attn, v)
        return output, attn


class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, n_head: int, d: int, dropout: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_head = n_head
        d_k = d // n_head
        self.q_layers = nn.ModuleList([nn.Linear(d, d_k, bias=False) for _ in range(n_head)])
        self.k_layers = nn.ModuleList([nn.Linear(d, d_k, bias=False) for _ in range(n_head)])
        self.v_layer = nn.Linear(d, d_k, bias=False)
        self.attention = ScaledDotProductAttention()
        self.w_o = nn.Linear(d_k, d, bias=False)

        self.dropout = nn.Dropout(dropout)
        nn.MultiheadAttention

    def forward(self, q, k, v):
        heads = []
        attention = []

        vs = self.v_layer(v)

        for i in range(self.n_head):
            qs = self.q_layers[i](q)
            ks = self.k_layers[i](k)
            head, attn = self.attention(qs, ks, vs)

            head = self.dropout(head)
            heads.append(head)
            attention.append(attn)

        head = torch.stack(heads).mean(0)
        attn = torch.stack(attention)

        outputs = self.w_o(head)
        outputs = self.dropout(outputs)

        return outputs, attn


if __name__ == '__main__':
    embedding_s = ContinuesEmbedding(8).to('mps')
    embedding_t = ContinuesEmbedding(8).to('mps')
    svs = StaticVarSelection(4, 8, 0.1).to('mps')
    sve = StaticVarEncoder(8, 0.1).to('mps')
    t_enc = LSTMEncoder(8, 0.1, 1).to('mps')
    se = StaticEnrichment(8, 0.1).to('mps')
    imha = InterpretableMultiHeadAttention(1, 8, 0.1).to('mps')

    sc = torch.randn((4096, 4)).to('mps')
    t = torch.randn((4096, 140)).to('mps')

    sc = embedding_s(sc)
    sc, mask = svs(sc)
    c_s, c_c, c_h, c_e = sve(sc)
    t = embedding_t(t)
    o = t_enc(t, c_s, c_c, c_h)
    o = se(o, c_e)
    o, a = imha(o, o, o)
    print(mask)
