import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=500):
        super().__init__()
        self.d_model = d_model

        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        x = x + Variable(self.pe[:, :seq_len], requires_grad=False).to(x.device)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        output = self.out(concat)

        return output

    def attention(self, q, k, v, d_k, mask=None, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)

        if dropout is not None:
            scores = self.dropout(scores)

        output = torch.matmul(scores, v)
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.norm = Norm(d_model)

    def forward(self, x):
        residual = x
        x = self.linear_2(F.relu(self.linear_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.norm(x)
        return x


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model, d_ff=d_model * 2)
        self.dropout_1 = nn.Dropout(dropout)

    def forward(self, x, mask):
        residual = x
        x = self.dropout_1(self.attn(x, x, x, mask))
        x2 = self.norm_1(residual + x)
        x = self.ff(x2)
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, N, heads):
        super().__init__()
        self.N = N
        self.pe = PositionalEncoder(d_model)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, heads) for _ in range(N)
        ])
        self.norm = Norm(d_model)

    def forward(self, src, mask3d=None):
        x = self.pe(src)
        for i in range(self.N):
            x = self.layers[i](x, mask3d)
        return self.norm(x)


class Attention(nn.Module):  # 定义注意力机制的神经网络模块

    def __init__(self, hid_dim):  # 初始化方法，传入隐藏层维度
        super().__init__()  # 调用父类的初始化方法
        self.hid_dim = hid_dim  # 保存隐藏层维度

        self.attn = nn.Linear(self.hid_dim * 2, self.hid_dim)  # 定义一个线性层，用于计算注意力能量
        self.v = nn.Linear(self.hid_dim, 1, bias=False)  # 定义一个线性层，将能量映射为一个分数，不使用偏置

    def forward(self, query, key, value, attn_mask):  # 前向传播方法，输入query、key、value和注意力mask
        # query = [batch size, src len, hid dim]
        # key = [batch size, src len, candi num, hid dim]
        bs, src_len = query.shape[0], query.shape[1]  # 获取batch size和源序列长度
        candi_num = key.shape[-2]  # 获取候选数量
        # repeat decoder hidden state src_len times
        query = query.unsqueeze(-2).repeat(1, 1, candi_num, 1)  # 扩展query维度并在candi_num维度上重复

        energy = torch.tanh(self.attn(torch.cat((query, key), dim=-1)))  # 拼接query和key，经过线性层和tanh激活得到能量

        attention = self.v(energy).squeeze(-1)  # 将能量通过线性层映射为注意力分数，并去掉最后一维
        attention = attention.masked_fill(attn_mask == 0, -1e10)  # 使用mask将padding位置的分数设为极小值
        # using mask to force the attention to only be over non-padding elements.
        scores = F.softmax(attention, dim=-1)  # 对注意力分数在最后一维做softmax归一化
        weighted = torch.bmm(scores.reshape(bs*src_len, candi_num).unsqueeze(-2), value.reshape(bs*src_len, candi_num, -1)).squeeze(-2)  # 计算加权和
        weighted = weighted.reshape(bs, src_len, -1)  # 恢复加权结果的形状

        return scores, weighted  # 返回注意力分数和加权结果


class GPSLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()

        self.norm_1 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model, d_ff=d_model * 2)
        self.dropout_1 = nn.Dropout(dropout)

    def forward(self, x, mask):
        residual = x
        x = self.dropout_1(self.attn(x, x, x, mask))
        x2 = self.norm_1(residual + x)
        x = self.ff(x2)
        return x


class GPSFormer(nn.Module):
    def __init__(self, d_model, N, heads):
        super().__init__()
        self.N = N
        self.layers = nn.ModuleList([
            GPSLayer(d_model, heads) for _ in range(N)
        ])

    def forward(self, src, mask3d=None):
        x = src
        for i in range(self.N):
            x = self.layers[i](x, mask3d)
        return x


class RouteLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()

        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.slf_attn = MultiHeadAttention(heads, d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model, d_ff=d_model * 2)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, route, gps, route_mask, inter_mask):
        route1 = self.dropout_1(self.slf_attn(route, route, route, route_mask))
        route_out = self.norm_1(route + route1)

        route2 = self.dropout_2(self.attn(route_out, gps, gps, inter_mask))
        route_out2 = self.norm_2(route_out + route2)

        x = self.ff(route_out2)
        return x


class GRLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.gps_enc = GPSLayer(d_model, heads, dropout)
        self.route_enc = RouteLayer(d_model, heads, dropout)

    def forward(self, route, route_mask, gps, gps_mask, inter_mask):
        gps_emb = self.gps_enc(gps, gps_mask)
        route_emb = self.route_enc(route, gps_emb, route_mask, inter_mask)
        return route_emb, gps_emb


class GRFormer(nn.Module):
    def __init__(self, d_model, N, heads):
        super().__init__()
        self.N = N
        self.layers = nn.ModuleList([
            GRLayer(d_model, heads) for _ in range(N)
        ])

    def forward(self, src, route, mask3d, route_mask3d, inter_mask):
        x = route
        y = src
        for i in range(self.N):
            x, y = self.layers[i](x, route_mask3d, y, mask3d, inter_mask)
        return x


def sequence_mask(X, valid_len, value=0.):
    """Mask irrelevant entries in sequences."""

    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


def sequence_mask3d(X, valid_len, valid_len2, value=0.):
    """Mask irrelevant entries in sequences."""

    maxlen = X.size(1)
    maxlen2 = X.size(2)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    mask2 = torch.arange((maxlen2), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len2[:, None]
    mask_fin = torch.bmm(mask.float().unsqueeze(-1), mask2.float().unsqueeze(-2)).bool()
    X[~mask_fin] = value
    return X
