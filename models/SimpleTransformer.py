import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding
"<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<Transformer>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
import math
from torch.nn.modules import Transformer, TransformerEncoder, TransformerEncoderLayer, LayerNorm

def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """ Generates a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# def greedy_decode(
#         model,
#         src: torch.Tensor,
#         max_len: int,
#         real_target: torch.Tensor,
#         unsqueeze_dim=1,
#         output_len=1,
#         device='cpu',
#         multi_targets=1,
#         probabilistic=False,
#         scaler=None):
#     """
#     Mechanism to sequentially decode the model
#     :src The Historical time series values
#     :real_target The real values (they should be masked), however if you want can include known real values.
#     :returns torch.Tensor
#     """
#     src = src.float()
#     real_target = real_target.float()
#     if hasattr(model, "mask"):
#         src_mask = model.mask #[1,30,3]
#     memory = model.encode_sequence(src, src_mask)  #[30,1,128]
#     # Get last element of src array to forecast from
#     ys = src[:, -1, :].unsqueeze(unsqueeze_dim)  #[1,1,3]
#     for i in range(max_len):
#         mask = generate_square_subsequent_mask(i + 1).to(device) #i=2 [3,3]
#         with torch.no_grad():
#             out = model.decode_seq(memory,
#                                    Variable(ys),
#                                    Variable(mask), i + 1)
#             real_target[:, i, 0] = out[:, i]
#             src = torch.cat((src, real_target[:, i, :].unsqueeze(1)), 1)  #[1,31,3]
#             ys = torch.cat((ys, real_target[:, i, :].unsqueeze(1)), 1) #[1,2,3]
#         memory = model.encode_sequence(src[:, i + 1:, :], src_mask)
#     return ys[:, 1:, :]
class SimplePositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(SimplePositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Creates a basic positional encoding"""
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Model(torch.nn.Module):
    def __init__(
            self,args):
        """A full transformer model

        :param number_time_series: The total number of time series present
            (e.g. n_feature_time_series + n_targets)
        :type number_time_series: int
        :param seq_length: The length of your input sequence, defaults to 48
        :type seq_length: int, optional
        :param output_seq_len: The length of your output sequence, defaults
            to None
        :type output_seq_len: int, optional
        :param d_model: The dimensions of your model, defaults to 128
        :type d_model: int, optional
        :param n_heads: The number of heads in each encoder/decoder block,
            defaults to 8
        :type n_heads: int, optional
        :param dropout: The fraction of dropout you wish to apply during
            training, defaults to 0.1 (currently not functional)
        :type dropout: float, optional
        :param forward_dim: Currently not functional, defaults to 2048
        :type forward_dim: int, optional
        :param sigmoid: Whether to apply a sigmoid activation to the final
            layer (useful for binary classification), defaults to False
        :type sigmoid: bool, optional
        """
        super().__init__()
        if args.output_seq_len is None:
            output_seq_len = args.seq_length
        self.out_seq_len = args.output_seq_len
        self.mask = generate_square_subsequent_mask(args.seq_length)
        self.dense_shape = torch.nn.Linear(args.number_time_series, args.d_model)
        self.pe = SimplePositionalEncoding(args.d_model)
        self.transformer = Transformer(args.d_model, nhead=args.n_heads)
        self.final_layer = torch.nn.Linear(args.d_model, 1)
        self.sequence_size = args.seq_length
        self.tgt_mask = generate_square_subsequent_mask(args.output_seq_len)
        self.sigmoid = None
        if args.sigmoid:
            self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor, t: torch.Tensor, tgt_mask=None, src_mask=None):
        x = self.encode_sequence(x[:, :-1, :], src_mask)
        return self.decode_seq(x, t, tgt_mask)

    def basic_feature(self, x: torch.Tensor):
        x = self.dense_shape(x)  #[1,30,128]
        x = self.pe(x)#[1,30,128]
        x = x.permute(1, 0, 2) # [30,1,128]
        return x

    def encode_sequence(self, x, src_mask=None):
        x = self.basic_feature(x)  #[30,1,128]
        x=x.cuda()
        x = self.transformer.encoder(x, src_mask.cuda())#[30,1,128]
        return x

    def decode_seq(self, mem, t, tgt_mask=None, view_number=None) -> torch.Tensor:
        if view_number is None:
            view_number = self.out_seq_len
        if tgt_mask is None:
            tgt_mask = self.tgt_mask  #generate_square_subsequent_mask(output_seq_len)
        t = self.basic_feature(t)
        x = self.transformer.decoder(t.cuda(), mem.cuda(), tgt_mask=tgt_mask.cuda())
        x = self.final_layer(x)
        if self.sigmoid:
            x = self.sigmoid(x)
        return x.view(-1, view_number)