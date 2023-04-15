import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding



"<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<Informer>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
class Informer(nn.Module):
    def __init__(self, args):
                #  enc_in, dec_in, c_out, seq_len, label_len, out_len,
                # factor=5, d_model=512, n_heads=8,q_len=3, e_layers=3, d_layers=2, d_ff=512,
                # dropout=0.0,attn='prob', embed='fixed', freq='h', activation='gelu',
                # output_attention = False, distil=True, mix=True,local_casual=False,
                # device=torch.device('cuda:0')):
        super(Informer, self).__init__()
        self.pred_len = args.pred_len
        self.attn = args.attn
        self.output_attention = args.output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(args.enc_in, args.d_model, args.embed, args.freq, args.dropout)
        self.dec_embedding = DataEmbedding(args.dec_in, args.d_model, args.embed, args.freq, args.dropout)
        # Attention
        Attn = ProbAttention if args.attn=='prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, args.factor, attention_dropout=args.dropout, output_attention=args.output_attention),
                                args.d_model, args.n_heads,args.q_len, mix=False,local_casual=args.local_casual),
                    args.d_model,
                    args.d_ff,
                    dropout=args.dropout,
                    activation=args.activation
                ) for l in range(args.e_layers)
            ],
            [
                ConvLayer(
                    args.d_model
                ) for l in range(args.e_layers-1)
            ] if args.distil else None,
            norm_layer=torch.nn.LayerNorm(args.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, args.factor, attention_dropout=args.dropout, output_attention=False),
                                args.d_model, args.n_heads,args.q_len,mix=args.mix,local_casual=args.local_casual),
                    AttentionLayer(FullAttention(False, args.factor, attention_dropout=args.dropout, output_attention=False),
                                args.d_model, args.n_heads, args.q_len,mix=False),
                    args.d_model,
                    args.d_ff,
                    dropout=args.dropout,
                    activation=args.activation,
                )
                for l in range(args.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(args.d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(args.d_model, args.c_out, bias=True)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], attns
        else:
            return dec_out[:,-self.pred_len:,:] # [B, L, D]

"<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<InformerStack>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

class InformerStack(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                factor=5, d_model=512, n_heads=8, e_layers=[3,2,1], d_layers=2, d_ff=512, 
                dropout=0.0,attn='prob', embed='fixed', freq='h', activation='gelu',
                output_attention = False, distil=True, mix=True,local_casual=False,
                device=torch.device('cuda:0')):
        super(InformerStack, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder

        inp_lens = list(range(len(e_layers))) # [0,1,2,...] you can customize here
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                    d_model, n_heads,q_len, mix=False,local_casual=local_casual),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(el)
                ],
                [
                    ConvLayer(
                        d_model
                    ) for l in range(el-1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            ) for el in e_layers]
        self.encoder = EncoderStack(encoders, inp_lens)
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                d_model, n_heads,q_len,mix=mix,local_casual=local_casual),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads,q_len, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)  #d_layers:2
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)   #enc_out [32,72,512]

        dec_out = self.dec_embedding(x_dec, x_mark_dec)  #[32,72,512]
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)  #32,72,512
        dec_out = self.projection(dec_out)  #[32,72,7]
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], attns
        else:
            return dec_out[:,-self.pred_len:,:] # [B, L, D]  [32,24,7]


"<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<Logsparse Transformer>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

activation_dict = {"ReLU": torch.nn.ReLU(), "Softplus": torch.nn.Softplus(), "Softmax": torch.nn.Softmax}
import numpy as np
import torch
import torch.nn as nn
import math
# from torch.distributions.normal import Normal
import copy
from torch.nn.parameter import Parameter
from typing import Dict
def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def swish(x):
    return x * torch.sigmoid(x)

ACT_FNS = {
    'relu': nn.ReLU(),
    'swish': swish,
    'gelu': gelu
}


class Attention(nn.Module):
    def __init__(self, n_head, n_embd, win_len, scale, q_len, sub_len, sparse=True, attn_pdrop=0.1, resid_pdrop=0.1):
        super(Attention, self).__init__()

        # if(sparse):
        #     print('Activate log sparse!')
        #     mask = self.log_mask(win_len, sub_len)
        # else:
        #     mask = torch.tril(torch.ones(win_len, win_len)).view(1, 1, win_len, win_len)

        #self.register_buffer('mask_tri', mask)
        self.win_len=win_len
        self.sub_len=sub_len
        self.sparse=sparse
        self.n_head = n_head
        self.split_size = n_embd * self.n_head
        self.scale = scale
        self.q_len = q_len
        self.query_key = nn.Conv1d(n_embd, n_embd * n_head * 2, self.q_len).cuda()
        self.value = Conv1D(n_embd * n_head, 1, n_embd)
        self.c_proj = Conv1D(n_embd, 1, n_embd * self.n_head)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
    # def mask(self,sparse):
    #     if(sparse):
    #         print('Activate log sparse!')
    #         return self.log_mask(win_len, sub_len)
    #     else:
    #         return torch.tril(torch.ones(win_len, win_len)).view(1, 1, win_len, win_len)
    def log_mask(self, win_len, sub_len):
        mask = torch.zeros((win_len, win_len), dtype=torch.float)
        for i in range(win_len):
            mask[i] = self.row_mask(i, sub_len, win_len)
        return mask.view(1, 1, mask.size(0), mask.size(1))

    def row_mask(self, index, sub_len, win_len):
        """
        Remark:
        1 . Currently, dense matrices with sparse multiplication are not supported by Pytorch. Efficient implementation
            should deal with CUDA kernel, which we haven't implemented yet.

        2 . Our default setting here use Local attention and Restart attention.

        3 . For index-th row, if its past is smaller than the number of cells the last
            cell can attend, we can allow current cell to attend all past cells to fully
            utilize parallel computing in dense matrices with sparse multiplication."""
        log_l = math.ceil(np.log2(sub_len))
        mask = torch.zeros((win_len), dtype=torch.float)
        if((win_len // sub_len) * 2 * (log_l) > index):
            mask[:(index + 1)] = 1
        else:
            while(index >= 0):
                if((index - log_l + 1) < 0):
                    mask[:index] = 1
                    break
                mask[index - log_l + 1:(index + 1)] = 1  # Local attention
                for i in range(0, log_l):
                    new_index = index - log_l + 1 - 2**i
                    if((index - new_index) <= sub_len and new_index >= 0):
                        mask[new_index] = 1
                index -= sub_len
        return mask

    def attn(self, query: torch.Tensor, key, value: torch.Tensor, activation="Softmax"):
        activation = activation_dict[activation](dim=-1)
        pre_att = torch.matmul(query, key)
        if self.scale:
            pre_att = pre_att / math.sqrt(value.size(-1))
        if self.sparse:
            #print('Activate log sparse!')
            mask=self.log_mask(self.win_len, self.sub_len)
        else:
            mask = torch.tril(torch.ones(self.win_len, self.win_len)).view(1, 1, self.win_len, self.win_len)
        mask = mask[:, :, :pre_att.size(-2), :pre_att.size(-1)].cuda()  #[1,1,20,20]
        pre_att = pre_att * mask + -1e9 * (1 - mask)
        pre_att = activation(pre_att)
        pre_att = self.attn_dropout(pre_att)
        attn = torch.matmul(pre_att, value)

        return attn

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)#[2, 20, 8, 16]
        x = x.view(*new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, x):

        value = self.value(x)   #[2,20,131]
        qk_x = nn.functional.pad(x.permute(0, 2, 1), pad=(self.q_len - 1, 0)).cuda()  #[2,131,20]
        query_key = self.query_key(qk_x).cuda().permute(0, 2, 1)  #[2,20,2096]
        query, key = query_key.split(self.split_size, dim=2)   #[2,20,1048] [2,20,1048]  self.split_size=n_embd * self.n_head
        query = self.split_heads(query)   #torch.Size([2, 8, 20, 131])
        key = self.split_heads(key, k=True)  #torch.Size([2, 8, 131, 20])
        value = self.split_heads(value)   #torch.Size([2, 8, 20, 131])
        attn = self.attn(query, key, value)
        attn = self.merge_heads(attn)  #[2,20,1048]
        attn = self.c_proj(attn)  #torch.Size([2, 20, 131])
        attn = self.resid_dropout(attn)  #torch.Size([2, 20, 131])
        return attn


class Conv1D(nn.Module):
    def __init__(self, out_dim, rf, in_dim):
        super(Conv1D, self).__init__()
        self.rf = rf
        self.out_dim = out_dim
        if rf == 1:
            w = torch.empty(in_dim, out_dim)
            nn.init.normal_(w, std=0.02)
            self.w = Parameter(w)
            self.b = Parameter(torch.zeros(out_dim))
        else:
            raise NotImplementedError

    def forward(self, x):
        if self.rf == 1:
            size_out = x.size()[:-1] + (self.out_dim,)  #[2,20,1048]
            x = torch.addmm(self.b.cuda(), x.view(-1, x.size(-1)).cuda(), self.w.cuda()) # [40,1048]
            x = x.view(*size_out)   #[2,20,1048]
        else:
            raise NotImplementedError
        return x


class LayerNorm(nn.Module):
    "Construct a layernorm module in the OpenAI style (epsilon inside the square root)."

    def __init__(self, n_embd, e=1e-5):
        super(LayerNorm, self).__init__()
        self.g = nn.Parameter(torch.ones(n_embd))
        self.b = nn.Parameter(torch.zeros(n_embd))
        self.e = e

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = (x - mu).pow(2).mean(-1, keepdim=True)
        x = (x - mu) / torch.sqrt(sigma + self.e)
        return self.g.cuda() * x + self.b.cuda()


class MLP(nn.Module):
    def __init__(self, n_state, n_embd, acf='relu'):
        super(MLP, self).__init__()
        n_embd = n_embd
        self.c_fc = Conv1D(n_state, 1, n_embd)
        self.c_proj = Conv1D(n_embd, 1, n_state)
        self.act = ACT_FNS[acf]
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        hidden1 = self.act(self.c_fc(x))
        hidden2 = self.c_proj(hidden1)
        return self.dropout(hidden2)


class Block(nn.Module):
    def __init__(self, n_head, win_len, n_embd, scale, q_len, sub_len, additional_params: Dict):
        super(Block, self).__init__()
        n_embd = n_embd
        self.attn = Attention(n_head, n_embd, win_len, scale, q_len, sub_len, **additional_params)
        self.ln_1 = LayerNorm(n_embd)
        self.mlp = MLP(4 * n_embd, n_embd)
        self.ln_2 = LayerNorm(n_embd)

    def forward(self, x):
        attn = self.attn(x)
        x=x.cuda()
        attn=attn.cuda()
        ln1 = self.ln_1(x + attn)
        mlp = self.mlp(ln1)
        hidden = self.ln_2(ln1 + mlp)
        return hidden


class TransformerModel(nn.Module):
    """ Transformer model """

    def __init__(self, n_time_series, n_head, sub_len, num_layer, n_embd,
                 forecast_history: int, dropout: float, scale_att, q_len, additional_params: Dict, seq_num=None):
        super(TransformerModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = n_time_series
        self.n_head = n_head
        self.seq_num = None
        if seq_num:
            self.seq_num = seq_num
            self.id_embed = nn.Embedding(seq_num, n_embd).to(self.device)
            nn.init.normal_(self.id_embed.weight, std=0.02)
        self.n_embd = n_embd
        self.win_len = forecast_history
        # The following is the implementation of this paragraph
        """ For positional encoding in Transformer, we use learnable position embedding.
        For covariates, following [3], we use all or part of year, month, day-of-the-week,
        hour-of-the-day, minute-of-the-hour, age and time-series-ID according to the granularities of datasets.
        age is the distance to the first observation in that time series [3]. Each of them except time series
        ID has only one dimension and is normalized to have zero mean and unit variance (if applicable).
        """
        self.po_embed = nn.Embedding(forecast_history, n_embd).to(self.device)
        self.drop_em = nn.Dropout(dropout)
        block = Block(n_head, forecast_history, n_embd + n_time_series, scale=scale_att,
                      q_len=q_len, sub_len=sub_len, additional_params=additional_params)
        self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(num_layer)])
        nn.init.normal_(self.po_embed.weight, std=0.02)

    def forward(self, series_id: int, x: torch.Tensor):
        """Runs  forward pass of the DecoderTransformer model.

        :param series_id:   ID of the time series
        :type series_id: int
        :param x: [description]
        :type x: torch.Tensor
        :return: [description]
        :rtype: [type]
        """
        batch_size = x.size(0)
        length = x.size(1)  # (Batch_size, length, input_dim)
        embedding_sum = torch.zeros(batch_size, length, self.n_embd).to(self.device)
        if self.seq_num:
            embedding_sum = torch.zeros(batch_size, length)
            embedding_sum = embedding_sum.fill_(series_id).type(torch.LongTensor).to(self.device)
            embedding_sum = self.id_embed(embedding_sum)
        # print("shape below")
        # print(embedding_sum.shape)
        # print(x.shape)
        # print(series_id)
        position = torch.tensor(torch.arange(length), dtype=torch.long).to(self.device)
        po_embedding = self.po_embed(position)
        embedding_sum[:] = po_embedding
        x=x.to(self.device)
        embedding_sum=embedding_sum.to(self.device)
        x = torch.cat((x, embedding_sum), dim=2).to(self.device)
        for block in self.blocks:
            x = block(x)  #[2,20,131]
        return x


class DecoderTransformer(nn.Module):
    def __init__(self, n_time_series: int, n_head: int, num_layer: int,
                 n_embd: int, forecast_history: int, dropout: float, q_len: int, additional_params: Dict,
                 activation="Softmax", forecast_length=24, scale_att: bool = False, seq_num1=None,
                 sub_len=1, mu=None):
        #(3, 8, 4, 128, 20, 0.2, 1, {}, seq_num1=3, forecast_length=1)
        """
        Args:
            n_time_series: Number of time series present in input
            n_head: Number of heads in the MultiHeadAttention mechanism
            seq_num: The number of targets to forecast
            sub_len: sub_len of the sparse attention
            num_layer: The number of transformer blocks in the model.
            n_embd: The dimention of Position embedding and time series ID embedding
            forecast_history: The number of historical steps fed into the time series model
            dropout: The dropout for the embedding of the model.
            additional_params: Additional parameters used to initalize the attention model. Can inc
        """
        super(DecoderTransformer, self).__init__()
        self.transformer = TransformerModel(n_time_series, n_head, sub_len, num_layer, n_embd, forecast_history,
                                            dropout, scale_att, q_len, additional_params, seq_num=seq_num1)
        self.softplus = nn.Softplus()
        self.mu = torch.nn.Linear(n_time_series + n_embd, 7, bias=True).cuda()
        self.sigma = torch.nn.Linear(n_time_series + n_embd, 7, bias=True).cuda()
        self._initialize_weights()
        self.mu_mode = mu
        self.forecast_len_layer = None
        if forecast_length:
            self.forecast_len_layer = nn.Linear(forecast_history, forecast_length).cuda()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, series_id: int = None):
        """
        Args:
            x: Tensor of dimension (batch_size, seq_len, number_of_time_series)
            series_id: Optional id of the series in the dataframe. Currently  not supported
        Returns:
            Case 1: tensor of dimension (batch_size, forecast_length)
            Case 2: GLoss sigma and mu: tuple of ((batch_size, forecast_history, 1), (batch_size, forecast_history, 1))
        """
        h = self.transformer(series_id, x)
        mu = self.mu(h)  #[2,20,1]
        sigma = self.sigma(h)  #[2,20,1]
        if self.mu_mode:
            sigma = self.softplus(sigma)
            return mu, sigma
        if self.forecast_len_layer:
            # Swap to (batch_size, 1, features) for linear layer
            sigma = sigma.permute(0, 2, 1)
            # Output (batch_size, forecast_len_)
            sigma = self.forecast_len_layer(sigma).permute(0, 2, 1)   #[2,1,1] [2,3,1]  [2,7,1]  [2,7,3]
        return sigma



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

class SimpleTransformer(torch.nn.Module):
    def __init__(
            self,
            number_time_series: int,
            seq_length: int = 48,
            output_seq_len: int = None,
            d_model: int = 128,
            n_heads: int = 8,
            dropout=0.1,
            forward_dim=2048,
            sigmoid=False):
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
        if output_seq_len is None:
            output_seq_len = seq_length
        self.out_seq_len = output_seq_len
        self.mask = generate_square_subsequent_mask(seq_length)
        self.dense_shape = torch.nn.Linear(number_time_series, d_model)
        self.pe = SimplePositionalEncoding(d_model)
        self.transformer = Transformer(d_model, nhead=n_heads)
        self.final_layer = torch.nn.Linear(d_model, 1)
        self.sequence_size = seq_length
        self.tgt_mask = generate_square_subsequent_mask(output_seq_len)
        self.sigmoid = None
        if sigmoid:
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
