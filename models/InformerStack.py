import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding
"<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<InformerStack>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.pred_len=args.pred_len
        self.attn=args.attn
        self.output_attention=args.output_attention

        # Encoding
        self.enc_embedding=DataEmbedding(args.enc_in, args.d_model,args. embed, args.freq, args.dropout)
        self.dec_embedding=DataEmbedding(args.dec_in,args.d_model, args.embed, args.freq, args.dropout)
        # Attention
        Attn=ProbAttention if args.attn=='prob' else FullAttention
        # Encoder

        inp_lens=list(range(len(args.s_layers)))  # [0,1,2,...] you can customize here
        encoders=[
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            Attn(False, args.factor, attention_dropout=args.dropout, output_attention=args.output_attention),
                            args.d_model, args.n_heads, args.q_len, mix=False, local_casual=args.local_casual),
                        args. d_model,
                        args.d_ff,
                        dropout=args.dropout,
                        activation=args.activation
                    ) for l in range(el)
                ],
                [
                    ConvLayer(
                        args.d_model
                    ) for l in range(el-1)
                ] if args.distil else None,
                norm_layer=torch.nn.LayerNorm(args.d_model)
            ) for el in args.s_layers]
        self.encoder=EncoderStack(encoders, inp_lens)
        # Decoder
        self.decoder=Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, args.factor, attention_dropout=args.dropout, output_attention=False),
                                   args.d_model, args.n_heads, args.q_len, mix=args.mix, local_casual=args.local_casual),
                    AttentionLayer(FullAttention(False, args.factor, attention_dropout=args.dropout, output_attention=False),
                                   args.d_model, args.n_heads, args.q_len, mix=False),
                    args.d_model,
                    args.d_ff,
                    dropout=args.dropout,
                    activation=args.activation,
                )
                for l in range(args.d_layers)  # d_layers:2
            ],
            norm_layer=torch.nn.LayerNorm(args.d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection=nn.Linear(args.d_model, args.c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out=self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns=self.encoder(enc_out, attn_mask=enc_self_mask)  # enc_out [32,72,512]

        dec_out=self.dec_embedding(x_dec, x_mark_dec)  # [32,72,512]
        dec_out=self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)  # 32,72,512
        dec_out=self.projection(dec_out)  # [32,72,7]

        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]  [32,24,7]