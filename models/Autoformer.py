import torch
import torch.nn as nn
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos, DataEmbedding_wo_pre
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer, AutoCorrelationLayer_cross, AutoCorrelation_loss, AutoCorrelation_loss_V
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
import math
import numpy as np
from models.encoder import ConvLayer
from models.attn import FullAttention, ProbAttention, logsparseAttention
from layers.Embed import PositionalEmbedding, PositionalEmbedding_part,PositionalEmbedding_decoder
class Model(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.quantiles = configs.quantiles
        self.quan = configs.quan
        # Decomp
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        attn_map = {'prob':ProbAttention, 'logsparse':logsparseAttention, 'auto':AutoCorrelation, 'auto_loss':AutoCorrelation_loss, 'auto_loss_V':AutoCorrelation_loss_V, 'full':FullAttention}
        Attn=attn_map[configs.attn]
        self.avgpool=nn.AvgPool1d(2)
        self.upsmp=torch.nn.Upsample(size=configs.seq_len,mode ='linear')#mode：'nearest', 'linear', 'bilinear', 'bicubic' and 'trilinear'
        self.upsmp2=torch.nn.Upsample(size=(configs.seq_len+self.configs.pred_len),mode ='linear')#mode：'nearest', 'linear', 'bilinear', 'bicubic' and 'trilinear'

        self.upsmp3=torch.nn.Upsample(size=int(configs.seq_len/2),mode ='linear')#mode：'nearest', 'linear', 'bilinear', 'bicubic' and 'trilinear'
        self.upsmp4=torch.nn.Upsample(size=int((configs.seq_len+self.configs.pred_len)/2),mode ='linear')#mode：'nearest', 'linear', 'bilinear', 'bicubic' and 'trilinear'
        self.con=nn.Conv1d(configs.d_model,configs.d_model,1)
        Attn_model=Attn(False, configs.sub_len, attention_dropout=configs.dropout,output_attention=configs.output_attention) if configs.attn=='logsparse' else\
                Attn(False, configs.factor, attention_dropout=configs.dropout,output_attention=configs.output_attention)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        Attn_model,
                        configs.d_model, configs.n_heads,configs.q_len,local_casual=configs.local_casual),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            [
                ConvLayer(
                    configs.d_model
                ) for l in range(configs.e_layers-1)
            ] if configs.distil else None,
            norm_layer=my_Layernorm(configs.d_model)
        )

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        Attn_model,
                        configs.d_model, configs.n_heads,configs.q_len,local_casual=configs.local_casual),
                    AutoCorrelationLayer(#cross  改两个
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads,configs.q_len,local_casual=False),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out*len(configs.quantiles) if self.quan else configs.c_out, bias=False)
        )
        self.decoder_1 = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        Attn_model,
                        configs.d_model, configs.n_heads,configs.q_len,local_casual=configs.local_casual),
                    AutoCorrelationLayer(#cross 改两个
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),  
                        configs.d_model, configs.n_heads,configs.q_len,local_casual=False),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=None
        )
        self.pos_emb=PositionalEmbedding(configs.d_model)
        self.pos_embpart=PositionalEmbedding_part(configs.d_model)

    def reset_param(self):
        self.upsmp2=torch.nn.Upsample(size=(self.configs.seq_len+self.configs.pred_len), mode='linear')
        self.upsmp4=torch.nn.Upsample(size=int((self.configs.seq_len+self.configs.pred_len)/2), mode='linear')

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
            enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init
        self.reset_param()
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]]).cuda()
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # enc
        enc_out0 = self.enc_embedding(x_enc, x_mark_enc)
        # import matplotlib.pyplot as plt
        # plt.plot(enc_out[-1, :, -1].cpu().detach().numpy())
        # plt.legend()
        # plt.show()
        enc_out, attns = self.encoder(enc_out0, attn_mask=enc_self_mask)
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)

        # dec
        # seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,trend=trend_init)
        # plt.plot(enc_out[-1, :, -1].cpu().detach().numpy())


        #11111111111111
        # enc_out=self.avgpool(enc_out.transpose(1,2)).transpose(1,2) #96-48
        # enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask) #48
        # enc_out=self.upsmp(enc_out.transpose(1,2)).transpose(1,2)
        # seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,trend=trend_init)


        #22222222222
        # mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        # zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]]).cuda()
        # seasonal_init, trend_init = self.decomp(x_enc)
        # # decoder input
        # trend_init = torch.cat([trend_init[:, -self.seq_len:, :], mean], dim=1)
        # seasonal_init = torch.cat([seasonal_init[:, -self.seq_len:, :], zeros], dim=1)
        # # enc
        # enc_out = self.enc_embedding(x_enc, x_mark_enc)
        # enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        #
        #
        # enc_out=self.avgpool(enc_out.transpose(1,2)).transpose(1,2) #96-48
        # enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask) #48
        # seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
        #                                          trend=trend_init)



        #3333333333333改decoder+layer
        # enc_out_48=self.avgpool(enc_out.transpose(1,2)).transpose(1,2) #96-48
        # enc_out_48, attns = self.encoder(enc_out_48, attn_mask=enc_self_mask) #48
        # enc_out_96=self.upsmp(enc_out_48.transpose(1,2)).transpose(1,2)  #mode：'nearest', 'linear', 'bilinear', 'bicubic' and 'trilinear'
        # seasonal_part, trend_part = self.decoder(dec_out, enc_out, enc_out_96,x_mask=dec_self_mask, cross_mask=dec_enc_mask,trend=trend_init)


        #444444444444444444444444(decoder输入96  )-----------------'label_len'**********

        # enc_out_1=self.avgpool(enc_out.transpose(1, 2)).transpose(1, 2)  # 96-48
        # # enc_out_1+=self.pos_emb(enc_out_1)
        # # enc_out_1+=self.pos_embpart(enc_out_1)
        #
        # enc_out_2, attns = self.encoder(enc_out_1, attn_mask=enc_self_mask) #48
        # enc_out_3=self.upsmp(enc_out_2.transpose(1,2)).transpose(1,2)#96
        # # enc_out_3=self.con(enc_out_3.transpose(1, 2)).transpose(1, 2)
        #
        # dec_out_1=self.avgpool(dec_out.transpose(1, 2)).transpose(1, 2)  # 96+24-池化
        # # dec_out_1+=self.pos_emb(dec_out_1)
        # # dec_out_1+=self.pos_embpart(dec_out_1)
        #
        # trend_init_0=self.avgpool(trend_init.transpose(1, 2)).transpose(1, 2)  # 96+24-池化
        # seasonal_part, trend_part_0=self.decoder_1(dec_out_1, enc_out_1, x_mask=dec_self_mask, cross_mask=dec_enc_mask,trend=trend_init_0)
        # # trend_part_1=self.upsmp2(trend_part_0.transpose(1, 2)).transpose(1, 2)  # 上采样  96+24
        # seasonal_part_1=self.upsmp2(seasonal_part.transpose(1, 2)).transpose(1, 2)  # 上采样  96+24
        # # seasonal_part_1+=self.pos_emb(seasonal_part_1)
        #
        # # seasonal_part_1=self.con(seasonal_part_1.transpose(1, 2)).transpose(1, 2)
        # seasonal_part, trend_part=self.decoder(seasonal_part_1, enc_out_3, x_mask=dec_self_mask, cross_mask=dec_enc_mask,trend=trend_init)

        # import matplotlib.pyplot as plt
        # t=trend_part
        # s=seasonal_part
        # plt.plot(t[-1, :, -1].cpu().detach().numpy(),label='t')
        # plt.plot(s[-1, :, -1].cpu().detach().numpy(),label='s')
        # plt.legend()
        # plt.show()
        #只要encoder44444444
        # mean=torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)

        # zeros=torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]]).cuda()
        # enc = torch.cat([x_enc, mean], dim=1)
        # x_mark_enc=torch.cat([x_mark_enc, x_mark_dec[:, -self.pred_len:, :]], dim=1)
        # enc_out = self.enc_embedding(enc, x_mark_enc)
        # enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # enc_out_1=self.avgpool(enc_out.transpose(1, 2)).transpose(1, 2)  # 96-48
        # enc_out_2, attns = self.encoder(enc_out_1, attn_mask=enc_self_mask) #48
        # enc_out_3=self.upsmp(enc_out_2.transpose(1,2)).transpose(1,2)#96




        #555555555555555555555(decoder输入96   )-----------------'label_len'**********

        enc_out_1=self.avgpool(enc_out.transpose(1, 2)).transpose(1, 2)  # 96-48
        enc_out_1+=self.pos_embpart(enc_out_1)

        enc_out_2, attns = self.encoder(enc_out_1, attn_mask=enc_self_mask) #48
        dec_out_1=self.avgpool(dec_out.transpose(1, 2)).transpose(1, 2)  # 96+24-池化
        dec_out_1+=self.pos_embpart(dec_out_1)

        trend_init_0=self.avgpool(trend_init.transpose(1, 2)).transpose(1, 2)  # 96+24-池化
        seasonal_part, trend_part_0=self.decoder_1(dec_out_1, enc_out_2, x_mask=dec_self_mask, cross_mask=dec_enc_mask,trend=trend_init_0)
        seasonal_part_1=self.upsmp2(seasonal_part.transpose(1, 2)).transpose(1, 2)  # 上采样  96+24
        seasonal_part_1+=self.pos_emb(seasonal_part_1)

        seasonal_part, trend_part=self.decoder(seasonal_part_1, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,trend=trend_init)

        #6666666666666666666(decoder输入96  )-----------------'label_len'**********
        # enc_out_1=self.avgpool(enc_out.transpose(1, 2)).transpose(1, 2)  # 96-48
        # enc_out_2, attns = self.encoder(enc_out_1, attn_mask=enc_self_mask) #48
        # enc_out_3=self.avgpool(enc_out_2.transpose(1, 2)).transpose(1, 2)  # 48-24
        # enc_out_4, attns = self.encoder(enc_out_3, attn_mask=enc_self_mask) #24
        # enc_out_5=self.upsmp3(enc_out_4.transpose(1,2)).transpose(1,2)#24-48
        # enc_out_6, attns = self.encoder(enc_out_5, attn_mask=enc_self_mask) #48
        # enc_out_7=self.upsmp(enc_out_6.transpose(1,2)).transpose(1,2)#48-96
        # dec_out_1=self.avgpool(dec_out.transpose(1, 2)).transpose(1, 2)  # 96+24-池化
        # trend_init_1=self.avgpool(trend_init.transpose(1, 2)).transpose(1, 2)  # 96+24-池化
        # seasonal_part_1, trend_part_1=self.decoder_1(dec_out_1, enc_out_1, x_mask=dec_self_mask, cross_mask=dec_enc_mask,trend=trend_init_1)
        # dec_out_2=self.avgpool(seaso
        # nal_part_1.transpose(1, 2)).transpose(1, 2)  # 48-池化
        # trend_init_2=self.avgpool(trend_part_1.transpose(1, 2)).transpose(1, 2)  # 48-池化
        # seasonal_part_2, trend_part_2=self.decoder_1(dec_out_2, enc_out_3, x_mask=dec_self_mask, cross_mask=dec_enc_mask,trend=trend_init_2)
        # seasonal_part_3=self.upsmp4(seasonal_part_2.transpose(1, 2)).transpose(1, 2)  # 上采样  96+24
        # trend_part_3=self.upsmp4(trend_part_2.transpose(1, 2)).transpose(1, 2)  # 上采样  96+24
        # seasonal_part_4, trend_part_4=self.decoder_1(seasonal_part_3, enc_out_5, x_mask=dec_self_mask, cross_mask=dec_enc_mask,trend=trend_part_3)
        # seasonal_part_5=self.upsmp2(seasonal_part_4.transpose(1, 2)).transpose(1, 2)  # 上采样  96+24
        # # trend_part_5=self.upsmp2(trend_part_4.transpose(1, 2)).transpose(1, 2)  # 上采样  96+24
        # seasonal_part, trend_part=self.decoder(seasonal_part_5, enc_out_7, x_mask=dec_self_mask, cross_mask=dec_enc_mask,trend=trend_init)


        # final
        if self.configs.data=='bus':
            trend_part=trend_part[..., -self.configs.c_out].unsqueeze(-1)
        if self.quan:
            trend_part=trend_part.unsqueeze(-1)
            trend_part=torch.cat((trend_part, trend_part, trend_part), -1)
            seasonal_part=seasonal_part.view(seasonal_part.size(0), seasonal_part.size(1),trend_part.size(2), len(self.quantiles))
        dec_out = trend_part + seasonal_part
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out



"--------------------------------去掉sd ->ln  -----------------------------"

# class Model(nn.Module):
#     """
#     Autoformer is the first method to achieve the series-wise connection,
#     with inherent O(LlogL) complexity
#     """
#     def __init__(self, configs):
#         super(Model, self).__init__()
#         self.seq_len = configs.seq_len
#         self.label_len = configs.label_len
#         self.pred_len = configs.pred_len
#         self.output_attention = configs.output_attention
#
#         # Decomp
#         kernel_size = configs.moving_avg
#         self.decomp = series_decomp(kernel_size)
#
#         # Embedding
#         # The series-wise connection inherently contains the sequential information.
#         # Thus, we can discard the position embedding of transformers.
#         self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
#                                                   configs.dropout)
#         self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
#                                                   configs.dropout)
#
#
#
#         # Attn=ProbAttention if configs.attn=='prob' else AutoCorrelation  AutoCorrelation_loss
#         attn_map = {'prob':ProbAttention, 'auto':AutoCorrelation, 'auto_loss':AutoCorrelation_loss, 'auto_loss_V':AutoCorrelation_loss_V, 'full':FullAttention}
#         Attn=attn_map[configs.attn]
#         self.avgpool=nn.AvgPool1d(2)
#         self.upsmp=torch.nn.Upsample(size=configs.seq_len,mode ='linear')#mode：'nearest', 'linear', 'bilinear', 'bicubic' and 'trilinear'
#
#         # Encoder
#         self.encoder = Encoder(
#             [
#                 EncoderLayer(
#                     AutoCorrelationLayer(
#                         Attn(False, configs.factor, attention_dropout=configs.dropout,
#                                         output_attention=configs.output_attention),
#                         configs.d_model, configs.n_heads,configs.q_len,local_casual=configs.local_casual),
#                     configs.d_model,
#                     configs.d_ff,
#                     moving_avg=configs.moving_avg,
#                     dropout=configs.dropout,
#                     activation=configs.activation
#                 ) for l in range(configs.e_layers)
#             ],
#             [
#                 ConvLayer(
#                     configs.d_model
#                 ) for l in range(configs.e_layers-1)
#             ] if configs.distil else None,
#             norm_layer=my_Layernorm(configs.d_model)
#         )
#
#         # Decoder
#         self.decoder = Decoder(
#             [
#                 DecoderLayer(
#                     AutoCorrelationLayer(
#                         Attn(True, configs.factor, attention_dropout=configs.dropout,
#                                         output_attention=False),
#                         configs.d_model, configs.n_heads,configs.q_len,local_casual=configs.local_casual),
#                     AutoCorrelationLayer(
#                         AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
#                                         output_attention=False),
#                         configs.d_model, configs.n_heads,configs.q_len,local_casual=configs.local_casual_cross),
#                     configs.d_model,
#                     configs.c_out,
#                     configs.d_ff,
#                     moving_avg=configs.moving_avg,
#                     dropout=configs.dropout,
#                     activation=configs.activation,
#                 )
#                 for l in range(configs.d_layers)
#             ],
#             norm_layer=my_Layernorm(configs.d_model),
#             projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
#         )
#
#     def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
#                 enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
#         # decomp init
#         mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
#         zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]]).cuda()
#         seasonal_init, trend_init = self.decomp(x_enc)
#         # decoder input
#         trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
#         seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
#         # enc
#         enc_out = self.enc_embedding(x_enc, x_mark_enc)
#         enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
#         dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
#         # dec
#         seasonal_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
#                                                  trend=trend_init)
#
#
#         #11111111111111
#         # enc_out=self.avgpool(enc_out.transpose(1,2)).transpose(1,2) #96-48
#         # enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask) #48
#         # enc_out=self.upsmp(enc_out.transpose(1,2)).transpose(1,2)
#         # seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,trend=trend_init)
#
#
#         #22222222222
#         # mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
#         # zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]]).cuda()
#         # seasonal_init, trend_init = self.decomp(x_enc)
#         # # decoder input
#         # trend_init = torch.cat([trend_init[:, -self.pred_len:, :], mean], dim=1)
#         # seasonal_init = torch.cat([seasonal_init[:, -self.pred_len:, :], zeros], dim=1)
#         # # enc
#         # enc_out = self.enc_embedding(x_enc, x_mark_enc)
#         # enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
#         # dec_out = self.dec_embedding(seasonal_init, x_mark_dec[:,-self.label_len:,:])
#         # enc_out=self.avgpool(enc_out.transpose(1,2)).transpose(1,2) #96-48
#         # enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask) #48
#         # seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
#         #                                          trend=trend_init)
#
#
#
#         #3333333333333改decoder+layer
#         # enc_out_48=self.avgpool(enc_out.transpose(1,2)).transpose(1,2) #96-48
#         # enc_out_48, attns = self.encoder(enc_out_48, attn_mask=enc_self_mask) #48
#         # enc_out_96=self.upsmp(enc_out_48.transpose(1,2)).transpose(1,2)  #mode：'nearest', 'linear', 'bilinear', 'bicubic' and 'trilinear'
#         # seasonal_part, trend_part = self.decoder(dec_out, enc_out, enc_out_96,x_mask=dec_self_mask, cross_mask=dec_enc_mask,
#         #                                  trend=trend_init)
#
#
#         # final
#         dec_out =   seasonal_part
#
#         if self.output_attention:
#             return dec_out[:, -self.pred_len:, :], attns
#         else:
#             return dec_out[:, -self.pred_len:, :]  # [B, L, D]
