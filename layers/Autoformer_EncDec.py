import torch
import torch.nn as nn
import torch.nn.functional as F
# from PyEMD import EMD
from models.attn import FullAttention, ProbAttention
from layers.Embed import  DataEmbedding_wo_pos

class my_Layernorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """
    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
    def forward(self, x):
        # padding on the both ends of time series
        pad= (self.kernel_size) // 2  if self.kernel_size%2==0 else (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, pad, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1) #92  #87
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        # kernel_size=torch.arange(5,30,5)
        # self.moving_avg_1 = moving_avg(kernel_size[0].item(), stride=1)
        # self.moving_avg_2 = moving_avg(kernel_size[1].item(), stride=1)
        # self.moving_avg_3 = moving_avg(kernel_size[2].item(), stride=1)
        # self.moving_avg_4 = moving_avg(kernel_size[3].item(), stride=1)
        self.moving_avg = moving_avg(kernel_size, stride=1)
    def forward(self, x):
        # moving_mean_1 = self.moving_avg_1(x)
        # moving_mean_2 = self.moving_avg_2(x)
        # moving_mean_3 = self.moving_avg_3(x)
        # moving_mean_4 = self.moving_avg_4(x)
        moving_mean_5 = self.moving_avg(x) #[32,96,7]

        res = x - moving_mean_5
        return res, moving_mean_5

# def stl(x):
#     emd = EMD()
#     emd.emd(x)
#     imfs, res = emd.get_imfs_and_residue()
#     # if imfs.shape[0]>0:
#     return imfs[int(imfs.shape[0]/2),:]
    # else:
    #     return

# class series_decomp_emd(nn.Module):
#     """
#     Series decomposition block
#     """
#     def __init__(self, kernel_size):
#         super(series_decomp_emd, self).__init__()
#         self.moving_avg = moving_avg(kernel_size, stride=1)
#     def forward(self, x):
#         # moving_mean = self.moving_avg(x)
#         x=x.cpu().detach().numpy()
#         moving_mean=[stl(x[i, :,j]) for i in range(x.shape[0]) for j in range(x.shape[2])]
#         x=torch.tensor(x).cuda()
#         moving_mean=torch.tensor(moving_mean).cuda()
#         moving_mean=moving_mean.view(x.size(0),x.size(2),x.size(1))
#         moving_mean=moving_mean.transpose(1,2)
#         res = x - moving_mean
#         return  res,moving_mean

# class series_decomp(nn.Module):
#     """
#     Series decomposition block
#     """
#     def __init__(self, kernel_size):
#         super(series_decomp, self).__init__()
#         self.moving_avg = moving_avg(kernel_size, stride=1)
#
#     def forward(self, x):
#         moving_mean = self.moving_avg(x)
#         res = x - moving_mean
#         return res, moving_mean





# class con(nn.Module):
#     def __init__(self):
#         super(con, self).__init__()
#         self.conv1=nn.Sequential(
#             nn.Conv1d(in_channels=512,
#                       out_channels=512,
#                       kernel_size=3,
#                       ),
#             nn.MaxPool1d(kernel_size=2)
#         )
#         self.conv2=nn.Sequential(
#             nn.Conv1d(in_channels=512,
#                       out_channels=512,
#                       kernel_size=3,
#                       ),
#             nn.MaxPool1d(kernel_size=2)
#         )
#
#     def forward(self, x):
#         x=self.conv1(x)
#         x=self.conv2(x) #[32,512,]
#         return x

"-------------------------autoformer--------------------------------------"

# class EncoderLayer(nn.Module):
#     """
#     Autoformer encoder layer with the progressive decomposition architecture
#     """
#     def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
#         super(EncoderLayer, self).__init__()
#         d_ff = d_ff or 4 * d_model
#
#         self.attention = attention
#         self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
#         self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
#         self.decomp1 = series_decomp(moving_avg)
#         self.decomp2 = series_decomp(moving_avg)
#         self.dropout = nn.Dropout(dropout)
#         self.activation = F.relu if activation == "relu" else F.gelu
#
#     def forward(self, x, attn_mask=None):
#         new_x, attn = self.attention(
#             x, x, x,
#             attn_mask=attn_mask
#         )
#         x = x + self.dropout(new_x)
#         x, _ = self.decomp1(x)  #[32,96,512]
#         y = x
#         y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))  #[32,2048,96]
#         y = self.dropout(self.conv2(y).transpose(-1, 1))  #[32,96,512]
#         res, _ = self.decomp2(x + y)
#         return res, attn
#
#
# class DecoderLayer(nn.Module):
#     """
#     Autoformer decoder layer with the progressive decomposition architecture
#     """
#     def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
#                  moving_avg=25, dropout=0.1, activation="relu"):
#         super(DecoderLayer, self).__init__()
#         d_ff = d_ff or 4 * d_model
#         self.self_attention = self_attention
#         self.cross_attention = cross_attention
#         self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
#         self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
#         self.decomp1 = series_decomp(moving_avg)
#         self.decomp2 = series_decomp(moving_avg)
#         self.decomp3 = series_decomp(moving_avg)
#         self.dropout = nn.Dropout(dropout)
#         self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
#                                     padding_mode='circular', bias=False)
#         self.activation = F.relu if activation == "relu" else F.gelu
#
#     def forward(self, x, cross, x_mask=None, cross_mask=None):
#         x = x + self.dropout(self.self_attention(
#             x, x, x,
#             attn_mask=x_mask
#         )[0])
#         x, trend1 = self.decomp1(x) #[32,72,512]
#         x = x + self.dropout(self.cross_attention(
#             x, cross, cross,
#             attn_mask=cross_mask
#         )[0])
#         x, trend2 = self.decomp2(x) #[32,72,512]
#         y = x
#         torch.cuda.empty_cache()
#         y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
#         torch.cuda.empty_cache()
#         y = self.dropout(self.conv2(y).transpose(-1, 1))
#         x, trend3 = self.decomp3(x + y)
#
#         residual_trend = trend1 + trend2 + trend3
#         residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
#         return x, residual_trend

class Encoder(nn.Module):
    """
    Autoformer encoder
    """
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer


    def forward(self, x,attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)
        return x, attns

class Decoder(nn.Module):
    """
    Autoformer encoder
    """
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection
        self.projection_1 = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            trend=trend.cuda()+residual_trend.cuda()
        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)

        return x, trend


"---------------------------sd前置----------------------------"

class EncoderLayer(nn.Module):
    """
    Autoformer encoder layer with the progressive decomposition architecture
    """
    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        x, _=self.decomp1(x)
        new_x, attn=self.attention(
            x, x, x,
            attn_mask=attn_mask
        )

        x=x+self.dropout(new_x)
        # [32,96,512]

        res, _=self.decomp2(x)
        y=res
        y=self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))  # [32,2048,96]
        y=self.dropout(self.conv2(y).transpose(-1, 1))  # [32,96,512]
        # y=self.activation(self.conv1(y.transpose(-1, 1))) # [32,2048,96]
        # y=self.conv2(y).transpose(-1, 1)  # [32,96,512]
        return res+y, attn

class DecoderLayer(nn.Module):
    """
    Autoformer decoder layer with the progressive decomposition architecture
    """
    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
                 moving_avg=25, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.cross_attention_2 = FullAttention(False, 5, attention_dropout=0.05,output_attention=False)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.decomp3 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)

        self.activation = F.relu if activation == "relu" else F.gelu
    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x, trend1 = self.decomp1(x) #[32,72,512]
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        # x = x + self.self_attention(
        #     x, x, x,
        #     attn_mask=x_mask
        # )[0]
        x, trend2 = self.decomp2(x) #[32,72,512]
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0] )
        # x = x + self.cross_attention(
        #     x, cross, cross,
        #     attn_mask=cross_mask
        # )[0]
        y = x
        torch.cuda.empty_cache()
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        # y = self.activation(self.conv1(y.transpose(-1, 1)))

        torch.cuda.empty_cache()
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        # y = self.conv2(y).transpose(-1, 1)

        x, trend3=self.decomp3(x+y)
        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        return x, residual_trend

"------------------------------残差2模块(-dec3)-------------------------"
'''
class EncoderLayer(nn.Module):
    """
    Autoformer encoder layer with the progressive decomposition architecture
    """
    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        y=x
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x =self.dropout(new_x)
        x, _ = self.decomp1(x)  #[32,96,512]
        y = x+y
        x=y
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))  #[32,2048,96]
        y = self.dropout(self.conv2(y).transpose(-1, 1))  #[32,96,512]
        res, _ = self.decomp2(y)
        return x+res, attn

class DecoderLayer(nn.Module):
    """
    Autoformer decoder layer with the progressive decomposition architecture
    """
    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
                 moving_avg=25, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.decomp3 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        y=x
        x =  self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x, trend1 = self.decomp1(x) #[32,72,512]
        x=x+y
        y=x
        x = self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])
        x, trend2 = self.decomp2(x) #[32,72,512]
        x=x+y
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)
        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        return x, residual_trend
'''

"---------------------------------decoder后3个模块换成informer/transformer-------------------"
'''
class EncoderLayer(nn.Module):
    """
    Autoformer encoder layer with the progressive decomposition architecture
    """
    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        x, _ = self.decomp1(x)  #[32,96,512]
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))  #[32,2048,96]
        y = self.dropout(self.conv2(y).transpose(-1, 1))  #[32,96,512]
        res, _ = self.decomp2(x + y)
        return res, attn

class DecoderLayer(nn.Module):
    """
    Autoformer decoder layer with the progressive decomposition architecture
    """
    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
                 moving_avg=25, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.decomp3 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu

        self.norm2=nn.LayerNorm(d_model)
    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x, trend1 = self.decomp1(x) #[32,72,512]
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])
        # x, trend2 = self.decomp2(x) #[32,72,512]
        x=self.norm2(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)
        residual_trend = trend1  + trend3
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        return x, residual_trend
'''
"------------------------------池化33333decoder+layer---------------------------"
#
# class DecoderLayer(nn.Module):
#     """
#     Autoformer decoder layer with the progressive decomposition architecture
#     """
#     def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
#                  moving_avg=25, dropout=0.1, activation="relu"):
#         super(DecoderLayer, self).__init__()
#         d_ff = d_ff or 4 * d_model
#         self.self_attention = self_attention
#         self.cross_attention = cross_attention
#         self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
#         self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
#         self.decomp1 = series_decomp(moving_avg)
#         self.decomp2 = series_decomp(moving_avg)
#         self.decomp3 = series_decomp(moving_avg)
#         self.dropout = nn.Dropout(dropout)
#         self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
#                                     padding_mode='circular', bias=False)
#         self.activation = F.relu if activation == "relu" else F.gelu
#
#     def forward(self, x, cross,cross_pool ,x_mask=None, cross_mask=None):
#         x = x + self.dropout(self.self_attention(
#             x, x, x,
#             attn_mask=x_mask
#         )[0])
#         x, trend1 = self.decomp1(x) #[32,72,512]
#         x = x + self.dropout(self.cross_attention(
#             x, cross, cross_pool,
#             attn_mask=cross_mask
#         )[0])
#         x, trend2 = self.decomp2(x) #[32,72,512]
#         y = x
#         y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
#         y = self.dropout(self.conv2(y).transpose(-1, 1))
#         x, trend3 = self.decomp3(x + y)
#
#         residual_trend = trend1 + trend2 + trend3
#         residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
#         return x, residual_trend
#
#
# class Decoder(nn.Module):
#     """
#     Autoformer encoder
#     """
#     def __init__(self, layers, norm_layer=None, projection=None):
#         super(Decoder, self).__init__()
#         self.layers = nn.ModuleList(layers)
#         self.norm = norm_layer
#         self.projection = projection
#
#     def forward(self, x, cross, cross_pool,x_mask=None, cross_mask=None, trend=None):
#         for layer in self.layers:
#             x, residual_trend = layer(x, cross, cross_pool,x_mask=x_mask, cross_mask=cross_mask)
#             trend = trend.cuda() + residual_trend.cuda()
#
#         if self.norm is not None:
#             x = self.norm(x)
#
#         if self.projection is not None:
#             x = self.projection(x)
#         return x, trend

"------------------------------池化33333decoder+layer 前置-dec3 ---------------------------"
#
# class EncoderLayer(nn.Module):
#     """
#     Autoformer encoder layer with the progressive decomposition architecture
#     """
#     def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
#         super(EncoderLayer, self).__init__()
#         d_ff = d_ff or 4 * d_model
#         self.attention = attention
#         self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
#         self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
#         self.decomp1 = series_decomp(moving_avg)
#         self.decomp2 = series_decomp(moving_avg)
#         self.dropout = nn.Dropout(dropout)
#         self.activation = F.relu if activation == "relu" else F.gelu
#
#     def forward(self, x, attn_mask=None):
#         x, _=self.decomp1(x)
#         new_x, attn=self.attention(
#             x, x, x,
#             attn_mask=attn_mask
#         )
#         x=x+self.dropout(new_x)
#         # [32,96,512]
#
#         res, _=self.decomp2(x)
#         y=res
#         y=self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))  # [32,2048,96]
#         y=self.dropout(self.conv2(y).transpose(-1, 1))  # [32,96,512]
#         return res+y, attn
#
# class DecoderLayer(nn.Module):
#     """
#     Autoformer decoder layer with the progressive decomposition architecture
#     """
#     def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
#                  moving_avg=25, dropout=0.1, activation="relu"):
#         super(DecoderLayer, self).__init__()
#         d_ff = d_ff or 4 * d_model
#         self.self_attention = self_attention
#         self.cross_attention = cross_attention
#         self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
#         self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
#         self.decomp1 = series_decomp(moving_avg)
#         self.decomp2 = series_decomp(moving_avg)
#         self.decomp3 = series_decomp(moving_avg)
#         self.dropout = nn.Dropout(dropout)
#         self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
#                                     padding_mode='circular', bias=False)
#         self.activation = F.relu if activation == "relu" else F.gelu
#
#     def forward(self, x, cross,cross_pool, x_mask=None, cross_mask=None):
#         x, trend1 = self.decomp1(x) #[32,72,512]
#         x = x + self.dropout(self.self_attention(
#             x, x, x,
#             attn_mask=x_mask
#         )[0])
#         x, trend2 = self.decomp2(x) #[32,72,512]
#         x = x + self.dropout(self.cross_attention(
#             x, cross, cross_pool,
#             attn_mask=cross_mask
#         )[0])
#         y = x
#         y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
#         y = self.dropout(self.conv2(y).transpose(-1, 1))
#         x, trend3=self.decomp3(x+y)
#         residual_trend = trend1 + trend2 + trend3
#         residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
#         return x, residual_trend
#
# class Decoder(nn.Module):
#     """
#     Autoformer encoder
#     """
#     def __init__(self, layers, norm_layer=None, projection=None):
#         super(Decoder, self).__init__()
#         self.layers = nn.ModuleList(layers)
#         self.norm = norm_layer
#         self.projection = projection
#
#     def forward(self, x, cross, cross_pool,x_mask=None, cross_mask=None, trend=None):
#         for layer in self.layers:
#             x, residual_trend = layer(x, cross, cross_pool,x_mask=x_mask, cross_mask=cross_mask)
#             trend = trend.cuda() + residual_trend.cuda()
#
#         if self.norm is not None:
#             x = self.norm(x)
#
#         if self.projection is not None:
#             x = self.projection(x)
#         return x, trend

"---------------------------(encoder__残差_2+sd前置-dec3)----------------------------"
'''
class EncoderLayer(nn.Module):
    """
    Autoformer encoder layer with the progressive decomposition architecture
    """
    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        y=x
        x, _=self.decomp1(x)
        new_x, attn=self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x=y+self.dropout(new_x)
        # [32,96,512]
        
        y=x
        y, _=self.decomp2(y)
        y=self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))  # [32,2048,96]
        y=self.dropout(self.conv2(y).transpose(-1, 1))  # [32,96,512]
        return x+y, attn

class DecoderLayer(nn.Module):
    """
    Autoformer decoder layer with the progressive decomposition architecture
    """
    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
                 moving_avg=25, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.decomp3 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x, trend1 = self.decomp1(x) #[32,72,512]
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x, trend2 = self.decomp2(x) #[32,72,512]
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        y = x        
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3=self.decomp3(x+y)
        
        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        return x, residual_trend
'''
"---------------------------sd前置enc+后3模块----------------------------"
'''
class EncoderLayer(nn.Module):
    """
    Autoformer encoder layer with the progressive decomposition architecture
    """
    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        x, _=self.decomp1(x)
        new_x, attn=self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x=x+self.dropout(new_x)
        # [32,96,512]

        res, _=self.decomp2(x)
        y=res
        y=self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))  # [32,2048,96]
        y=self.dropout(self.conv2(y).transpose(-1, 1))  # [32,96,512]
        return res+y, attn

class DecoderLayer(nn.Module):
    """
    Autoformer decoder layer with the progressive decomposition architecture
    """
    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
                 moving_avg=25, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.decomp3 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.norm2=nn.LayerNorm(d_model)
        
    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x, trend1 = self.decomp1(x) #[32,72,512]
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])
        # x, trend2 = self.decomp2(x) #[32,72,512]
        x=self.norm2(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)
        residual_trend = trend1  + trend3
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        return x, residual_trend
'''

"---------------------------sd + ln or bn----------------------------"
# class EncoderLayer(nn.Module):
#     """
#     Autoformer encoder layer with the progressive decomposition architecture
#     """
#     def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
#         super(EncoderLayer, self).__init__()
#         d_ff = d_ff or 4 * d_model
#
#         self.attention = attention
#         self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
#         self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
#         self.decomp1 = series_decomp(moving_avg)
#         self.decomp2 = series_decomp(moving_avg)
#         self.dropout = nn.Dropout(dropout)
#         self.activation = F.relu if activation == "relu" else F.gelu
#         # self.norm1=nn.LayerNorm(d_model)
#         # self.norm2=nn.LayerNorm(d_model)
#         self.norm1=nn.BatchNorm1d(d_model)
#         self.norm2=nn.BatchNorm1d(d_model)
#
#
#
#     def forward(self, x, attn_mask=None):
#         new_x, attn = self.attention(
#             x, x, x,
#             attn_mask=attn_mask
#         )
#         x = x + self.dropout(new_x)
#         x, _ = self.decomp1(x)  #[32,96,512]
#         # x=self.norm1(x)
#
#         x=self.norm1(x.transpose(1,2)).transpose(1,2)
#
#         y = x
#         y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))  #[32,2048,96]
#         y = self.dropout(self.conv2(y).transpose(-1, 1))  #[32,96,512]
#         res, _ = self.decomp2(x + y)
#         # res=self.norm2(x)
#
#         res=self.norm2((res).transpose(1,2)).transpose(1,2)
#
#         return res, attn
#
#
# class DecoderLayer(nn.Module):
#     """
#     Autoformer decoder layer with the progressive decomposition architecture
#     """
#     def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
#                  moving_avg=25, dropout=0.1, activation="relu"):
#         super(DecoderLayer, self).__init__()
#         d_ff = d_ff or 4 * d_model
#         self.self_attention = self_attention
#         self.cross_attention = cross_attention
#         self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
#         self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
#         self.decomp1 = series_decomp(moving_avg)
#         self.decomp2 = series_decomp(moving_avg)
#         self.decomp3 = series_decomp(moving_avg)
#         self.dropout = nn.Dropout(dropout)
#         self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
#                                     padding_mode='circular', bias=False)
#         self.activation = F.relu if activation == "relu" else F.gelu
#         # self.norm1=nn.LayerNorm(d_model)
#         # self.norm2=nn.LayerNorm(d_model)
#         # self.norm3=nn.LayerNorm(d_model)
#         self.norm1=nn.BatchNorm1d(d_model)
#         self.norm2=nn.BatchNorm1d(d_model)
#         self.norm3=nn.BatchNorm1d(d_model)
#
#
#     def forward(self, x, cross, x_mask=None, cross_mask=None):
#         x = x + self.dropout(self.self_attention(
#             x, x, x,
#             attn_mask=x_mask
#         )[0])
#         x, trend1 = self.decomp1(x) #[32,72,512]
#         # x=self.norm1(x)
#         x=self.norm1(x.transpose(1,2)).transpose(1,2)
#
#         x = x + self.dropout(self.cross_attention(
#             x, cross, cross,
#             attn_mask=cross_mask
#         )[0])
#         x, trend2 = self.decomp2(x) #[32,72,512]
#         # x=self.norm2(x)
#         x=self.norm2(x.transpose(1,2)).transpose(1,2)
#
#         y = x
#         y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
#         y = self.dropout(self.conv2(y).transpose(-1, 1))
#         x, trend3 = self.decomp3(x + y)
#         # x=self.norm3(x)
#         x=self.norm3(x.transpose(1,2)).transpose(1,2)
#
#         residual_trend = trend1 + trend2 + trend3
#         residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
#         return x, residual_trend

"---------------------------sd -> ln or bn (encoder+decoder注释掉,model换)----------------------------"
# class EncoderLayer(nn.Module):
#     """
#     Autoformer encoder layer with the progressive decomposition architecture
#     """
#     def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
#         super(EncoderLayer, self).__init__()
#         d_ff = d_ff or 4 * d_model
#
#         self.attention = attention
#         self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
#         self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
#         self.decomp1 = series_decomp(moving_avg)
#         self.decomp2 = series_decomp(moving_avg)
#         self.dropout = nn.Dropout(dropout)
#         self.activation = F.relu if activation == "relu" else F.gelu
#         # self.norm1=nn.LayerNorm(d_model)
#         # self.norm2=nn.LayerNorm(d_model)
#         self.norm1=nn.BatchNorm1d(d_model)
#         self.norm2=nn.BatchNorm1d(d_model)
#
#
#     def forward(self, x, attn_mask=None):
#         new_x, attn = self.attention(
#             x, x, x,
#             attn_mask=attn_mask
#         )
#         x = x + self.dropout(new_x)
#         # x, _ = self.decomp1(x)  #[32,96,512]
#         # x=self.norm1(x)
#         x=self.norm1(x.transpose(1, 2)).transpose(1, 2)
#         y = x
#         y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))  #[32,2048,96]
#         y = self.dropout(self.conv2(y).transpose(-1, 1))  #[32,96,512]
#         # res, _ = self.decomp2(x + y)
#         # res=self.norm2(x+y)
#         res=self.norm2((x+y).transpose(1, 2)).transpose(1, 2)
#         return res, attn
#
#
# class DecoderLayer(nn.Module):
#     """
#     Autoformer decoder layer with the progressive decomposition architecture
#     """
#     def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
#                  moving_avg=25, dropout=0.1, activation="relu"):
#         super(DecoderLayer, self).__init__()
#         d_ff = d_ff or 4 * d_model
#         self.self_attention = self_attention
#         self.cross_attention = cross_attention
#         self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
#         self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
#         self.decomp1 = series_decomp(moving_avg)
#         self.decomp2 = series_decomp(moving_avg)
#         self.decomp3 = series_decomp(moving_avg)
#         self.dropout = nn.Dropout(dropout)
#         self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
#                                     padding_mode='circular', bias=False)
#         self.activation = F.relu if activation == "relu" else F.gelu
#         # self.norm1=nn.LayerNorm(d_model)
#         # self.norm2=nn.LayerNorm(d_model)
#         # self.norm3=nn.LayerNorm(d_model)
#         self.norm1=nn.BatchNorm1d(d_model)
#         self.norm2=nn.BatchNorm1d(d_model)
#         self.norm3=nn.BatchNorm1d(d_model)
#
#     def forward(self, x, cross, x_mask=None, cross_mask=None):
#         x = x + self.dropout(self.self_attention(
#             x, x, x,
#             attn_mask=x_mask
#         )[0])
#         # x=self.norm1(x)
#         x=self.norm1(x.transpose(1, 2)).transpose(1, 2)
#
#         x = x + self.dropout(self.cross_attention(
#             x, cross, cross,
#             attn_mask=cross_mask
#         )[0])
#         # x=self.norm2(x)
#         x=self.norm2(x.transpose(1, 2)).transpose(1, 2)
#         y = x
#         y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
#         y = self.dropout(self.conv2(y).transpose(-1, 1))
#         # x=self.norm3(x+y)
#         x=self.norm3((x+y).transpose(1, 2)).transpose(1, 2)
#
#         return x
#
#
#
# class Encoder(nn.Module):
#     """
#     Autoformer encoder
#     """
#     def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
#         super(Encoder, self).__init__()
#         self.attn_layers = nn.ModuleList(attn_layers)
#         self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
#         self.norm = norm_layer
#
#     def forward(self, x, attn_mask=None):
#         attns = []
#         if self.conv_layers is not None:
#             for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
#                 x, attn = attn_layer(x, attn_mask=attn_mask)
#                 x = conv_layer(x)
#                 attns.append(attn)
#             x, attn = self.attn_layers[-1](x)
#             attns.append(attn)
#         else:
#             for attn_layer in self.attn_layers:
#                 x, attn = attn_layer(x, attn_mask=attn_mask)
#                 attns.append(attn)
#
#         if self.norm is not None:
#             x = self.norm(x)
#
#         return x, attns
#
# class Decoder(nn.Module):
#     """
#     Autoformer encoder
#     """
#     def __init__(self, layers, norm_layer=None, projection=None):
#         super(Decoder, self).__init__()
#         self.layers = nn.ModuleList(layers)
#         self.norm = norm_layer
#         self.projection = projection
#
#     def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
#         for layer in self.layers:
#             x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
#         if self.norm is not None:
#             x = self.norm(x)
#
#         if self.projection is not None:
#             x = self.projection(x)
#         return x
