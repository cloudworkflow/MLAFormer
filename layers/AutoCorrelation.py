import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt
import os
from models.losses_ts2vec_former import hierarchical_contrastive_loss
from utils.masking import TriangularCausalMask, ProbMask
from models.attn import FullAttention,ProbAttention

class AutoCorrelation(nn.Module):
    """
    AutoCorrelation Mechanism with the following two phases:
    (1) period-based dependencies discovery
    (2) time delay aggregation
    This block can replace the self-attention family mechanism seamlessly.
    """
    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1, output_attention=False):
        super(AutoCorrelation, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def time_delay_agg_training(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the training phase.
        """
        #corr[32,8,64,96]
        # corr ([32, 8, 64, 3, 32])
        head = values.shape[1]  #8
        channel = values.shape[2] #64
        length = values.shape[3] #96

        # find top k
        top_k = int(self.factor * math.log(length))  #22
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)  #[32,96]  先对8mean,再对64mea
        if corr.size(-1)<top_k:
            top_k = corr.size(-1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]  #[22]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)  #[32,22]
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)  #[32,22]
        tmp_values = values   #[32, 8, 64, 96]
        delays_agg = torch.zeros_like(values).float()
        torch.cuda.empty_cache()

        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), -1)
            torch.cuda.empty_cache()
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
            torch.cuda.empty_cache()
        return delays_agg

    def time_delay_agg_inference(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the inference phase.
        """
        batch = values.shape[0]  #torch.Size([32, 96,8, 64])
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1).cuda()
        # find top k
        top_k = int(self.factor * math.log(length))
        if corr.size(-1)<top_k:
            top_k = corr.size(-1)
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)  #[32,64,96]
        weights = torch.topk(mean_value, top_k, dim=-1)[0]  #[32,64,22]
        delay = torch.topk(mean_value, top_k, dim=-1)[1]  #[32,64,22]
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg

    def time_delay_agg_full(self, values, corr):
        """
        Standard version of Autocorrelation
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1).cuda()
        # find top k
        top_k = int(self.factor * math.log(length))
        if corr.size(-1)<top_k:
            top_k = corr.size(-1)
        weights = torch.topk(corr, top_k, dim=-1)[0]
        delay = torch.topk(corr, top_k, dim=-1)[1]
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[..., i].unsqueeze(-1)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * (tmp_corr[..., i].unsqueeze(-1))
        return delays_agg

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape  #torch.Size([32, 96, 8, 64])
        _, S, _, D = values.shape #torch.Size([32, 96, 8, 64])
        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]


        # period-based dependencies
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)   #[32, 8, 64, 49]
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)  #[32, 8, 64, 49]
        res = q_fft * torch.conj(k_fft)   #[32, 8, 64, 49]
        corr = torch.fft.irfft(res, dim=-1)  #[32, 8, 64, 96]
        # time delay agg
        # V=self.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
        if self.training:
            V = self.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
        else:
            V = self.time_delay_agg_inference(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
        return (V.contiguous(), None)


class AutoCorrelationLayer(nn.Module):
    def __init__(self, correlation, d_model, n_heads,q_len, d_keys=None,
                 d_values=None,local_casual=False):
        super(AutoCorrelationLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_correlation = correlation

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

        self.local_casual=local_casual
        self.split_size=d_model
        self.q_len=q_len
        self.query_key= nn.Conv1d(d_model, d_keys * n_heads * 2,self.q_len).cuda() #q_len
        # self.query_key=nn.Conv1d(d_model, d_keys*n_heads, self.q_len).cuda()  # q_len


    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        if self.local_casual:
            # if L>S:
            #     zeros=torch.zeros_like(queries[:, :(L-S), :]).float()
            #     values=torch.cat([values, zeros], dim=1)
            #     keys=torch.cat([keys, zeros], dim=1)
            # else:
            #     values=values[:, :L, :]
            #     keys=keys[:, :L, :]
            qk_x=nn.functional.pad(queries.permute(0, 2, 1), pad=(self.q_len-1, 0)).cuda() #queries  [32,96,512]# qk_x [32,512,97]
            query_key=self.query_key(qk_x).cuda().permute(0, 2, 1) # [2,20,2096]  [2,131,20]  [32,96,1024]  #query_key [32,96,1024]
            # [32,93,1024]  96 95 94 93
            #[32,45,1024]
            queries, keys=query_key.split(self.split_size,dim=2)
            queries = queries.view(B, L, H, -1)  # [32,96,8,64]
            keys = keys.view(B, S, H, -1)
            # keys = keys.view(B, L, H, -1)
            # values=self.value_projection(values).view(B, L, H, -1)

            # qk_x=nn.functional.pad(queries.permute(0, 2, 1), pad=(self.q_len-1, 0)).cuda()
            # queries=self.query_key(qk_x).cuda().permute(0, 2, 1)
            # keys=self.query_key(qk_x).cuda().permute(0, 2, 1)
        else:
            queries=self.query_projection(queries).view(B, L, H, -1)  # [32,96,8,64]
            keys=self.key_projection(keys).view(B, S, H, -1)
            # values = self.value_projection(values).view(B, S, H, -1)

            # queries = self.query_projection(queries).view(B, L, H, -1)  #[32,96,8,64]
        # keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_correlation(
            queries,
            keys,
            values,
            attn_mask
        )  #[64,96,8,64] [32,96,8,96]

        out = (out).view(B, L, -1)

        return self.out_projection(out), attn

class AutoCorrelationLayer_cross(nn.Module):
    def __init__(self, correlation, d_model, n_heads,q_len, d_keys=None,
                 d_values=None,local_casual=False):
        super(AutoCorrelationLayer_cross, self).__init__()

        d_keys=d_keys or (d_model//n_heads)
        d_values=d_values or (d_model//n_heads)

        self.inner_correlation=correlation
        self.cross_attention_2 = ProbAttention(False, 5, attention_dropout=0.05,output_attention=False)
        self.query_projection=nn.Linear(d_model, d_keys*n_heads)
        self.key_projection=nn.Linear(d_model, d_keys*n_heads)
        self.value_projection=nn.Linear(d_model, d_values*n_heads)
        self.out_projection=nn.Linear(d_values*n_heads, d_model)
        self.n_heads=n_heads

        self.local_casual=local_casual
        self.split_size=d_model
        self.q_len=q_len
        # self.query_key=nn.Conv1d(d_model, d_keys*n_heads*2, self.q_len).cuda()  # q_len
        self.query_key=nn.Conv1d(d_model, d_keys*n_heads, self.q_len).cuda()  # q_len

    def forward(self, queries, keys, values, attn_mask):
        B, L, _=queries.shape
        _, S, _=keys.shape
        H=self.n_heads

        if self.local_casual:
            qk_x=nn.functional.pad(queries.permute(0, 2, 1),pad=(self.q_len-1, 0)).cuda()  # queries  [32,96,512]# qk_x [32,512,97]
            queries=self.query_key(qk_x).cuda().permute(0, 2,1)  # [2,20,2096]  [2,131,20]  [32,96,1024]  #query_key [32,96,1024]
            # [32,93,1024]  96 95 94 93
            # [32,45,1024]
            # queries, keys=query_key.split(self.split_size, dim=2)
            qk_x_key=nn.functional.pad(keys.permute(0, 2, 1),pad=(self.q_len-1, 0)).cuda()  # queries  [32,96,512]# qk_x [32,512,97]
            keys=self.query_key(qk_x_key).cuda().permute(0, 2, 1)  # [2,20,2096]  [2,131,20]  [32,96,1024]  #query_key [32,96,1024]

            queries=queries.view(B, L, H, -1)  # [32,96,8,64]
            keys=keys.view(B, S, H, -1)

            # qk_x=nn.functional.pad(queries.permute(0, 2, 1), pad=(self.q_len-1, 0)).cuda()
            # queries=self.query_key(qk_x).cuda().permute(0, 2, 1)
            # keys=self.query_key(qk_x).cuda().permute(0, 2, 1)
        else:
            queries=self.query_projection(queries).view(B, L, H, -1)  # [32,96,8,64]
            keys=self.key_projection(keys).view(B, S, H, -1)

            # queries = self.query_projection(queries).view(B, L, H, -1)  #[32,96,8,64]
        # keys = self.key_projection(keys).view(B, S, H, -1)
        values=self.value_projection(values).view(B, S, H, -1)
        out, attn=self.inner_correlation(
            queries,
            keys,
            values,
            attn_mask
        )  # [64,96,8,64] [32,96,8,96]
        # out_2, attn = self.cross_attention_2(
        #     queries,
        #     keys,
        #     values,
        #     attn_mask
        # )  #[64,96,8,64] [32,96,8,96]

        out=(out ).view(B, L, -1)

        return self.out_projection(out), attn

class AutoCorrelation_loss(nn.Module):
    """
    AutoCorrelation Mechanism with the following two phases:
    (1) period-based dependencies discovery
    (2) time delay aggregation
    This block can replace the self-attention family mechanism seamlessly.
    """
    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1, output_attention=False):
        super(AutoCorrelation_loss, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def time_delay_agg_training(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the training phase.
        """
        head = values.shape[1]  #8
        channel = values.shape[2] #64
        length = values.shape[3] #96
        # find top k
        top_k = int(self.factor * math.log(corr.size()[1]))  #22
        index = torch.topk(torch.mean(corr, dim=0), top_k, dim=-1)[1]  #[22]
        weights = torch.stack([corr[:, index[i]] for i in range(top_k)], dim=-1)  #[32,22]
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)  #[32,22]
        # aggregation
        tmp_values = values   #[32, 8, 64, 96]
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), -1) #[32, 8, 64, 96]
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))#[32, 8, 64, 96]
        return delays_agg

    def time_delay_agg_inference(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the inference phase.
        """
        batch = values.shape[0]  #torch.Size([32, 96,8, 64])
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1).cuda() #[32,8,64,96]
        # find top k
        top_k = int(self.factor * math.log(corr.size()[1]))
        mean_value = torch.mean(corr, dim=1)
        weights = torch.topk(mean_value, top_k, dim=-1)[0]
        delay = torch.topk(mean_value, top_k, dim=-1)[1] #[22]
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg

    def time_delay_agg_full(self, values, corr):
        """
        Standard version of Autocorrelation
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1).cuda()
        # find top k
        top_k = int(self.factor * math.log(length))
        weights = torch.topk(corr, top_k, dim=-1)[0]
        delay = torch.topk(corr, top_k, dim=-1)[1]
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[..., i].unsqueeze(-1)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * (tmp_corr[..., i].unsqueeze(-1))
        return delays_agg

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape  #torch.Size([32, 96, 8, 64])
        _, S, _, D = values.shape #torch.Size([32, 96, 8, 64])
        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        corr= hierarchical_contrastive_loss(queries.view(B, L,-1),keys.view(B, L,-1))  #[32,96]

        # time delay agg
        V = self.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)


        if self.output_attention:
            return (V.contiguous(), corr.permute(0, 3, 1, 2))
        else:
            return (V.contiguous(), None)


class AutoCorrelation_loss_V(nn.Module):
    """
    AutoCorrelation Mechanism with the following two phases:
    (1) period-based dependencies discovery
    (2) time delay aggregation
    This block can replace the self-attention family mechanism seamlessly.
    """
    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1, output_attention=False):
        super(AutoCorrelation_loss_V, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape  #32,8,96,64
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)   #32,8,64
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else: # use mask
            assert(L_Q == L_V) # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape
        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)  #32,8,72,[32,8,25],[32,8,25,72]
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1) # nn.Softmax(dim=-1)(scores)  #32,8,25,96

        context_in[torch.arange(B)[:, None, None],   #[32,8,96,64]
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)   #  32,8,22,96   32,8,96,64
        return (context_in, None)


    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape  #torch.Size([32, 96, 8, 64])
        _, S, _, D = values.shape #torch.Size([32, 96, 8, 64])
        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        corr= hierarchical_contrastive_loss(queries.view(B, L,-1),keys.view(B, L,-1))  #[32,96]
        #corr=corr.unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)) # [32, 8, 25,96]
        top_k = int(self.factor * math.log(corr.size()[1]))  #22
        index = torch.topk(torch.mean(corr, dim=0), top_k, dim=-1)[1] #[22]
        corr_top=torch.stack([corr[:, index[i]] for i in range(top_k)], dim=-1)  #[32,22]
        corr_top =corr_top.unsqueeze(1).repeat(1,H,1)  # [32,8,22]
        corr_top=corr_top.unsqueeze(1).repeat(1, L, 1,1).transpose(1,2)
        index=index.unsqueeze(1).transpose(0,1).repeat(B,1).unsqueeze(1).repeat(1,H,1)  #32,8,22
        values=values.transpose(2, 1)
        context=self._get_initial_context(values, L)
        context, attn = self._update_context(context, values, corr_top.transpose(2,3), index, L, attn_mask)

        return (context, None)