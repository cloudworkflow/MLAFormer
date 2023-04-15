import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
from torch.nn.parameter import Parameter
import  math
activation_dict = {"ReLU": torch.nn.ReLU(), "Softplus": torch.nn.Softplus(), "Softmax": torch.nn.Softmax}
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

class logsparseAttention(nn.Module):
    def __init__(self,mask_flag=True,sub_len=5,scale=None, attention_dropout=0.1,output_attention=False, q_len=2,  sparse=True):
        super(logsparseAttention, self).__init__()

        # if(sparse):
        #     print('Activate log sparse!')
        #     mask = self.log_mask(win_len, sub_len)
        # else:
        #     mask = torch.tril(torch.ones(win_len, win_len)).view(1, 1, win_len, win_len)

        #self.register_buffer('mask_tri', mask)
        self.sub_len=sub_len
        self.sparse=sparse
        self.scale = scale
        self.q_len = q_len
        self.attn_dropout = nn.Dropout(attention_dropout)
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
        win_len=query.size(1)
        query = query.transpose(1, 2)
        key = key.permute(0, 2, 3, 1)
        value = value.transpose(1, 2)
        activation = activation_dict[activation](dim=-1)
        pre_att = torch.matmul(query, key)
        if self.scale:
            pre_att = pre_att / math.sqrt(value.size(-1))
        if self.sparse:
            #print('Activate log sparse!')
            mask=self.log_mask(win_len, self.sub_len)
        else:
            mask = torch.tril(torch.ones(win_len, win_len)).view(1, 1, win_len, win_len)
        mask = mask[:, :, :pre_att.size(-2), :pre_att.size(-1)].cuda()  #[1,1,20,20]
        pre_att = pre_att * mask + -1e9 * (1 - mask)
        pre_att = activation(pre_att)
        pre_att = self.attn_dropout(pre_att)
        attn = torch.matmul(pre_att, value)  #[32,8,96,64]

        return attn

    def forward(self, queries, keys, values, attn_mask):
        attn = self.attn(queries, keys, values)
        return attn, None

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape  #[32,72,8,64]
        _, S, _, D = values.shape #[32,96,8,64]
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)  #[32,8,72,96]
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))#[32,8,72,96]
        V = torch.einsum("bhls,bshd->blhd", A, values)  #[32,72,8,64]
        
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False,local_casual=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.local_casual=local_casual
    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape   #32,8,96,64
        _, _, L_Q, _ = Q.shape     #96

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E) #32,8,96,96,64
        index_sample = torch.randint(L_K, (L_Q, sample_k)) # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]  #32,8,95,25,64
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()#32,8,96,25

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K) #[32,8,96]
        M_top = M.topk(n_top, sorted=False)[1]   #[32,8,25]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :] # factor*ln(L_q)  [32,8,25,64]  Q:[32,8,96,64]  Q_reduce[32,8,25,64]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) # factor*ln(L_q)*L_k   [32,8,25,96]

        return Q_K, M_top

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
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device) #32,8,72,[32,8,25],[32,8,25,72]
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1) # nn.Softmax(dim=-1)(scores)  #32,8,25,96

        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)   #32,8,96,64 attn [32,8,25,96]  V[32,8,96,64]
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V])/L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn   #attns 32,8,96,96
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape  #32,96,8,64
        _, L_K, _, _ = keys.shape     #96

        queries = queries.transpose(2,1)   #32,8,96,64
        keys = keys.transpose(2,1)
        values = values.transpose(2,1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() # c*ln(L_k)   #25
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item() # c*ln(L_q)    #25

        U_part = U_part if U_part<L_K else L_K
        u = u if u<L_Q else L_Q
        
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u) #[32,8,25,96 ]  [32,8,25]

        # add scale factor
        scale = self.scale or 1./sqrt(D)   #0.125
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)  #[32,8,96,64]
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)  #[32,8,96,64]
        
        return context.transpose(2,1).contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads,q_len,
                 d_keys=None, d_values=None, mix=False,local_casual=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention   #probattention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix


        self.local_casual=local_casual
        self.split_size=d_model#*n_heads
        self.q_len=q_len
        self.query_key= nn.Conv1d(d_model, d_keys * n_heads * 2,self.q_len).cuda() #q_len

    def forward(self, queries, keys, values, attn_mask):     #x, x, x, attn_mask
        B, L, _ = queries.shape   #32,96,512
        _, S, _ = keys.shape  #96
        H = self.n_heads

        if  self.local_casual:
            qk_x=nn.functional.pad(queries.permute(0, 2, 1), pad=(self.q_len-1, 0)).cuda()
            query_key=self.query_key(qk_x).cuda().permute(0, 2, 1) # [2,20,2096]  [2,131,20]  [32,96,1024]
            #[32,93,1024]  96 95 94 93
            #[32,45,1024]
            queries, keys=query_key.split(self.split_size,dim=2)  # [2,20,1048] [2,20,1048] [32,96,512] self.split_size=n_embd * self.n_head
            #print(query_key.size(),queries.size(),keys.size())
            queries = queries.view(B, L, H, -1)  # [32,96,8,64]
            keys = keys.view(B, S, H, -1)

        else:
            queries = self.query_projection(queries).view(B, L, H, -1)  #32,96,8,64 ... [32,48,8,60]
            keys = self.key_projection(keys).view(B, S, H, -1)   #32,96,8
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = out.transpose(2,1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
