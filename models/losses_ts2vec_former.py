import torch
from torch import nn
import torch.nn.functional as F


def hierarchical_contrastive_loss(z1, z2, alpha=0.5):
    loss=alpha*instance_contrastive_loss(z1, z2)+(1-alpha)*temporal_contrastive_loss(z1, z2)
    return loss


def instance_contrastive_loss(z1, z2):
    B, T=z1.size(0), z1.size(1)  # 32,78
    if B==1:
        return z1.new_tensor(0.)
    z=torch.cat([z1, z2], dim=0)  # 2B x T x C   [64,78,320]
    z=z.transpose(0, 1)  # T x 2B x C [78,64,320]
    sim=torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B  [78,64,64]
    logits=torch.tril(sim, diagonal=-1)[:, :, :-1]  # T x 2B x (2B-1) [78,64,63]
    logits+=torch.triu(sim, diagonal=1)[:, :, 1:]  # [1937,4,3]  [78,64,63]
    logits=-F.log_softmax(logits, dim=-1)  # [1937,4,3]   [78,64,63]

    i=torch.arange(B, device=z1.device)  # [0,1,....31]
    loss=(logits[:, i, B+i-1]+logits[:, B+i,i])/2  # 0~31 31~62  32~63  0~31 [78,32] [78,32]  [19,32] [9,32]
    return loss.transpose(0, 1)*(-1)


def temporal_contrastive_loss(z1, z2):
    B, T=z1.size(0), z1.size(1)
    if T==1:
        return z1.new_tensor(0.)
    z=torch.cat([z1, z2], dim=1)  # B x 2T x C [2,3874,320]
    sim=torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    logits=torch.tril(sim, diagonal=-1)[:, :, :-1]  # B x 2T x (2T-1)
    logits+=torch.triu(sim, diagonal=1)[:, :, 1:]
    logits=-F.log_softmax(logits, dim=-1)  # [2,3874,3873] [32,156,155]

    t=torch.arange(T, device=z1.device)
    loss=(logits[:, t, T+t-1]+logits[:, T+t, t])/2  # [32,78]  [32,19]
    return loss
