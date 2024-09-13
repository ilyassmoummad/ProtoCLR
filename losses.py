import torch
from torch import nn
from torch.nn import functional as F

class SupConLoss(nn.Module): # adapted from : https://github.com/HobbitLong/SupContrast/blob/master/losses.py
    """ It takes 2 features and labels, if labels is None it degenrates to to SimCLR """
    def __init__(self, tau=1.0):
        super().__init__()
        self.tau = tau

    def forward(self, z1, z2, labels=None):

        z1, z2 = F.normalize(z1), F.normalize(z2)
        features = torch.cat([z1.unsqueeze(1), z2.unsqueeze(1)], dim=1)
        batch_size = features.shape[0]

        if labels is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(features.device)
        else:
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(features.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        anchor_dot_contrast = torch.div(torch.matmul(contrast_feature, contrast_feature.T), self.tau)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(contrast_count, contrast_count)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size * contrast_count).view(-1, 1).to(features.device), 0)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - mean_log_prob_pos
        loss = loss.view(contrast_count, batch_size).mean()
        
        return loss

class ProtoCLRLoss(nn.Module):
    def __init__(self, tau):
        super().__init__()
        self.tau = tau

    def forward(self, z1, z2, y):

        y = torch.cat([y, y], dim=0)

        unique_ys = torch.unique(y)

        z1 = F.normalize(z1)
        z2 = F.normalize(z2)

        z = torch.cat([z1, z2], dim=0)

        label_mapping = {label.item(): i for i, label in enumerate(unique_ys)}
        mapped_y = torch.tensor([label_mapping.get(label.item(), -1) for label in y]).to(z.device)

        unique_labels, counts = torch.unique(mapped_y, return_counts=True)
        one_hot_labels = torch.nn.functional.one_hot(mapped_y, num_classes=len(unique_labels)).float()
        z_mean = torch.matmul(one_hot_labels.T, z) / counts.view(-1, 1)        

        sim = z.mm(z_mean.T)

        ce = F.cross_entropy(sim/self.tau, mapped_y)

        return ce