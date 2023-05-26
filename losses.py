import torch


def get_aff_loss(logits, target):
    pos_label = (target == 1).int()
    pos_count = pos_label.sum() + 1
    neg_label = (target == 0).int()
    neg_count = neg_label.sum() + 1

    pos_loss = torch.sum(pos_label * (1 - logits)) / pos_count
    neg_loss = torch.sum(neg_label * (logits)) / neg_count
    loss = 0.5 * pos_loss + 0.5 * neg_loss
    return loss
