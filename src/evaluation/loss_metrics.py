import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

SMOOTH = 1e-6

def resize_gt(gt, pred):
    gt = F.interpolate(gt, size=pred.shape[2:], mode="nearest")
    return gt


class DiceLoss(nn.Module):
    def __init__(self, weights=[0.3, 0.3, 0.4]):
        super().__init__()
        self.weights = torch.tensor(weights)

    def forward(self, logits, targets):

        targets = resize_gt(targets, logits).float()
        probs = torch.sigmoid(logits.float())

        weights = self.weights.to(logits.device)

        dice_per_class = []

        for c in range(probs.shape[1]):

            p = probs[:, c]
            t = targets[:, c]

            intersection = (p * t).sum(dim=(1,2))
            union = p.sum(dim=(1,2)) + t.sum(dim=(1,2))

            dice = (2 * intersection + SMOOTH) / (union + SMOOTH)
            dice = dice.mean()

            dice_per_class.append(dice)

        dice_per_class = torch.stack(dice_per_class)

        weighted_dice = (weights * dice_per_class).sum()

        return 1 - weighted_dice

def dice_score(logits, targets):

    targets = resize_gt(targets, logits)
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    dices = []

    for c in range(preds.shape[1]):  
        pred_c = preds[:, c]
        target_c = targets[:, c]

        intersection = (pred_c * target_c).sum(dim=(1,2))
        union = pred_c.sum(dim=(1,2)) + target_c.sum(dim=(1,2))

        dice = (2 * intersection + SMOOTH) / (union + SMOOTH)
        dices.append(dice.mean().item())

    return dices 


def jaccard_score(logits, targets):

    targets = resize_gt(targets, logits)
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    jaccards = []

    for c in range(preds.shape[1]):
        pred_c = preds[:, c]
        target_c = targets[:, c]

        intersection = (pred_c * target_c).sum(dim=(1,2))
        union = pred_c.sum(dim=(1,2)) + target_c.sum(dim=(1,2)) - intersection

        jac = (intersection + SMOOTH) / (union + SMOOTH)
        jaccards.append(jac.mean().item())

    return jaccards

def paper_jaccard(logits, targets):

    targets = resize_gt(targets, logits)
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    paper_jaccs = []

    for c in range(preds.shape[1]):

        pred_c = preds[:, c]
        target_c = targets[:, c]

        TP = (pred_c * target_c).sum(dim=(1,2))
        FP = (pred_c * (1 - target_c)).sum(dim=(1,2))
        FN = ((1 - pred_c) * target_c).sum(dim=(1,2))

        # 2TP / (2TP + FP + FN)
        pj = (2 * TP + SMOOTH) / (2 * TP + FP + FN + SMOOTH)

        paper_jaccs.append(pj.mean().item())

    return paper_jaccs