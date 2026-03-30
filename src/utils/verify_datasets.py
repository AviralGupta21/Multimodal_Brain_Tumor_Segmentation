import torch
import numpy as np
from unet_model import UNet
from brats_cv_dataset import get_fold_loaders

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SMOOTH = 1e-6


def evaluate_fold(year=2018, fold=0):

    print(f"\n🔎 Evaluating Year {year} Fold {fold+1}")

    model = UNet().to(DEVICE)
    ckpt = f"checkpoints/best_model_year{year}_fold{fold}.pth"
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()

    _, val_loader = get_fold_loaders(year, fold)

    dice_all = []
    slice_iou_all = []
    total_inter = np.zeros(3)
    total_union = np.zeros(3)

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            logits = model(x)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            for c in range(3):

                pred_c = preds[:, c]
                target_c = y[:, c]

                intersection = (pred_c * target_c).sum(dim=(1,2))
                union = pred_c.sum(dim=(1,2)) + target_c.sum(dim=(1,2)) - intersection

                dice = (2 * intersection + SMOOTH) / (
                    pred_c.sum(dim=(1,2)) + target_c.sum(dim=(1,2)) + SMOOTH
                )
                dice_all.append(dice.cpu().numpy())

                iou_slice = (intersection + SMOOTH) / (union + SMOOTH)
                slice_iou_all.append(iou_slice.cpu().numpy())

                total_inter[c] += intersection.sum().item()
                total_union[c] += union.sum().item()

    dice_all = np.concatenate(dice_all).reshape(-1,3).mean(axis=0)
    slice_iou = np.concatenate(slice_iou_all).reshape(-1,3).mean(axis=0)
    global_iou = (total_inter + SMOOTH) / (total_union + SMOOTH)

    dice_derived_iou = dice_all / (2 - dice_all)

    print("\n===== RESULTS =====")
    print("Dice (slice mean):       ", dice_all)
    print("Slice-mean IoU:          ", slice_iou)
    print("Global IoU:              ", global_iou)
    print("Dice-derived IoU:        ", dice_derived_iou)

    return dice_all, slice_iou, global_iou, dice_derived_iou


if __name__ == "__main__":
    evaluate_fold(2018, 0)