import os
import torch
import numpy as np
from tqdm import tqdm
from unet_model import UNet
from brats_cv_dataset import get_fold_loaders
from loss_metrics import dice_score, jaccard_score, paper_jaccard

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_FOLDS = 5

def get_available_folds(year, model_type):
    folds = []

    for fold in range(5):
        if model_type == "VAL":
            ckpt = f"checkpoints/best_VAL_model_year{year}_fold{fold}.pth"
        else:
            ckpt = f"checkpoints/best_TRAIN_model_year{year}_fold{fold}.pth"

        if os.path.exists(ckpt):
            folds.append(fold)

    return folds

@torch.no_grad()
def evaluate_checkpoint(year, fold, model_type):

    if model_type == "VAL":
        ckpt = f"checkpoints/best_VAL_model_year{year}_fold{fold}.pth"
    else:
        ckpt = f"checkpoints/best_TRAIN_model_year{year}_fold{fold}.pth"

    print(f"\nEvaluating {model_type} model — Fold {fold+1}")

    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()

    train_loader, val_loader = get_fold_loaders(year, fold)

    def run(loader):

        dice_batches = []
        iou_batches = []
        paper_batches = []

        for x, y in tqdm(loader, leave = False):

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)

            dice_batches.append(np.array(dice_score(logits, y)))
            iou_batches.append(np.array(jaccard_score(logits, y)))
            paper_batches.append(np.array(paper_jaccard(logits, y)))

        dice = np.stack(dice_batches).mean(axis=0)
        iou = np.stack(iou_batches).mean(axis=0)
        paper = np.stack(paper_batches).mean(axis=0)

        return dice, iou, paper

    train_metrics = run(train_loader)
    val_metrics = run(val_loader)

    return train_metrics, val_metrics


def evaluate_all_folds(year=2018):

    results = {
        "VAL": [],
        "TRAIN": []
    }

    for model_type in ["VAL", "TRAIN"]:

        train_dices = []
        val_dices = []
        train_ious = []
        val_ious = []
        train_papers = []
        val_papers = []

        available_folds = get_available_folds(year, model_type)

        if len(available_folds) == 0:
             print(f"No checkpoints found for {model_type} model")
             continue

        print(f"\nUsing folds for {model_type}: {[f+1 for f in available_folds]}")

        for fold in available_folds:

            train_metrics, val_metrics = evaluate_checkpoint(year, fold, model_type)

            train_dice, train_iou, train_paper = train_metrics
            val_dice, val_iou, val_paper = val_metrics

            train_dices.append(train_dice)
            val_dices.append(val_dice)

            train_ious.append(train_iou)
            val_ious.append(val_iou)

            train_papers.append(train_paper)
            val_papers.append(val_paper)

        train_dices = np.mean(train_dices, axis=0)
        val_dices = np.mean(val_dices, axis=0)

        train_ious = np.mean(train_ious, axis=0)
        val_ious = np.mean(val_ious, axis=0)

        train_papers = np.mean(train_papers, axis=0)
        val_papers = np.mean(val_papers, axis=0)

        print("\n======================================")
        print(f"FINAL RESULTS — {model_type} SELECTED MODEL")
        print("======================================")

        print("\nTRAIN SET")
        print(f"Dice       → WT:{train_dices[0]:.3f} TC:{train_dices[1]:.3f} ET:{train_dices[2]:.3f}")
        print(f"True IoU   → WT:{train_ious[0]:.3f} TC:{train_ious[1]:.3f} ET:{train_ious[2]:.3f}")
        print(f"Paper Jacc → WT:{train_papers[0]:.3f} TC:{train_papers[1]:.3f} ET:{train_papers[2]:.3f}")

        print("\nVAL SET")
        print(f"Dice       → WT:{val_dices[0]:.3f} TC:{val_dices[1]:.3f} ET:{val_dices[2]:.3f}")
        print(f"True IoU   → WT:{val_ious[0]:.3f} TC:{val_ious[1]:.3f} ET:{val_ious[2]:.3f}")
        print(f"Paper Jacc → WT:{val_papers[0]:.3f} TC:{val_papers[1]:.3f} ET:{val_papers[2]:.3f}")

        print()


if __name__ == "__main__":

    evaluate_all_folds(2020)