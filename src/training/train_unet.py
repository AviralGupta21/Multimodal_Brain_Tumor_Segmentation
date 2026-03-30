import os
import random
import numpy as np
import torch
from tqdm import tqdm
from brats_cv_dataset import get_fold_loaders
from BraTS.BraTS_python_files.unet_model import UNet
from loss_metrics import DiceLoss, dice_score, jaccard_score, paper_jaccard
import torch.nn as nn


def init_weights_kaiming(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_EPOCHS = 300
LR = 1e-4

os.makedirs("checkpoints", exist_ok=True)


def train_epoch(model, loader, optimizer, loss_fn):
    model.train()

    total_loss = 0
    dice_batches = []

    for x, y in tqdm(loader, leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        dice_batches.append(np.array(dice_score(logits, y)))

    dice_batches = np.stack(dice_batches)

    return total_loss / len(loader), dice_batches.mean(axis=0)

@torch.no_grad()
def eval_epoch(model, loader, loss_fn):
    model.eval()

    losses = []
    dice_batches = []
    iou_batches = []
    paper_batches = []

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        logits = model(x)
        loss = loss_fn(logits, y)

        losses.append(loss.item())
        dice_batches.append(np.array(dice_score(logits, y)))
        iou_batches.append(np.array(jaccard_score(logits, y)))
        paper_batches.append(np.array(paper_jaccard(logits, y)))

    dice_batches = np.stack(dice_batches)
    iou_batches = np.stack(iou_batches)
    paper_batches = np.stack(paper_batches)

    return (
        np.mean(losses),
        dice_batches.mean(axis=0),
        iou_batches.mean(axis=0),
        paper_batches.mean(axis=0),
    )

def train_one_fold(year=2018, fold=0):

    print(f"\nTraining BraTS{year} — Fold {fold+1}/5")

    train_loader, test_loader = get_fold_loaders(year, fold)

    model = UNet().to(DEVICE)
    model.apply(init_weights_kaiming)

    loss_fn = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=75, gamma=0.5)

    best_val_mean_dice = -1
    best_train_mean_dice = -1

    val_checkpoint_path = f"checkpoints/best_VAL_model_year{year}_fold{fold}.pth"
    train_checkpoint_path = f"checkpoints/best_TRAIN_model_year{year}_fold{fold}.pth"

    for epoch in range(MAX_EPOCHS):

        train_loss, dice_train = train_epoch(model, train_loader, optimizer, loss_fn)
        val_loss, dice_val, iou_val, paper_val = eval_epoch(model, test_loader, loss_fn)

        scheduler.step()

        mean_train_dice = np.mean(dice_train)
        mean_val_dice = np.mean(dice_val)

        if (epoch + 1) % 10 == 0 or epoch == 0:

            print(f"\nEpoch {epoch+1}/{MAX_EPOCHS}")

            print(f"""
Train Loss: {train_loss:.4f}
Train Dice → WT:{dice_train[0]:.3f} TC:{dice_train[1]:.3f} ET:{dice_train[2]:.3f}

Val Loss: {val_loss:.4f}
Val Dice → WT:{dice_val[0]:.3f} TC:{dice_val[1]:.3f} ET:{dice_val[2]:.3f}
Val True IoU → WT:{iou_val[0]:.3f} TC:{iou_val[1]:.3f} ET:{iou_val[2]:.3f}
Val Paper Jacc → WT:{paper_val[0]:.3f} TC:{paper_val[1]:.3f} ET:{paper_val[2]:.3f}
""")

        if mean_val_dice > best_val_mean_dice:
            best_val_mean_dice = mean_val_dice
            torch.save(model.state_dict(), val_checkpoint_path)
            print("Best VAL model saved!")

        if mean_train_dice > best_train_mean_dice:
            best_train_mean_dice = mean_train_dice
            torch.save(model.state_dict(), train_checkpoint_path)
            print("Best TRAIN model saved!")

    print("\n==============================================")
    print("Evaluating BEST VALIDATION-SELECTED MODEL")
    print("==============================================")

    model.load_state_dict(torch.load(val_checkpoint_path))
    model.eval()

    loss_train_val_model, dice_train_val_model, iou_train_val_model, paper_train_val_model = eval_epoch(model, train_loader, loss_fn)
    loss_test_val_model, dice_test_val_model, iou_test_val_model, paper_test_val_model = eval_epoch(model, test_loader, loss_fn)

    print(f"""
--- TRAIN SET (VAL-selected model) ---
Dice       → WT:{dice_train_val_model[0]:.3f} TC:{dice_train_val_model[1]:.3f} ET:{dice_train_val_model[2]:.3f}
True IoU   → WT:{iou_train_val_model[0]:.3f} TC:{iou_train_val_model[1]:.3f} ET:{iou_train_val_model[2]:.3f}
Paper Jacc → WT:{paper_train_val_model[0]:.3f} TC:{paper_train_val_model[1]:.3f} ET:{paper_train_val_model[2]:.3f}

--- TEST SET (VAL-selected model) ---
Dice       → WT:{dice_test_val_model[0]:.3f} TC:{dice_test_val_model[1]:.3f} ET:{dice_test_val_model[2]:.3f}
True IoU   → WT:{iou_test_val_model[0]:.3f} TC:{iou_test_val_model[1]:.3f} ET:{iou_test_val_model[2]:.3f}
Paper Jacc → WT:{paper_test_val_model[0]:.3f} TC:{paper_test_val_model[1]:.3f} ET:{paper_test_val_model[2]:.3f}
""")

    print("\n==============================================")
    print("Evaluating BEST TRAINING-SELECTED MODEL")
    print("==============================================")

    model.load_state_dict(torch.load(train_checkpoint_path))
    model.eval()

    loss_train_train_model, dice_train_train_model, iou_train_train_model, paper_train_train_model = eval_epoch(model, train_loader, loss_fn)
    loss_test_train_model, dice_test_train_model, iou_test_train_model, paper_test_train_model = eval_epoch(model, test_loader, loss_fn)

    print(f"""
--- TRAIN SET (TRAIN-selected model) ---
Dice       → WT:{dice_train_train_model[0]:.3f} TC:{dice_train_train_model[1]:.3f} ET:{dice_train_train_model[2]:.3f}
True IoU   → WT:{iou_train_train_model[0]:.3f} TC:{iou_train_train_model[1]:.3f} ET:{iou_train_train_model[2]:.3f}
Paper Jacc → WT:{paper_train_train_model[0]:.3f} TC:{paper_train_train_model[1]:.3f} ET:{paper_train_train_model[2]:.3f}

--- TEST SET (TRAIN-selected model) ---
Dice       → WT:{dice_test_train_model[0]:.3f} TC:{dice_test_train_model[1]:.3f} ET:{dice_test_train_model[2]:.3f}
True IoU   → WT:{iou_test_train_model[0]:.3f} TC:{iou_test_train_model[1]:.3f} ET:{iou_test_train_model[2]:.3f}
Paper Jacc → WT:{paper_test_train_model[0]:.3f} TC:{paper_test_train_model[1]:.3f} ET:{paper_test_train_model[2]:.3f}
""")

    return {
    "val_model": {
        "train_dice": dice_train_val_model,
        "test_dice": dice_test_val_model,
        "train_iou": iou_train_val_model,
        "test_iou": iou_test_val_model,
        "train_paper": paper_train_val_model,
        "test_paper": paper_test_val_model,
    },
    "train_model": {
        "train_dice": dice_train_train_model,
        "test_dice": dice_test_train_model,
        "train_iou": iou_train_train_model,
        "test_iou": iou_test_train_model,
        "train_paper": paper_train_train_model,
        "test_paper": paper_test_train_model,
    },
}

if __name__ == "__main__":
    train_one_fold(year = 2020, fold = 0)