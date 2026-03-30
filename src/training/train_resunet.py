import os
import random
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.amp import autocast, GradScaler
from brats_cv_dataset_resunet import get_fold_loaders
from resunet_model import ResUNetPlusStrict3Branch
from loss_metrics import DiceLoss, dice_score, jaccard_score, paper_jaccard

def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved  = torch.cuda.memory_reserved() / 1024**3
        max_alloc = torch.cuda.max_memory_allocated() / 1024**3

        print(f"\n[GPU MEMORY]")
        print(f"Allocated: {allocated:.2f} GB")
        print(f"Reserved : {reserved:.2f} GB")
        print(f"Max Used : {max_alloc:.2f} GB\n")


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_EPOCHS = 400  
LR = 1e-4

os.makedirs("checkpoints_resunet", exist_ok=True)

scaler = GradScaler()  

def init_weights_kaiming(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def train_epoch(model, loader, optimizer, loss_fn, epoch):
    model.train()
    total_loss = 0
    dice_batches = []

    for batch_idx, (x, y) in enumerate(tqdm(loader, leave=False)):

        try:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda"):
                logits = model(x)
                loss = loss_fn(logits, y)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            dice_batches.append(
                np.array(dice_score(logits.float(), y))
            )

        except RuntimeError as e:

            if "out of memory" in str(e).lower():
                print("\n" + "="*60)
                print("CUDA OUT OF MEMORY DETECTED DURING TRAINING")
                print(f"Epoch: {epoch+1}")
                print(f"Batch index: {batch_idx}")
                print(f"Batch size: {x.size(0)}")
                print(f"Input shape: {tuple(x.shape)}")
                print_gpu_memory()
                print("="*60)

                torch.cuda.empty_cache()
                scaler.update()
                return None, None  

            else:
                raise e

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

        with autocast("cuda"):
            logits = model(x)
            loss = loss_fn(logits, y)

        losses.append(loss.item())
        dice_batches.append(np.array(dice_score(logits.float(), y)))
        iou_batches.append(np.array(jaccard_score(logits.float(), y)))
        paper_batches.append(np.array(paper_jaccard(logits.float(), y)))

    return (
        np.mean(losses),
        np.mean(dice_batches, axis=0),
        np.mean(iou_batches, axis=0),
        np.mean(paper_batches, axis=0),
    )

def train_one_fold(year=2018, fold=0):

    print(f"\nTraining BraTS{year} — Fold {fold+1}/5")

    train_loader, test_loader = get_fold_loaders(year, fold)

    model = ResUNetPlusStrict3Branch(
        base_ch=64,
        n_classes=3
    ).to(DEVICE)

    model.apply(init_weights_kaiming)

    loss_fn = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=100, gamma=0.5
    )

    best_val_mean_dice = -1
    best_train_mean_dice = -1

    val_checkpoint_path = f"checkpoints_resunet/best_VAL_resunet_year{year}_fold{fold}.pth"
    train_checkpoint_path = f"checkpoints_resunet/best_TRAIN_resunet_year{year}_fold{fold}.pth"

    for epoch in range(MAX_EPOCHS):

        train_loss, dice_train = train_epoch(model, train_loader, optimizer, loss_fn, epoch)
        if train_loss is None:
            print("Training stopped due to CUDA OOM.")
            return
        
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

    print("\nTraining Completed.")

if __name__ == "__main__":
        
    train_one_fold(year=2018, fold=0)